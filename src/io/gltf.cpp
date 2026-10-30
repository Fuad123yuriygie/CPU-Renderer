// Static glTF adapter: validate the supported contract, load local resources,
// flatten active nodes into world space, and return only renderer-owned data.
#include "scene.h"
#include "file.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string_view>
#include <unordered_set>

// The pinned parser implementation is compiled in this translation unit only.
// Its diagnostics are isolated without weakening warnings for our adapter code.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define CGLTF_IMPLEMENTATION
#include "../../third_party/cgltf.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace cr::io {
namespace {
// Separate limits bound parser allocations, element counts, and decoded scene
// storage. Small compressed files can otherwise request very large allocations.
constexpr std::size_t maxElements = 16ull * 1024 * 1024;
constexpr std::size_t maxSceneBytes = 1024ull * 1024 * 1024;
constexpr std::size_t maxParseBytes = 256ull * 1024 * 1024;

// Prefix failures consistently so CLI diagnostics identify the asset adapter.
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("glTF: " + message); }

// Subtraction-based accounting avoids overflow while accumulating resource sizes.
void addBudget(std::size_t& used, std::size_t size, std::size_t limit, const std::string& label) {
    if (size > limit || used > limit - size) fail(label + " exceeds the resource allocation limit; simplify the asset");
    used += size;
}

// cgltf allocator callbacks cannot throw across the C interface. Store the size
// in an aligned prefix, enforce a live-allocation budget, and return null on failure.
struct ParseMemory {
    struct alignas(std::max_align_t) Header { std::size_t size; };
    std::size_t used = 0;
    static void* allocate(void* context, cgltf_size size) {
        auto& pool = *static_cast<ParseMemory*>(context);
        if (size > maxParseBytes || pool.used > maxParseBytes - size) return nullptr;
        auto* header = static_cast<Header*>(std::malloc(sizeof(Header) + size));
        if (!header) return nullptr;
        header->size = size;
        pool.used += size;
        return header + 1;
    }
    static void release(void* context, void* pointer) {
        if (!pointer) return;
        auto* header = static_cast<Header*>(pointer) - 1;
        static_cast<ParseMemory*>(context)->used -= header->size;
        std::free(header);
    }
};

// Shared hex decoder for JSON Unicode escapes and percent-encoded resource bytes.
int hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Decode parser-owned escaped strings into UTF-8, including surrogate pairs.
// Reject embedded NULs so a C-style filename/name cannot silently truncate.
std::string jsonString(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c != '\\') { out.push_back(c); continue; }
        if (++i >= text.size()) fail("Incomplete JSON string escape");
        c = text[i];
        if (c == 'u') {
            auto codeUnit = [&]() {
                unsigned code = 0;
                for (int j = 0; j < 4; ++j) {
                    if (++i >= text.size() || hex(text[i]) < 0) fail("Invalid Unicode escape");
                    code = code * 16 + static_cast<unsigned>(hex(text[i]));
                }
                return code;
            };
            unsigned code = codeUnit();
            if (code >= 0xd800 && code <= 0xdbff) {
                if (i + 2 >= text.size() || text[i + 1] != '\\' || text[i + 2] != 'u') fail("Unpaired Unicode surrogate");
                i += 2;
                const unsigned low = codeUnit();
                if (low < 0xdc00 || low > 0xdfff) fail("Unpaired Unicode surrogate");
                code = 0x10000 + (code - 0xd800) * 1024 + low - 0xdc00;
            } else if (code >= 0xdc00 && code <= 0xdfff) fail("Unpaired Unicode surrogate");
            if (code <= 0x7f) out.push_back(static_cast<char>(code));
            else {
                if (code > 0xffff) out.push_back(static_cast<char>(0xf0 | (code >> 18)));
                else if (code > 0x7ff) out.push_back(static_cast<char>(0xe0 | (code >> 12)));
                else out.push_back(static_cast<char>(0xc0 | (code >> 6)));
                if (code > 0xffff) out.push_back(static_cast<char>(0x80 | ((code >> 12) & 63)));
                if (code > 0x7ff) out.push_back(static_cast<char>(0x80 | ((code >> 6) & 63)));
                out.push_back(static_cast<char>(0x80 | (code & 63)));
            }
        } else {
            switch (c) {
            case '"': case '\\': case '/': out.push_back(c); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default: fail("Invalid JSON string escape");
            }
        }
    }
    if (out.find('\0') != std::string::npos) fail("NUL characters are not allowed in names or resource URIs");
    return out;
}

// Missing optional names use a caller-provided label rather than a null pointer.
std::string name(const char* value, const std::string& fallback = {}) {
    return value ? jsonString(value) : fallback;
}

// This explicit allowlist is the material/lighting contract implemented by the
// renderer; recognizing a glTF extension in cgltf alone does not implement it.
bool supported(std::string_view extension) {
    return extension == "KHR_texture_transform" || extension == "KHR_lights_punctual" ||
           extension == "KHR_materials_emissive_strength" || extension == "KHR_materials_transmission" ||
           extension == "KHR_materials_ior" || extension == "KHR_materials_volume";
}

// Optional descriptive metadata is harmless; unknown appearance or required
// extensions are not silently replaced with a different-looking material.
void checkExtension(const std::string& extension, bool required) {
    if (supported(extension) || (!required && extension == "KHR_xmp_json_ld")) return;
    fail(std::string(required ? "Unsupported required extension '" : "Unsupported appearance extension '") + extension +
         "'; export static, uncompressed metallic/roughness materials and bake unsupported effects into supported textures");
}

// Inspect extension dictionaries as well as extensionsUsed: cgltf intentionally discards unknown texture-info extensions.
std::size_t preflight(const std::vector<unsigned char>& bytes) {
    // Locate JSON in either a plain glTF document or the first GLB chunk.
    const char* json = reinterpret_cast<const char*>(bytes.data());
    std::size_t size = bytes.size();
    auto u32 = [&](std::size_t offset) {
        uint32_t value = 0;
        if (offset > bytes.size() || bytes.size() - offset < 4) fail("Truncated GLB header");
        std::memcpy(&value, bytes.data() + offset, 4);
        return value;
    };
    if (size >= 4 && u32(0) == 0x46546c67) {
        if (size < 20 || u32(4) != 2 || u32(8) != size || u32(16) != 0x4e4f534a)
            fail("Invalid GLB 2.0 header or declared length");
        size = u32(12);
        if (size > bytes.size() - 20 || size % 4) fail("Invalid GLB JSON chunk length");
        json += 20;
    }
    if (size > 64ull * 1024 * 1024) fail("JSON exceeds the 64 MiB limit; use external buffers or GLB");
    // Count before allocating token storage, and bound nesting before walking
    // dictionaries. The later full parse uses this validated token count too.
    jsmn_parser parser{};
    jsmn_init(&parser);
    const int count = jsmn_parse(&parser, json, size, nullptr, 0);
    if (count <= 0 || count > 2000000) fail("Invalid JSON or JSON token allocation limit exceeded");
    std::vector<jsmntok_t> tokens(static_cast<std::size_t>(count));
    jsmn_init(&parser);
    if (jsmn_parse(&parser, json, size, tokens.data(), tokens.size()) != count || tokens[0].type != JSMN_OBJECT)
        fail("Invalid JSON document");
    std::vector<unsigned short> depth(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].parent >= 0) {
            const auto parent = static_cast<std::size_t>(tokens[i].parent);
            if (parent >= i || depth[parent] >= 128) fail("JSON nesting exceeds the safe parsing limit");
            depth[i] = static_cast<unsigned short>(depth[parent] + 1);
        } else if (i != 0) fail("Multiple JSON root values");
    }
    auto text = [&](std::size_t i) {
        return jsonString(std::string_view(json + tokens[i].start, static_cast<std::size_t>(tokens[i].end - tokens[i].start)));
    };
    // Skip an entire token subtree using its byte extent, not recursive descent.
    auto after = [&](std::size_t i) {
        const auto end = tokens[i].end;
        do { ++i; } while (i < tokens.size() && tokens[i].start < end);
        return i;
    };
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
        // Ignore metadata payloads, then inspect both advertised extension lists
        // and actual extension dictionaries, even when the exporter omits a list.
        if (tokens[i].type != JSMN_STRING || tokens[i].size != 1) continue;
        const auto key = text(i);
        if (key == "extras" || key == "KHR_xmp_json_ld") { i = after(i + 1) - 1; continue; }
        if (key == "extensionsUsed" || key == "extensionsRequired") {
            if (tokens[i + 1].type != JSMN_ARRAY) fail(key + " must be an array");
            const auto end = after(i + 1);
            for (std::size_t j = i + 2; j < end; ++j) {
                if (tokens[j].type != JSMN_STRING) fail(key + " must contain extension names");
                checkExtension(text(j), key == "extensionsRequired");
            }
        } else if (key == "extensions") {
            if (tokens[i + 1].type != JSMN_OBJECT) fail("extensions must be an object");
            const auto end = after(i + 1);
            for (std::size_t j = i + 2; j < end; j = after(j + 1)) checkExtension(text(j), false);
        } else if (key == "alphaMode") {
            const auto mode = text(i + 1);
            if (mode != "OPAQUE" && mode != "MASK")
                fail("Unsupported alphaMode '" + mode + "'; use OPAQUE, alpha MASK, or KHR_materials_transmission for glass");
        }
    }
    return tokens.size();
}

// Percent decoding is also needed for binary data-URI payloads, where zero bytes
// are legal; the filesystem-path layer performs its stricter checks afterward.
std::string percentDecode(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%') result.push_back(value[i]);
        else {
            if (i + 2 >= value.size() || hex(value[i + 1]) < 0 || hex(value[i + 2]) < 0) fail("Malformed percent-escaped resource URI");
            result.push_back(static_cast<char>(hex(value[i + 1]) * 16 + hex(value[i + 2])));
            i += 2;
        }
    }
    return result;
}

// Embedded buffers/images share decoding but have different allowed MIME types.
// Validate base64 length, padding, and unused tail bits rather than accepting
// malformed input whose decoded size might differ from its declaration.
std::vector<unsigned char> dataUri(const std::string& uri, bool image) {
    const auto comma = uri.find(',');
    if (comma == std::string::npos) fail("Malformed data URI (missing comma)");
    const auto header = uri.substr(5, comma - 5);
    const auto semicolon = header.find(';');
    const auto mime = header.substr(0, semicolon);
    if (image && mime != "image/png" && mime != "image/jpeg") fail("Embedded textures must use image/png or image/jpeg");
    if (!image && !mime.empty() && mime != "application/octet-stream" && mime != "application/gltf-buffer")
        fail("Buffer data URI must use application/octet-stream or application/gltf-buffer");
    const auto payload = percentDecode(std::string_view(uri).substr(comma + 1));
    if (semicolon == std::string::npos) return {payload.begin(), payload.end()};
    if (header.substr(semicolon) != ";base64") fail("Unsupported data URI encoding; use base64");
    if (payload.empty() || payload.size() % 4) fail("Malformed base64 data URI length");
    auto digit = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<unsigned char> result;
    if (payload.size() / 4 > detail::maxFileBytes / 3) fail("Embedded resource exceeds the allocation limit");
    result.reserve(payload.size() / 4 * 3);
    for (std::size_t i = 0; i < payload.size(); i += 4) {
        const int a = digit(payload[i]), b = digit(payload[i + 1]);
        const bool pad2 = payload[i + 2] == '=', pad3 = payload[i + 3] == '=';
        const int c = pad2 ? 0 : digit(payload[i + 2]), d = pad3 ? 0 : digit(payload[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0 || (pad2 && !pad3) ||
            ((pad2 || pad3) && i + 4 != payload.size()) || (pad2 && (b & 15)) || (!pad2 && pad3 && (c & 3)))
            fail("Malformed base64 data URI payload");
        result.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
        if (!pad2) result.push_back(static_cast<unsigned char>((b << 4) | (c >> 2)));
        if (!pad3) result.push_back(static_cast<unsigned char>((c << 6) | d));
    }
    return result;
}

// Resolve files relative to the glTF directory, never the process cwd. Network
// schemes, remote file URIs, and UNC paths are deliberately outside the contract.
std::filesystem::path resourcePath(std::string uri, const std::filesystem::path& directory) {
    if (uri.find_first_of("?#") != std::string::npos) fail("Resource URI query/fragment is unsupported; percent-escape literal filename characters");
    if (uri.starts_with("file:")) {
        uri.erase(0, 5);
        if (uri.starts_with("//localhost/")) uri.erase(0, 11);
        else if (uri.starts_with("///")) uri.erase(0, 2);
        else if (uri.starts_with("//")) fail("Remote file URI is forbidden; resources must be local");
#ifdef _WIN32
        if (uri.size() >= 3 && uri[0] == '/' && uri[2] == ':') uri.erase(0, 1);
#endif
    } else {
        const auto colon = uri.find(':');
        if (colon != std::string::npos && !(colon == 1 && uri.size() > 2 && (uri[2] == '/' || uri[2] == '\\')))
            fail("Remote/unsupported resource URI scheme; only local files and data URIs are allowed");
    }
    uri = percentDecode(uri);
    if (uri.empty() || uri.find('\0') != std::string::npos || uri.starts_with("//") || uri.starts_with("\\\\"))
        fail("Empty, NUL-containing, or remote resource path");
    const auto path = std::filesystem::path(std::u8string(uri.begin(), uri.end()));
    return detail::canonicalInput(path.is_absolute() ? path : directory / path);
}

// Deduplicate canonical file identities for the CLI's output-overwrite protection.
void track(Scene& scene, const std::filesystem::path& path) {
    if (std::find(scene.resources.begin(), scene.resources.end(), path) == scene.resources.end()) scene.resources.push_back(path);
}

// Dispatch embedded bytes or a bounded local read; only external data adds another file.
std::vector<unsigned char> resource(const char* rawUri, const std::filesystem::path& directory, Scene& scene, bool image) {
    const auto uri = name(rawUri);
    if (uri.starts_with("data:")) return dataUri(uri, image);
    const auto path = resourcePath(uri, directory);
    track(scene, path);
    return detail::readFile(path);
}

// Shared numeric/layout guards keep invalid values out of rendering workers.
void range(float value, float lo, float hi, const std::string& label) {
    if (!std::isfinite(value) || value < lo || value > hi) fail(label + " is non-finite or outside its supported range");
}

void countLimit(std::size_t count, std::size_t limit, const char* label) {
    if (count > limit) fail(std::string(label) + " count exceeds the allocation limit");
}

// glTF indices are unsigned integer scalars, not normalized values or floats.
bool unsignedIndex(cgltf_component_type type) {
    return type == cgltf_component_type_r_8u || type == cgltf_component_type_r_16u || type == cgltf_component_type_r_32u;
}

// Prove the final strided element fits by division, avoiding count*stride overflow.
void checkSpan(const cgltf_buffer_view* view, std::size_t offset, std::size_t count,
               std::size_t stride, std::size_t element, const std::string& label) {
    if (!view || !count || !element || stride < element || offset > view->size || element > view->size - offset ||
        count - 1 > (view->size - offset - element) / stride)
        fail(label + ": accessor exceeds its bufferView; check byteOffset, count, and byteStride");
}

// Represent sparse replacement locations as a normal tightly packed accessor.
cgltf_accessor sparseIndices(const cgltf_accessor& accessor) {
    cgltf_accessor result{};
    result.type = cgltf_type_scalar;
    result.component_type = accessor.sparse.indices_component_type;
    result.count = accessor.sparse.count;
    result.buffer_view = accessor.sparse.indices_buffer_view;
    result.offset = accessor.sparse.indices_byte_offset;
    result.stride = cgltf_component_size(result.component_type);
    return result;
}

// Sparse replacement VALUES have their own packed layout, independent of the
// base accessor's interleaved stride. Preserve its component type/normalization.
cgltf_accessor sparseValues(const cgltf_accessor& accessor) {
    cgltf_accessor result = accessor;
    result.is_sparse = false;
    result.buffer_view = accessor.sparse.values_buffer_view;
    result.offset = accessor.sparse.values_byte_offset;
    result.count = accessor.sparse.count;
    result.stride = cgltf_calc_size(result.type, result.component_type);
    return result;
}

// Structural checks run before dereferencing buffer contents: counts, extents,
// alignments, primitive kinds, supported attributes, and a bounded static hierarchy.
void validateLayout(cgltf_data& data) {
    countLimit(data.nodes_count, 100000, "Node");
    countLimit(data.materials_count, 65536, "Material");
    countLimit(data.textures_count, 65536, "Texture");
    countLimit(data.images_count, 4096, "Image");
    countLimit(data.buffers_count, 4096, "Buffer");
    if (data.animations_count) fail("Animations are unsupported; export a static baked frame without animations");
    if (data.skins_count) fail("Skinning is unsupported; apply the armature and export a static mesh");
    if (data.variants_count) fail("Material variants are unsupported; export the chosen appearance");
    for (std::size_t i = 0; i < data.buffer_views_count; ++i) {
        const auto& view = data.buffer_views[i];
        if (view.has_meshopt_compression) fail("EXT_meshopt_compression is unsupported; export uncompressed geometry");
        if (!view.buffer || !view.size || view.offset > view.buffer->size || view.size > view.buffer->size - view.offset)
            fail("bufferView " + std::to_string(i) + " lies outside its buffer");
        if (view.stride && (view.stride < 4 || view.stride > 252 || view.stride % 4)) fail("Invalid bufferView byteStride");
    }
    for (std::size_t i = 0; i < data.accessors_count; ++i) {
        // Validate base and sparse spans separately; a missing base means implicit zeros.
        const auto& a = data.accessors[i];
        const std::string label = "Accessor " + std::to_string(i);
        countLimit(a.count, maxElements, "Accessor element");
        const auto element = cgltf_calc_size(a.type, a.component_type), component = cgltf_component_size(a.component_type);
        if (!a.count || !element || !component || a.stride < element || a.stride % component || a.offset % component)
            fail(label + ": invalid element type, count, stride or alignment");
        if (a.normalized && (a.component_type == cgltf_component_type_r_32f || a.component_type == cgltf_component_type_r_32u))
            fail(label + ": normalized FLOAT/UNSIGNED_INT is invalid");
        if (a.buffer_view) {
            checkSpan(a.buffer_view, a.offset, a.count, a.stride, element, label);
            if (a.buffer_view->offset % component) fail(label + ": bufferView has misaligned component data");
        } else if (a.offset) fail(label + ": byteOffset without bufferView");
        if (a.is_sparse) {
            const auto indices = sparseIndices(a), values = sparseValues(a);
            if (!unsignedIndex(indices.component_type) || a.sparse.count > a.count ||
                !indices.buffer_view || !values.buffer_view || indices.buffer_view->stride || values.buffer_view->stride)
                fail(label + ": invalid sparse accessor layout");
            checkSpan(indices.buffer_view, indices.offset, indices.count, indices.stride, indices.stride, label + " sparse indices");
            checkSpan(values.buffer_view, values.offset, values.count, values.stride, element, label + " sparse values");
            if ((indices.offset + indices.buffer_view->offset) % indices.stride ||
                (values.offset + values.buffer_view->offset) % component) fail(label + ": misaligned sparse data");
        }
    }
    for (std::size_t i = 0; i < data.meshes_count; ++i) {
        // Require authoring-time baking/triangulation instead of guessing how to
        // interpret skinning, shape keys, compression, variants, or vertex colors.
        const auto& mesh = data.meshes[i];
        if (mesh.weights_count || mesh.target_names_count) fail("Morph targets/weights are unsupported; apply shape keys before export");
        for (std::size_t j = 0; j < mesh.primitives_count; ++j) {
            const auto& p = mesh.primitives[j];
            if (p.type != cgltf_primitive_type_triangles) fail("Only TRIANGLES primitives are supported; triangulate meshes before export");
            if (p.targets_count) fail("Morph targets are unsupported; bake shape keys before export");
            if (p.has_draco_mesh_compression) fail("KHR_draco_mesh_compression is unsupported; export uncompressed geometry");
            if (p.mappings_count) fail("Material variants are unsupported");
            for (std::size_t k = 0; k < p.attributes_count; ++k) {
                const auto& a = p.attributes[k];
                if (a.type == cgltf_attribute_type_color) fail("Vertex COLOR attributes are unsupported; bake vertex colors into base-color textures");
                if (a.type == cgltf_attribute_type_joints || a.type == cgltf_attribute_type_weights) fail("Skinning attributes are unsupported; apply the armature before export");
                if (a.type == cgltf_attribute_type_invalid) fail("Unknown vertex attribute '" + name(a.name) + "'");
            }
        }
    }
    for (std::size_t i = 0; i < data.nodes_count; ++i) {
        // Matrix and TRS are alternative encodings. Reject non-affine/singular
        // hierarchy inputs before flattening camera, mesh, or light transforms.
        const auto& node = data.nodes[i];
        if (node.skin) fail("Skinning is unsupported");
        if (node.weights_count) fail("Morph weights are unsupported");
        if (node.has_mesh_gpu_instancing) fail("EXT_mesh_gpu_instancing is unsupported; realize instances before export");
        if (node.has_matrix && (node.has_translation || node.has_rotation || node.has_scale)) fail("Node specifies both matrix and TRS");
        if (node.has_rotation) {
            double norm = 0;
            for (float v : node.rotation) norm += static_cast<double>(v) * v;
            if (!std::isfinite(norm) || std::abs(norm - 1) > 0.002) fail("Node rotation must be a finite unit quaternion");
        }
        Mat4 local;
        cgltf_node_transform_local(&node, local.m.data());
        for (float value : local.m) if (!std::isfinite(value)) fail("Non-finite node transform");
        if (local.m[3] != 0 || local.m[7] != 0 || local.m[11] != 0 || local.m[15] != 1) fail("Node matrix is not affine");
        const cgltf_node* parent = node.parent;
        for (int depth = 0; parent; ++depth, parent = parent->parent)
            if (depth >= 128 || parent == &node) fail("Cyclic node hierarchy or hierarchy deeper than 128");
    }
    // Run cgltf's structural checks before loading buffers, then validate final sparse/index values ourselves.
    const auto valid = cgltf_validate(&data);
    if (valid != cgltf_result_success) fail("Invalid accessor, mesh, or scene hierarchy (cgltf validation " + std::to_string(valid) + ")");
}

// Decode an attribute into owned floats, then apply sparse replacements. No
// tightly packed assumption is made about the base vertex buffer's byteStride.
std::vector<float> floats(const cgltf_accessor* accessor, cgltf_type type, std::size_t count, const char* label) {
    if (!accessor) return {};
    if (accessor->type != type || accessor->count != count) fail(std::string(label) + " has the wrong accessor type or vertex count");
    const std::size_t components = cgltf_num_components(type);
    std::vector<float> result(count * components);
    auto base = *accessor;
    base.is_sparse = false;
    if (cgltf_accessor_unpack_floats(&base, result.data(), result.size()) != result.size()) fail(std::string("Cannot read ") + label);
    if (accessor->is_sparse) {
        // In pinned cgltf 1.15 the bulk sparse helper advances values by the BASE stride.
        // Read tightly packed sparse values separately, preserving strided base accessor support.
        const auto indices = sparseIndices(*accessor), values = sparseValues(*accessor);
        for (std::size_t i = 0; i < indices.count; ++i) {
            const auto index = cgltf_accessor_read_index(&indices, i);
            if (!cgltf_accessor_read_float(&values, i, result.data() + index * components, components)) fail("Cannot read sparse accessor values");
        }
    }
    for (float value : result) if (!std::isfinite(value)) fail(std::string(label) + " contains non-finite values");
    return result;
}

// Normalize indexed and nonindexed primitives into one uint32 triangle stream;
// sparse index accessors are patched just like sparse vertex attributes.
std::vector<uint32_t> indices(const cgltf_accessor* accessor, std::size_t vertexCount) {
    const std::size_t count = accessor ? accessor->count : vertexCount;
    if (count % 3) fail("TRIANGLES index/vertex count must be divisible by three");
    std::vector<uint32_t> result(count);
    if (!accessor) std::iota(result.begin(), result.end(), 0u);
    else {
        if (!unsignedIndex(accessor->component_type) || accessor->type != cgltf_type_scalar || accessor->normalized)
            fail("Indices must be non-normalized unsigned SCALAR accessors");
        auto base = *accessor;
        base.is_sparse = false;
        if (base.buffer_view && cgltf_accessor_unpack_indices(&base, result.data(), sizeof(uint32_t), count) != count) fail("Cannot read indices");
        if (accessor->is_sparse) {
            const auto sparse = sparseIndices(*accessor), values = sparseValues(*accessor);
            for (std::size_t i = 0; i < sparse.count; ++i)
                result[cgltf_accessor_read_index(&sparse, i)] = static_cast<uint32_t>(cgltf_accessor_read_index(&values, i));
        }
    }
    for (uint32_t index : result) if (index >= vertexCount) fail("Triangle index exceeds POSITION vertex count");
    return result;
}

// Convert parser pointers into stable engine indices and retain per-binding UV
// transforms. Color-space interpretation belongs to the material use, not the image.
TextureSlot slot(const cgltf_texture_view& view, const cgltf_data& data) {
    TextureSlot result;
    if (!view.texture) return result;
    result.texture = static_cast<int>(view.texture - data.textures);
    result.texcoord = view.texcoord;
    if (view.has_transform) {
        result.offset = {view.transform.offset[0], view.transform.offset[1]};
        result.scale = {view.transform.scale[0], view.transform.scale[1]};
        result.rotation = view.transform.rotation;
        if (view.transform.has_texcoord) result.texcoord = view.transform.texcoord;
    }
    if (result.texcoord < 0 || result.texcoord > 1) fail("Only TEXCOORD_0 and TEXCOORD_1 are supported; remap the material UV set");
    for (float value : {result.offset.x, result.offset.y, result.scale.x, result.scale.y, result.rotation})
        if (!std::isfinite(value)) fail("Non-finite KHR_texture_transform");
    return result;
}

// Translate supported metallic/roughness factors/maps and dielectric extensions.
// Unsupported appearance features fail explicitly instead of becoming plain diffuse.
Material material(const cgltf_material& source, const cgltf_data& data, std::size_t index) {
    Material result;
    result.name = name(source.name, "Material " + std::to_string(index));
    const std::string label = "Material '" + result.name + "': ";
    if (source.has_pbr_specular_glossiness || source.has_clearcoat || source.has_specular || source.has_sheen ||
        source.has_iridescence || source.has_diffuse_transmission || source.has_anisotropy || source.has_dispersion || source.unlit)
        fail(label + "unsupported material extension; bake unsupported appearance into metallic/roughness maps");
    if (source.alpha_mode == cgltf_alpha_mode_blend) fail(label + "alpha BLEND is unsupported; use MASK or transmission");
    if (source.occlusion_texture.texture && source.occlusion_texture.scale != 0)
        fail(label + "occlusionTexture is unsupported; remove the baked AO map (the path tracer computes geometric occlusion)");
    if (source.has_volume && source.volume.thickness_texture.texture)
        fail(label + "variable thicknessTexture is unsupported; use a closed mesh with constant positive thicknessFactor and actual world-space path attenuation");
    const auto& pbr = source.pbr_metallic_roughness;
    for (float value : pbr.base_color_factor) range(value, 0, 1, label + "baseColorFactor");
    result.baseColor = {pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2], pbr.base_color_factor[3]};
    range(pbr.metallic_factor, 0, 1, label + "metallicFactor");
    range(pbr.roughness_factor, 0, 1, label + "roughnessFactor");
    result.metallic = pbr.metallic_factor;
    result.roughness = pbr.roughness_factor;
    result.baseColorTexture = slot(pbr.base_color_texture, data);
    result.metallicRoughnessTexture = slot(pbr.metallic_roughness_texture, data);
    result.normalTexture = slot(source.normal_texture, data);
    result.emissiveTexture = slot(source.emissive_texture, data);
    if (source.normal_texture.texture) {
        range(source.normal_texture.scale, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), label + "normalTexture.scale");
        result.normalScale = source.normal_texture.scale;
    }
    for (float value : source.emissive_factor) range(value, 0, 1, label + "emissiveFactor");
    const float strength = source.has_emissive_strength ? source.emissive_strength.emissive_strength : 1;
    // Fold emissive strength into linear emission once, not at each shading event.
    range(strength, 0, std::numeric_limits<float>::max(), label + "emissiveStrength");
    result.emission = Vec3{source.emissive_factor[0], source.emissive_factor[1], source.emissive_factor[2]} * strength;
    if (!finite(result.emission)) fail(label + "emission overflows finite radiance");
    result.doubleSided = source.double_sided != 0;
    result.alphaMode = source.alpha_mode == cgltf_alpha_mode_mask ? AlphaMode::Mask : AlphaMode::Opaque;
    range(source.alpha_cutoff, 0, 1, label + "alphaCutoff (renderer supports [0,1])");
    result.alphaCutoff = source.alpha_cutoff;
    if (source.has_transmission) {
        range(source.transmission.transmission_factor, 0, 1, label + "transmissionFactor");
        result.transmission = source.transmission.transmission_factor;
        result.transmissionTexture = slot(source.transmission.transmission_texture, data);
    }
    if (source.has_ior) {
        range(source.ior.ior, 1, std::numeric_limits<float>::max(), label + "ior (use a finite dielectric IOR >= 1)");
        result.ior = source.ior.ior;
    }
    if (source.has_volume) {
        // Positive thickness marks a closed body; the integrator uses traveled
        // world-space distance for Beer absorption, not a thickness-map estimate.
        range(source.volume.thickness_factor, 0, std::numeric_limits<float>::max(), label + "thicknessFactor");
        result.thickness = source.volume.thickness_factor;
        for (float value : source.volume.attenuation_color) range(value, 0, 1, label + "attenuationColor");
        result.attenuationColor = {source.volume.attenuation_color[0], source.volume.attenuation_color[1], source.volume.attenuation_color[2]};
        const float distance = source.volume.attenuation_distance;
        if (!std::isfinite(distance) || distance <= 0) fail(label + "attenuationDistance must be finite and positive when specified");
        result.attenuationDistance = distance == std::numeric_limits<float>::max() ? infinity : distance;
    }
    return result;
}

// Keep parser enums out of engine-facing texture data.
Wrap wrap(cgltf_wrap_mode mode) {
    switch (mode) {
    case cgltf_wrap_mode_repeat: return Wrap::Repeat;
    case cgltf_wrap_mode_clamp_to_edge: return Wrap::Clamp;
    case cgltf_wrap_mode_mirrored_repeat: return Wrap::Mirror;
    default: fail("Invalid texture sampler wrap mode");
    }
}

// Column-major parent*local composition matches glTF's matrix convention.
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int col = 0; col < 4; ++col) for (int row = 0; row < 4; ++row) {
        float value = 0;
        for (int k = 0; k < 4; ++k) value += a.m[static_cast<std::size_t>(k * 4 + row)] * b.m[static_cast<std::size_t>(col * 4 + k)];
        result.m[static_cast<std::size_t>(col * 4 + row)] = value;
    }
    return result;
}

// Use double intermediates so a small, valid scale is not mistaken for singular.
double determinant(const Mat4& matrix) {
    const auto& m = matrix.m;
    return static_cast<double>(m[0]) * (static_cast<double>(m[5]) * m[10] - static_cast<double>(m[9]) * m[6]) +
           static_cast<double>(m[4]) * (static_cast<double>(m[9]) * m[2] - static_cast<double>(m[1]) * m[10]) +
           static_cast<double>(m[8]) * (static_cast<double>(m[1]) * m[6] - static_cast<double>(m[5]) * m[2]);
}

// Normalize transformed directions robustly before storing compact float vectors.
Vec3 unit(Vec3 vector) {
    const double magnitude = std::hypot(static_cast<double>(vector.x), static_cast<double>(vector.y), static_cast<double>(vector.z));
    if (!std::isfinite(magnitude) || magnitude == 0) fail("Zero or non-finite direction in node transform");
    return {static_cast<float>(vector.x / magnitude), static_cast<float>(vector.y / magnitude), static_cast<float>(vector.z / magnitude)};
}

// Realize one mesh instance into owned world-space triangles. Vertices are copied
// deliberately: renderer traversal does not depend on node graphs or parser lifetime.
void appendMesh(Scene& scene, const cgltf_mesh& mesh, const cgltf_data& data, const Mat4& world, std::size_t& budget) {
    const bool mirrored = determinant(world) < 0;
    for (std::size_t p = 0; p < mesh.primitives_count; ++p) {
        const auto& primitive = mesh.primitives[p];
        const auto* position = cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0);
        if (!position || position->component_type != cgltf_component_type_r_32f || position->normalized)
            fail("Mesh '" + name(mesh.name) + "' needs FLOAT VEC3 POSITION data; quantized geometry is unsupported");
        const std::size_t count = position->count;
        const auto vertices = floats(position, cgltf_type_vec3, count, "POSITION");
        const auto* normal = cgltf_find_accessor(&primitive, cgltf_attribute_type_normal, 0);
        const auto* tangent = cgltf_find_accessor(&primitive, cgltf_attribute_type_tangent, 0);
        if ((normal && normal->component_type != cgltf_component_type_r_32f) ||
            (tangent && tangent->component_type != cgltf_component_type_r_32f)) fail("NORMAL/TANGENT accessors must be FLOAT");
        const auto normals = floats(normal, cgltf_type_vec3, count, "NORMAL");
        const auto tangents = floats(tangent, cgltf_type_vec4, count, "TANGENT");
        std::vector<float> uv[2];
        for (int set = 0; set < 2; ++set) {
            const auto* a = cgltf_find_accessor(&primitive, cgltf_attribute_type_texcoord, set);
            if (a && !(a->component_type == cgltf_component_type_r_32f ||
                       ((a->component_type == cgltf_component_type_r_8u || a->component_type == cgltf_component_type_r_16u) && a->normalized)))
                fail("TEXCOORD must be FLOAT or normalized unsigned byte/short");
            uv[set] = floats(a, cgltf_type_vec2, count, "TEXCOORD");
        }
        const auto materialIndex = primitive.material ? static_cast<uint32_t>(primitive.material - data.materials) : static_cast<uint32_t>(data.materials_count);
        // Unassigned primitives share one standard glTF default material.
        if (materialIndex == scene.materials.size()) {
            Material fallback;
            fallback.name = "Default glTF material";
            scene.materials.push_back(fallback);
        }
        const auto& mat = scene.materials[materialIndex];
        for (const auto* map : {&mat.baseColorTexture, &mat.metallicRoughnessTexture, &mat.normalTexture, &mat.emissiveTexture, &mat.transmissionTexture})
            if (map->texture >= 0 && uv[map->texcoord].empty()) fail("Material '" + mat.name + "' references a missing TEXCOORD_" + std::to_string(map->texcoord));
        const auto index = indices(primitive.indices, count);
        // Account for flattened instances, not just unique input meshes.
        addBudget(budget, index.size() / 3 * sizeof(Triangle), maxSceneBytes, "Flattened scene");
        for (std::size_t i = 0; i < index.size(); i += 3) {
            Triangle triangle;
            triangle.material = materialIndex;
            for (int corner = 0; corner < 3; ++corner) {
                const auto v = static_cast<std::size_t>(index[i + static_cast<std::size_t>(corner)]);
                auto& out = triangle.v[corner];
                out.position = world.point({vertices[v * 3], vertices[v * 3 + 1], vertices[v * 3 + 2]});
                if (!finite(out.position)) fail("World-space vertex position overflows finite coordinates");
                if (!normals.empty()) {
                    // Normals need inverse-transpose transformation under nonuniform scale.
                    const Vec3 n{normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]};
                    if (!(lengthSquared(n) > 0) || !std::isfinite(lengthSquared(n))) fail("NORMAL accessor contains a zero or invalid normal");
                    out.normal = world.normal(n);
                }
                if (!tangents.empty()) {
                    // Tangents transform as directions; their handedness changes
                    // under mirroring so tangent-space normal maps keep their orientation.
                    const float sign = tangents[v * 4 + 3];
                    if (sign != 1 && sign != -1) fail("TANGENT.w must be +1 or -1");
                    const Vec3 t = unit(world.vector({tangents[v * 4], tangents[v * 4 + 1], tangents[v * 4 + 2]}));
                    out.tangent = {t.x, t.y, t.z, mirrored ? -sign : sign};
                }
                for (int set = 0; set < 2; ++set) if (!uv[set].empty()) out.uv[set] = {uv[set][v * 2], uv[set][v * 2 + 1]};
            }
            // A negative determinant reverses winding as well as tangent handedness.
            if (mirrored) std::swap(triangle.v[1], triangle.v[2]);
            scene.triangles.push_back(triangle);
        }
    }
}

// Import a uniquely named perspective camera looking down its local -Z axis.
// Sheared camera frames are rejected rather than silently changing the lens model.
void appendCamera(Scene& scene, const cgltf_node& node, const Mat4& world) {
    const auto& source = *node.camera;
    if (source.type != cgltf_camera_type_perspective) fail("Orthographic cameras are unsupported; export a perspective camera");
    Camera camera;
    camera.name = name(source.name, name(node.name, "Camera " + std::to_string(scene.cameras.size())));
    for (const auto& existing : scene.cameras) if (existing.name == camera.name) fail("Duplicate active camera name '" + camera.name + "'; name camera instances uniquely");
    camera.position = world.point({});
    camera.forward = unit(world.vector({0, 0, -1}));
    camera.right = unit(world.vector({1, 0, 0}));
    camera.up = unit(world.vector({0, 1, 0}));
    if (std::abs(dot(camera.forward, camera.up)) > 0.0001f || std::abs(dot(camera.forward, camera.right)) > 0.0001f ||
        std::abs(dot(camera.right, camera.up)) > 0.0001f) fail("Sheared camera transform; bake an orthonormal camera transform before export");
    const auto& p = source.data.perspective;
    range(p.yfov, 0.000001f, pi - 0.000001f, "Camera yfov");
    range(p.znear, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), "Camera znear");
    camera.yfov = p.yfov;
    camera.nearClip = p.znear;
    if (p.has_aspect_ratio) {
        range(p.aspect_ratio, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), "Camera aspectRatio");
        camera.aspect = p.aspect_ratio;
    }
    if (p.has_zfar) {
        if (!std::isfinite(p.zfar) || p.zfar <= p.znear) fail("Camera zfar must exceed znear");
        camera.farClip = p.zfar;
    }
    scene.cameras.push_back(camera);
}

// Flatten punctual light pose while retaining intensity/range/cone parameters.
// Their distance and angular falloff is evaluated later by the light sampler.
void appendLight(Scene& scene, const cgltf_light& source, const Mat4& world) {
    PunctualLight light;
    switch (source.type) {
    case cgltf_light_type_point: light.type = LightType::Point; break;
    case cgltf_light_type_directional: light.type = LightType::Directional; break;
    case cgltf_light_type_spot: light.type = LightType::Spot; break;
    default: fail("Unknown KHR_lights_punctual light type");
    }
    for (float c : source.color) range(c, 0, 1, "Punctual light color");
    range(source.intensity, 0, std::numeric_limits<float>::max(), "Punctual light intensity");
    range(source.range, 0, std::numeric_limits<float>::max(), "Punctual light range");
    light.position = world.point({});
    light.direction = unit(world.vector({0, 0, -1}));
    light.color = {source.color[0], source.color[1], source.color[2]};
    light.intensity = source.intensity;
    light.range = source.range > 0 ? source.range : infinity;
    if (light.type == LightType::Spot) {
        range(source.spot_inner_cone_angle, 0, pi / 2, "Spot innerConeAngle");
        range(source.spot_outer_cone_angle, 0.000001f, pi / 2, "Spot outerConeAngle");
        if (source.spot_inner_cone_angle >= source.spot_outer_cone_angle) fail("Spot innerConeAngle must be smaller than outerConeAngle");
        light.innerCone = source.spot_inner_cone_angle;
        light.outerCone = source.spot_outer_cone_angle;
    }
    scene.lights.push_back(light);
}
}

// Loading stages: preflight/parse, validate layouts, acquire bytes, decode assets,
// walk the active scene, and finalize geometry. All temporary parser state is RAII-owned.
Scene loadGltf(const std::filesystem::path& input) {
    try {
        const auto path = detail::canonicalInput(input);
        const auto bytes = detail::readFile(path);
        const auto tokenCount = preflight(bytes);
        ParseMemory pool;
        cgltf_options options{};
        options.json_token_count = tokenCount;
        options.memory = {&ParseMemory::allocate, &ParseMemory::release, &pool};
        cgltf_data* raw = nullptr;
        const auto result = cgltf_parse(&options, bytes.data(), bytes.size(), &raw);
        if (result != cgltf_result_success) fail("Cannot parse '" + detail::pathText(path) + "' (cgltf error " + std::to_string(result) + "); export valid static glTF 2.0");
        std::unique_ptr<cgltf_data, decltype(&cgltf_free)> data(raw, &cgltf_free);
        if (name(data->asset.version) != "2.0" || (data->asset.min_version && name(data->asset.min_version) != "2.0")) fail("Only glTF version 2.0 is supported");
        validateLayout(*data);
        Scene scene;
        track(scene, path);
        std::size_t bufferBudget = 0, sceneBudget = 0;
        std::vector<std::vector<unsigned char>> buffers(data->buffers_count);
        // Buffers either borrow the GLB BIN chunk or our explicitly owned byte arrays.
        for (std::size_t i = 0; i < data->buffers_count; ++i) {
            auto& buffer = data->buffers[i];
            if (!buffer.size) fail("Buffer byteLength must be positive");
            addBudget(bufferBudget, buffer.size, detail::maxFileBytes, "Combined buffers");
            if (buffer.uri) {
                buffers[i] = resource(buffer.uri, path.parent_path(), scene, false);
                if (buffers[i].size() < buffer.size) fail("Buffer " + std::to_string(i) + " is shorter than its declared byteLength");
                addBudget(bufferBudget, buffers[i].size() - buffer.size, detail::maxFileBytes, "Combined buffers including trailing bytes");
                buffer.data = buffers[i].data();
            } else {
                if (i != 0 || !data->bin || data->bin_size < buffer.size || data->bin_size - buffer.size > 3)
                    fail("Buffer has no URI or matching GLB BIN chunk");
                buffer.data = const_cast<void*>(data->bin);
            }
            buffer.data_free_method = cgltf_data_free_method_none;
        }
        // Only after loading bytes can replacement index values be checked for
        // strict order/range; later scatter operations rely on these checks.
        for (std::size_t i = 0; i < data->accessors_count; ++i) if (data->accessors[i].is_sparse) {
            const auto& accessor = data->accessors[i];
            const auto sparse = sparseIndices(accessor);
            std::size_t previous = 0;
            for (std::size_t j = 0; j < sparse.count; ++j) {
                const auto index = cgltf_accessor_read_index(&sparse, j);
                if (index >= accessor.count || (j && index <= previous)) fail("Sparse accessor indices must be in range and strictly increasing");
                previous = index;
            }
        }
        for (std::size_t i = 0; i < data->materials_count; ++i) scene.materials.push_back(material(data->materials[i], *data, i));
        for (std::size_t i = 0; i < data->images_count; ++i) {
            // External, data-URI, and embedded bufferView images share one decoder.
            const auto& source = data->images[i];
            const auto mime = name(source.mime_type);
            if (!mime.empty() && mime != "image/png" && mime != "image/jpeg") fail("Unsupported image MIME type '" + mime + "'; use PNG or JPEG");
            const std::string label = "Image " + std::to_string(i) + " '" + name(source.name) + "' in " + detail::pathText(path);
            Image image;
            if (source.uri && !source.buffer_view) {
                const auto encoded = resource(source.uri, path.parent_path(), scene, true);
                image = decodeImage(encoded.data(), encoded.size(), label);
            } else if (source.buffer_view && !source.uri) {
                if (mime.empty()) fail(label + ": embedded bufferView image needs image/png or image/jpeg mimeType");
                image = decodeImage(cgltf_buffer_view_data(source.buffer_view), source.buffer_view->size, label);
            } else fail(label + ": image must have exactly one URI or bufferView");
            addBudget(sceneBudget, image.pixels.size() * sizeof(Vec4), maxSceneBytes, "Decoded images and triangles");
            scene.images.push_back(std::move(image));
        }
        for (std::size_t i = 0; i < data->textures_count; ++i) {
            // Preserve base-level sampler behavior and warn explicitly about LOD
            // features that this renderer does not implement.
            const auto& source = data->textures[i];
            if (source.has_basisu || source.has_webp) fail("Compressed/BasisU/WebP textures are unsupported; export PNG or JPEG");
            if (!source.image) fail("Texture has no source image");
            Texture texture;
            texture.image = static_cast<int>(source.image - data->images);
            if (source.sampler) {
                const auto& sampler = *source.sampler;
                texture.wrapS = wrap(sampler.wrap_s);
                texture.wrapT = wrap(sampler.wrap_t);
                if (sampler.mag_filter != cgltf_filter_type_undefined && sampler.mag_filter != cgltf_filter_type_nearest && sampler.mag_filter != cgltf_filter_type_linear)
                    fail("Invalid texture magFilter");
                if (sampler.min_filter != cgltf_filter_type_undefined && sampler.min_filter != cgltf_filter_type_nearest &&
                    sampler.min_filter != cgltf_filter_type_linear && sampler.min_filter != cgltf_filter_type_nearest_mipmap_nearest &&
                    sampler.min_filter != cgltf_filter_type_linear_mipmap_nearest && sampler.min_filter != cgltf_filter_type_nearest_mipmap_linear &&
                    sampler.min_filter != cgltf_filter_type_linear_mipmap_linear) fail("Invalid texture minFilter");
                const auto filter = sampler.mag_filter != cgltf_filter_type_undefined ? sampler.mag_filter : sampler.min_filter;
                switch (filter) {
                case cgltf_filter_type_nearest: case cgltf_filter_type_nearest_mipmap_nearest: case cgltf_filter_type_nearest_mipmap_linear: texture.linear = false; break;
                case cgltf_filter_type_undefined: case cgltf_filter_type_linear: case cgltf_filter_type_linear_mipmap_nearest: case cgltf_filter_type_linear_mipmap_linear: texture.linear = true; break;
                default: fail("Invalid texture minFilter");
                }
                if (sampler.min_filter >= cgltf_filter_type_nearest_mipmap_nearest)
                    scene.warnings.push_back("Texture " + std::to_string(i) + ": mipmap LOD is unavailable; using base-level nearest/bilinear filtering");
                else if (sampler.mag_filter && sampler.min_filter && sampler.mag_filter != sampler.min_filter)
                    scene.warnings.push_back("Texture " + std::to_string(i) + ": no ray-differential LOD; using magFilter for both minification and magnification");
            }
            scene.textures.push_back(texture);
        }
        struct Pending { const cgltf_node* node; Mat4 parent; };
        // Traverse only the selected scene, preserving node order with an explicit
        // stack. A node may contribute geometry, a camera, and/or a punctual light.
        std::vector<Pending> pending;
        const auto* active = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
        if (active) {
            for (std::size_t i = active->nodes_count; i > 0; --i) pending.push_back({active->nodes[i - 1], {}});
            if (!data->scene && data->scenes_count > 1) scene.warnings.push_back("No default scene selected; loading the first scene only");
        } else {
            for (std::size_t i = data->nodes_count; i > 0; --i) if (!data->nodes[i - 1].parent) pending.push_back({&data->nodes[i - 1], {}});
        }
        std::vector<bool> visited(data->nodes_count);
        while (!pending.empty()) {
            const auto current = pending.back();
            pending.pop_back();
            const auto& node = *current.node;
            const auto index = static_cast<std::size_t>(current.node - data->nodes);
            if (visited[index]) fail("Active scene repeats a node or contains a cyclic hierarchy");
            visited[index] = true;
            Mat4 local;
            cgltf_node_transform_local(&node, local.m.data());
            const Mat4 world = multiply(current.parent, local);
            for (float value : world.m) if (!std::isfinite(value)) fail("World node transform is non-finite");
            const double det = determinant(world);
            if (!std::isfinite(det) || det == 0) fail("Singular or overflowing node transform for '" + name(node.name) + "'; apply a nonzero scale");
            if (node.mesh) appendMesh(scene, *node.mesh, *data, world, sceneBudget);
            if (node.camera) appendCamera(scene, node, world);
            if (node.light) appendLight(scene, *node.light, world);
            for (std::size_t i = node.children_count; i > 0; --i) pending.push_back({node.children[i - 1], world});
        }
        if (scene.triangles.empty()) fail("Active scene has no triangle geometry; check the selected scene and export visibility");
        // Finalization supplies missing frames, validates engine references, and
        // removes diagnosed degenerates before the parser and buffers are released.
        scene.finalize();
        if (scene.triangles.empty()) fail("Active scene contains only degenerate triangles");
        return scene;
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("glTF allocation failed while loading '" + detail::pathText(input) + "'; reduce mesh/image sizes");
    }
}
}
