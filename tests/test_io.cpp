// Self-contained adapter tests build tiny binary/JSON/image fixtures at runtime,
// then compare the two shipped Blender export forms when they are available.
#include "scene.h"
#include "../third_party/stb_image_write.h"

#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace cr;
namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;
int checks = 0;

// Exception-backed assertions remain enabled in Release as well as Debug.
void require(bool condition, const std::string& label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}
void near(float a, float b, float epsilon, const std::string& label) {
    require(std::isfinite(a) && std::abs(a - b) <= epsilon, label + ": " + std::to_string(a) + " != " + std::to_string(b));
}
void near(Vec3 a, Vec3 b, float epsilon, const std::string& label) {
    near(a.x, b.x, epsilon, label + ".x"); near(a.y, b.y, epsilon, label + ".y"); near(a.z, b.z, epsilon, label + ".z");
}
// A negative test must reject with the expected diagnostic, not just fail somehow.
template<class Function> void rejects(Function action, const std::string& fragment) {
    ++checks;
    try { action(); }
    catch (const std::exception& error) {
        if (std::string(error.what()).find(fragment) != std::string::npos) return;
        throw std::runtime_error("Expected diagnostic containing '" + fragment + "', got: " + error.what());
    }
    throw std::runtime_error("Expected failure: " + fragment);
}

// Keep generated tests under the current build directory, never reuse an existing
// folder, and restore cwd before cleanup even if a test throws.
struct TemporaryDirectory {
    fs::path path, previous;
    TemporaryDirectory() : previous(fs::current_path()) {
        path = previous / ("CPUrenderer io tests " + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        if (!fs::create_directory(path)) throw std::runtime_error("Test directory already exists; refusing to reuse it");
        fs::create_directories(path / "textures");
        fs::create_directories(path / "unrelated cwd");
    }
    ~TemporaryDirectory() {
        std::error_code ec;
        fs::current_path(previous, ec);
        fs::remove_all(path, ec);
    }
};

// Fixture byte I/O is intentionally independent of the adapter being tested.
void write(const fs::path& path, const Bytes& bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Test fixture write failed");
}
void write(const fs::path& path, const std::string& text) { write(path, Bytes(text.begin(), text.end())); }
Bytes read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Test fixture read failed");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
// Binary fixtures are encoded explicitly little-endian, with glTF's 4-byte padding.
void u32(Bytes& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<unsigned char>((value >> (8 * i)) & 255));
}
void f32(Bytes& out, float value) { u32(out, std::bit_cast<uint32_t>(value)); }
void align4(Bytes& out) { while (out.size() % 4) out.push_back(0); }
void append(Bytes& out, const Bytes& tail) { out.insert(out.end(), tail.begin(), tail.end()); }
// Independent data-URI encoder supplies embedded buffers and images to the loader.
std::string base64(const Bytes& bytes) {
    constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t value = (static_cast<uint32_t>(bytes[i]) << 16) |
                               (i + 1 < bytes.size() ? static_cast<uint32_t>(bytes[i + 1]) << 8 : 0) |
                               (i + 2 < bytes.size() ? bytes[i + 2] : 0);
        out += table[(value >> 18) & 63]; out += table[(value >> 12) & 63];
        out += i + 1 < bytes.size() ? table[(value >> 6) & 63] : '=';
        out += i + 2 < bytes.size() ? table[value & 63] : '=';
    }
    return out;
}
// Mutate a known fixture fragment deliberately; a missing fragment is a test bug.
void change(std::string& text, const std::string& from, const std::string& to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("Test fixture replacement not found: " + from);
    text.replace(at, from.size(), to);
}
// Codec callback collects test PNG/JPEG bytes without filesystem/path dependencies.
void collect(void* context, void* bytes, int size) {
    auto& out = *static_cast<Bytes*>(context);
    const auto* begin = static_cast<unsigned char*>(bytes);
    out.insert(out.end(), begin, begin + size);
}
// Asymmetric colors and alpha reveal row flips, channel mistakes, and gamma errors.
Bytes png() {
    const unsigned char pixels[]{128,0,0,255, 0,255,0,128, 0,0,255,0, 255,255,255,255};
    Bytes result;
    require(stbi_write_png_to_func(&collect, &result, 2, 2, 4, pixels, 8) != 0, "Fixture PNG encoding");
    return result;
}
// Interleave position/normal/tangent/two UV sets so assuming packed attributes fails.
Bytes meshBytes() {
    Bytes bytes;
    const float s = std::sqrt(0.5f);
    for (int i = 0; i < 3; ++i) {
        for (float v : {i == 1 ? 1.0f : 0.0f, i == 2 ? 1.0f : 0.0f, 0.0f, s, 0.0f, s,
                        s, 0.0f, -s, 1.0f, i == 1 ? 0.8f : 0.2f, i == 2 ? 0.9f : 0.1f,
                        i == 1 ? 0.7f : 0.3f, i == 2 ? 0.8f : 0.2f}) f32(bytes, v);
    }
    for (int v : {0,0,1,0,2,0}) bytes.push_back(static_cast<unsigned char>(v));
    return bytes;
}

// Reusable tiny scene with nested/mirrored transforms, every supported material
// extension, a camera/light, and an inactive second scene that must not be imported.
std::string document(const std::string& bufferUri, std::size_t bufferSize, const std::string& image,
                     const std::string& extraViews = {}) {
    return R"({"asset":{"version":"2.0"},"buffers":[{"byteLength":)" + std::to_string(bufferSize) + bufferUri +
        R"(}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":168,"byteStride":56},{"buffer":0,"byteOffset":168,"byteLength":6})" + extraViews +
        R"(],"accessors":[{"bufferView":0,"byteOffset":0,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":0,"byteOffset":12,"componentType":5126,"count":3,"type":"VEC3"},
        {"bufferView":0,"byteOffset":24,"componentType":5126,"count":3,"type":"VEC4"},
        {"bufferView":0,"byteOffset":40,"componentType":5126,"count":3,"type":"VEC2"},
        {"bufferView":0,"byteOffset":48,"componentType":5126,"count":3,"type":"VEC2"},
        {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
        "images":[)" + image + R"(],"samplers":[{"magFilter":9728,"minFilter":9728,"wrapS":33071,"wrapT":33648}],
        "textures":[{"source":0,"sampler":0}],
        "materials":[{"name":"Test material","pbrMetallicRoughness":{"baseColorFactor":[0.8,0.7,0.6,0.5],"metallicFactor":0.2,"roughnessFactor":0.4,
        "baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"offset":[0.1,0.2],"scale":[1.6,1.2],"rotation":0.31,"texCoord":1}}},
        "metallicRoughnessTexture":{"index":0}},"normalTexture":{"index":0,"scale":0.35},"emissiveTexture":{"index":0},"emissiveFactor":[0.1,0.2,0.3],
        "alphaMode":"MASK","alphaCutoff":0.42,"doubleSided":true,
        "extensions":{"KHR_materials_transmission":{"transmissionFactor":0.8,"transmissionTexture":{"index":0}},
        "KHR_materials_ior":{"ior":1.333},"KHR_materials_volume":{"thicknessFactor":0.8,"attenuationColor":[0.2,0.72,0.81],"attenuationDistance":3},
        "KHR_materials_emissive_strength":{"emissiveStrength":6}}}],
        "meshes":[{"name":"Triangle","primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TANGENT":2,"TEXCOORD_0":3,"TEXCOORD_1":4},"indices":5,"material":0}]}],
        "nodes":[{"translation":[10,20,30],"children":[1,2,3]},{"mesh":0,"scale":[-2,3,4]},
        {"camera":0,"translation":[0,0,5]},{"extensions":{"KHR_lights_punctual":{"light":0}}},{"mesh":0,"translation":[10000,0,0]}],
        "cameras":[{"name":"Reference","type":"perspective","perspective":{"yfov":0.7,"znear":0.05,"zfar":200,"aspectRatio":1.7777778}}],
        "scenes":[{"nodes":[0]},{"nodes":[4]}],"scene":0,
        "extensionsUsed":["KHR_texture_transform","KHR_lights_punctual","KHR_materials_transmission","KHR_materials_ior","KHR_materials_volume","KHR_materials_emissive_strength"],
        "extensions":{"KHR_lights_punctual":{"lights":[{"type":"spot","color":[1,0.8,0.6],"intensity":70,"range":25,"spot":{"innerConeAngle":0.1,"outerConeAngle":0.5}}]}}})";
}

// Package the same scene as GLB: aligned JSON and BIN chunks with explicit headers.
Bytes glb(std::string json, Bytes binary) {
    while (json.size() % 4) json += ' ';
    align4(binary);
    Bytes result;
    u32(result, 0x46546c67); u32(result, 2); u32(result, static_cast<uint32_t>(28 + json.size() + binary.size()));
    u32(result, static_cast<uint32_t>(json.size())); u32(result, 0x4e4f534a);
    result.insert(result.end(), json.begin(), json.end());
    u32(result, static_cast<uint32_t>(binary.size())); u32(result, 0x004e4942); append(result, binary);
    return result;
}

// One set of expected engine values applies to external, data-URI, and GLB forms.
// It checks ownership/paths as well as visual semantics such as winding and TBN.
void loadedContract(const Scene& scene) {
    require(scene.triangles.size() == 1, "Only active-scene geometry is flattened");
    require(scene.materials.size() == 1 && scene.images.size() == 1 && scene.textures.size() == 1, "Owned material/image/texture counts");
    const auto& t = scene.triangles[0];
    near(t.v[0].position, {10,20,30}, 1e-5f, "Parent translation");
    near(t.v[1].position, {10,23,30}, 1e-5f, "Mirrored winding vertex 1");
    near(t.v[2].position, {8,20,30}, 1e-5f, "Mirrored winding vertex 2");
    near(t.geometricNormal, {0,0,1}, 1e-5f, "Mirrored outward winding");
    near(t.v[0].normal, normalize(Vec3{-0.5f,0,0.25f}), 1e-5f, "Inverse-transpose nonuniform normal");
    near(t.v[0].tangent.w, -1, 0, "Mirrored tangent parity");
    near(t.v[0].uv[0].y, 0.1f, 1e-6f, "UV0 is not flipped");
    near(t.v[1].uv[1].y, 0.8f, 1e-6f, "UV1 follows swapped vertex");
    require(scene.textures[0].wrapS == Wrap::Clamp && scene.textures[0].wrapT == Wrap::Mirror && !scene.textures[0].linear, "Sampler semantics");
    const auto& m = scene.materials[0];
    near(m.metallic, 0.2f, 1e-6f, "Metallic"); near(m.roughness, 0.4f, 1e-6f, "Roughness");
    near(m.emission, {0.6f,1.2f,1.8f}, 1e-5f, "Emissive strength");
    near(m.transmission, 0.8f, 1e-6f, "Transmission"); require(m.transmissionTexture.texture == 0, "Transmission texture");
    near(m.ior, 1.333f, 1e-6f, "IOR"); near(m.thickness, 0.8f, 1e-6f, "Closed-volume marker");
    near(m.attenuationDistance, 3, 0, "Actual-path attenuation distance"); near(m.attenuationColor, {0.2f,0.72f,0.81f}, 1e-6f, "Attenuation color");
    near(m.normalScale, 0.35f, 1e-6f, "Normal map scale");
    require(m.alphaMode == AlphaMode::Mask && m.doubleSided, "MASK/double sided");
    require(m.baseColorTexture.texcoord == 1, "Texture transform UV override");
    near(m.baseColorTexture.rotation, 0.31f, 1e-6f, "Texture transform rotation");
    near(m.baseColorTexture.offset.x, 0.1f, 1e-6f, "Texture transform offset");
    near(m.baseColorTexture.scale.x, 1.6f, 1e-6f, "Texture transform scale");
    require(scene.cameras.size() == 1 && scene.cameras[0].name == "Reference", "Named active perspective camera");
    near(scene.cameras[0].position, {10,20,35}, 1e-6f, "World camera position");
    near(scene.cameras[0].forward, {0,0,-1}, 1e-6f, "Camera forward");
    require(scene.lights.size() == 1 && scene.lights[0].type == LightType::Spot, "Punctual spot import");
    near(scene.lights[0].position, {10,20,30}, 1e-6f, "World light position");
    near(scene.lights[0].intensity, 70, 0, "Punctual intensity");
    near(scene.lights[0].innerCone, 0.1f, 1e-6f, "Spot cone");
    TextureSlot raw; raw.texture = 0;
    const auto top = scene.sample(raw, {0.25f,0.25f}, {}, false);
    const auto bottom = scene.sample(raw, {0.25f,0.75f}, {}, false);
    near(top.x, 128 / 255.0f, 1e-6f, "Top-left texel orientation");
    near(bottom.z, 1, 0, "Bottom-left texel orientation"); near(bottom.w, 0, 0, "Texture alpha preserved");
    near(scene.sample(raw, {0.25f,0.25f}, {}, true).x, srgbToLinear(128 / 255.0f), 1e-6f, "sRGB applied per slot, not during decode");
    for (const auto& path : scene.resources) require(path.is_absolute() && fs::is_regular_file(path), "Canonical input resource path");
}

// Check encoded texture channels separately from linear HDR, and verify that only
// PNG output receives exposure/filmic/sRGB conversion. Bad buffers/writes must fail.
void imageTests(const fs::path& directory) {
    const auto raw = png();
    const auto decoded = io::decodeImage(raw.data(), raw.size(), "raw PNG");
    require(decoded.width == 2 && decoded.height == 2 && decoded.pixels.size() == 4, "Memory PNG decode dimensions");
    near(decoded.pixels[1].w, 128 / 255.0f, 1e-6f, "Memory PNG alpha");
    const auto unicode = directory / fs::path(u8"image \u00e9 \U00020000.hdr");
    std::vector<Vec3> linear{{0,0,0},{0.18f,0.18f,0.18f},{1,2,4},{3.75f,0.25f,1}};
    io::writeHdr(unicode, 2, 2, linear);
    const auto hdr = io::loadImage(unicode, true);
    require(hdr.width == 2 && hdr.height == 2, "Unicode HDR dimensions");
    for (std::size_t i = 0; i < linear.size(); ++i) near(Vec3{hdr.pixels[i].x,hdr.pixels[i].y,hdr.pixels[i].z}, linear[i], 0.016f, "Raw unexposed HDR roundtrip");
    io::writePng(directory / "exposure zero.png", 2, 2, linear, 0);
    io::writePng(directory / "exposure one.png", 2, 2, linear, 1);
    const auto ldr = io::loadImage(directory / "exposure zero.png");
    const auto exposed = io::loadImage(directory / "exposure one.png");
    near(ldr.pixels[2].x, 232 / 255.0f, 1 / 255.0f, "ACES-like filmic then sRGB");
    require(exposed.pixels[1].x > ldr.pixels[1].x, "Exposure measured in stops");
    const unsigned char solid[]{64,128,192, 64,128,192, 64,128,192, 64,128,192};
    Bytes jpeg;
    require(stbi_write_jpg_to_func(&collect, &jpeg, 2, 2, 3, solid, 100) != 0, "Fixture JPEG encoding");
    write(directory / "texture JPEG.jpg", jpeg);
    const auto jpg = io::decodeImage(jpeg.data(), jpeg.size(), "JPEG");
    near(jpg.pixels[0].y, 128 / 255.0f, 2 / 255.0f, "JPEG decode");
    rejects([&]{ io::decodeImage(nullptr, 0, "empty"); }, "empty");
    rejects([&]{ io::decodeImage(raw.data(), raw.size(), "not HDR", true); }, "expected a Radiance");
    rejects([&]{ io::loadImage(unicode); }, "not HDR");
    rejects([&]{ io::loadImage(directory / "missing.png"); }, "Cannot read");
    rejects([&]{ io::writePng(directory / "bad.png", -1, 2, linear, 0); }, "dimensions");
    rejects([&]{ io::writePng(directory / "bad.png", std::numeric_limits<int>::max(), 2, linear, 0); }, "dimensions");
    rejects([&]{ io::writeHdr(directory / "bad.hdr", 1, 1, linear); }, "pixel count");
    rejects([&]{ io::writePng(directory / "bad.png", 2, 2, linear, infinity); }, "exposure");
    auto bad = linear; bad[0].x = std::numeric_limits<float>::quiet_NaN();
    rejects([&]{ io::writeHdr(directory / "bad.hdr", 2, 2, bad); }, "finite");
    bad[0].x = -1;
    rejects([&]{ io::writePng(directory / "bad.png", 2, 2, bad, 0); }, "nonnegative");
    bad[0].x = std::numeric_limits<float>::max();
    rejects([&]{ io::writeHdr(directory / "bad.hdr", 2, 2, bad); }, "exponent");
    rejects([&]{ io::writePng(directory / "missing parent" / "out.png", 2, 2, linear, 0); }, "Cannot write");
    rejects([&]{ io::writeHdr(directory, 2, 2, linear); }, "Cannot write");
    rejects([&]{ io::loadImage(fs::path(R"(\\server.invalid\share\image.png)")); }, "local filesystem");
    rejects([&]{ io::loadImage(fs::path(R"(/\server.invalid\share\image.png)")); }, "local filesystem");
}

// Resource resolution must work from an unrelated cwd, with spaces/Unicode, and
// for embedded data or local file URIs, without enabling network reads.
void loadingTests(const fs::path& directory) {
    auto binary = meshBytes();
    const auto image = png();
    write(directory / "mesh data.bin", binary);
    write(directory / "textures" / fs::path(u8"col or \u00e9.png"), image);
    const auto external = document(R"(,"uri":"mesh%20data.bin")", binary.size(), R"({"uri":"textures/col%20or%20\u00e9.png"})");
    const auto source = directory / fs::path(u8"scene \u00e9.gltf");
    write(source, external);
    fs::current_path(directory / "unrelated cwd");
    const auto scene = io::loadGltf(fs::path("..") / source.filename());
    loadedContract(scene);
    require(scene.resources.size() == 3, "External scene, buffer, texture overwrite protection");
    require(scene.resources[0] == fs::canonical(source), "Scene resource resolved from non-default cwd");

    const auto embedded = document(",\"uri\":\"data:application/octet-stream;base64," + base64(binary) + "\"", binary.size(),
                                   "{\"uri\":\"data:image/png;base64," + base64(image) + "\"}");
    write(directory / "embedded.gltf", embedded);
    const auto dataScene = io::loadGltf(directory / "embedded.gltf");
    loadedContract(dataScene);
    require(dataScene.resources.size() == 1, "Data resources are contained in source file");

    auto jpegExternal = external;
    change(jpegExternal, R"(textures/col%20or%20\u00e9.png)", "texture%20JPEG.jpg");
    write(directory / "jpeg external.gltf", jpegExternal);
    require(io::loadGltf(directory / "jpeg external.gltf").images[0].width == 2, "External glTF JPEG texture");
    auto jpegEmbedded = embedded;
    change(jpegEmbedded, "data:image/png;base64," + base64(image), "data:image/jpeg;base64," + base64(read(directory / "texture JPEG.jpg")));
    write(directory / "jpeg embedded.gltf", jpegEmbedded);
    require(io::loadGltf(directory / "jpeg embedded.gltf").images[0].height == 2, "Embedded glTF JPEG texture");

    std::string percentBytes;
    constexpr char hexDigits[] = "0123456789ABCDEF";
    for (unsigned char value : binary) { percentBytes += '%'; percentBytes += hexDigits[value >> 4]; percentBytes += hexDigits[value & 15]; }
    auto percentBuffer = embedded;
    change(percentBuffer, "data:application/octet-stream;base64," + base64(binary), "data:application/octet-stream," + percentBytes);
    write(directory / "percent data.gltf", percentBuffer);
    require(io::loadGltf(directory / "percent data.gltf").triangles.size() == 1, "Percent-encoded binary data URI");

    auto fileUri = external;
    const auto fileUtf8 = (directory / "mesh data.bin").generic_u8string();
    const std::string filePath(fileUtf8.begin(), fileUtf8.end());
    change(fileUri, "mesh%20data.bin", std::string(filePath.starts_with('/') ? "file://" : "file:///") + filePath);
    write(directory / "file URI.gltf", fileUri);
    require(io::loadGltf(directory / "file URI.gltf").resources.size() == 3, "Local absolute file URI");

    align4(binary);
    const auto imageOffset = binary.size();
    append(binary, image);
    const auto glbJson = document("", binary.size(), R"({"bufferView":2,"mimeType":"image/png"})",
        ",{\"buffer\":0,\"byteOffset\":" + std::to_string(imageOffset) + ",\"byteLength\":" + std::to_string(image.size()) + "}");
    write(directory / "packed.glb", glb(glbJson, binary));
    const auto packed = io::loadGltf(directory / "packed.glb");
    loadedContract(packed);
    require(packed.resources.size() == 1, "GLB resources are contained in source file");

    auto metadata = embedded;
    // Harmless optional metadata is not an unimplemented appearance feature.
    change(metadata, "\"asset\":", "\"extras\":{\"extensions\":{\"APPLICATION_notes\":{\"note\":\"not appearance\"}}},\"asset\":");
    change(metadata, "\"extensionsUsed\":[", "\"extensionsUsed\":[\"KHR_xmp_json_ld\",");
    write(directory / "metadata.gltf", metadata);
    require(io::loadGltf(directory / "metadata.gltf").triangles.size() == 1, "Optional metadata may pass");

    auto badFixture = [&](std::string json, const std::string& fragment) {
        // Mutate one valid document at a time to isolate each rejection path.
        write(directory / "bad.gltf", json);
        rejects([&]{ io::loadGltf(directory / "bad.gltf"); }, fragment);
    };
    rejects([&]{ io::loadGltf(directory / "missing.gltf"); }, "Missing");
    auto broken = external; change(broken, "mesh%20data.bin", "missing.bin"); badFixture(broken, "missing.bin");
    broken = external; change(broken, "textures/col%20or%20\\u00e9.png", "textures/missing.png"); badFixture(broken, "missing.png");
    broken = embedded; change(broken, "\"alphaMode\":\"MASK\"", "\"alphaMode\":\"BLEND\""); badFixture(broken, "BLEND");
    broken = embedded; change(broken, "\"extensionsUsed\":[", "\"extensionsRequired\":[\"KHR_draco_mesh_compression\"],\"extensionsUsed\":["); badFixture(broken, "required extension");
    broken = embedded; change(broken, "\"KHR_materials_ior\":{", "\"KHR_materials_clearcoat\":{},\"KHR_materials_ior\":{"); badFixture(broken, "KHR_materials_clearcoat");
    broken = embedded; change(broken, "\"KHR_texture_transform\":{", "\"VENDOR_unknown_texture_effect\":{},\"KHR_texture_transform\":{"); badFixture(broken, "VENDOR_unknown_texture_effect");
    broken = embedded; change(broken, "\"thicknessFactor\":0.8", "\"thicknessTexture\":{\"index\":0},\"thicknessFactor\":0.8"); badFixture(broken, "thicknessTexture");
    broken = embedded; change(broken, "\"normalTexture\":", "\"occlusionTexture\":{\"index\":0},\"normalTexture\":"); badFixture(broken, "occlusionTexture");
    broken = embedded; change(broken, "\"POSITION\":0", "\"COLOR_0\":1,\"POSITION\":0"); badFixture(broken, "COLOR");
    broken = embedded; change(broken, "\"scene\":0", "\"animations\":[{}],\"scene\":0"); badFixture(broken, "Animations");
    broken = embedded; change(broken, "\"scene\":0", "\"skins\":[{\"joints\":[1]}],\"scene\":0"); badFixture(broken, "Skinning");
    broken = embedded; change(broken, "\"indices\":5", "\"targets\":[{\"POSITION\":0}],\"indices\":5"); badFixture(broken, "Morph");
    broken = embedded; change(broken, "\"indices\":5", "\"mode\":5,\"indices\":5"); badFixture(broken, "TRIANGLES");
    broken = embedded; change(broken, "\"scale\":[-2,3,4]", "\"scale\":[0,3,4]"); badFixture(broken, "Singular");
    broken = embedded; change(broken, "\"scale\":[-2,3,4]", "\"scale\":[1e100,3,4]"); badFixture(broken, "Non-finite");
    broken = embedded; change(broken, "\"byteStride\":56", "\"byteStride\":4"); badFixture(broken, "stride");
    broken = embedded; change(broken, "\"count\":3", "\"count\":999999999"); badFixture(broken, "allocation limit");
    broken = embedded; change(broken, "\"texCoord\":1", "\"texCoord\":2"); badFixture(broken, "TEXCOORD");
    broken = embedded; change(broken, "\"minFilter\":9728", "\"minFilter\":99"); badFixture(broken, "minFilter");
    broken = external; change(broken, "mesh%20data.bin", "https://example.invalid/data.bin"); badFixture(broken, "local files");
    auto truncated = glb(glbJson, binary); truncated.pop_back(); write(directory / "broken.glb", truncated);
    rejects([&]{ io::loadGltf(directory / "broken.glb"); }, "declared length");
}

// Sparse values are packed even when their base accessor is strided. Also cover
// implicit-zero bases, sparse indices, nonindexed meshes, and invalid replacements.
void sparseTests(const fs::path& directory) {
    Bytes bytes;
    for (int i = 0; i < 12; ++i) f32(bytes, 0);
    bytes.push_back(1); bytes.push_back(2); align4(bytes);
    for (float f : {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f}) f32(bytes, f);
    bytes.push_back(1); bytes.push_back(2); align4(bytes);
    u32(bytes, 1); u32(bytes, 2);
    write(directory / "sparse.bin", bytes);
    std::string json = R"({"asset":{"version":"2.0"},"buffers":[{"uri":"sparse.bin","byteLength":88}],
      "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48,"byteStride":16},
      {"buffer":0,"byteOffset":48,"byteLength":2},{"buffer":0,"byteOffset":52,"byteLength":24},
      {"buffer":0,"byteOffset":76,"byteLength":2},{"buffer":0,"byteOffset":80,"byteLength":8}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","sparse":{"count":2,"indices":{"bufferView":1,"componentType":5121},"values":{"bufferView":2}}},
      {"componentType":5125,"count":3,"type":"SCALAR","sparse":{"count":2,"indices":{"bufferView":3,"componentType":5121},"values":{"bufferView":4}}}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0})";
    write(directory / "sparse.gltf", json);
    auto scene = io::loadGltf(directory / "sparse.gltf");
    require(scene.triangles.size() == 1 && scene.materials.size() == 1, "Sparse indexed primitive and default material");
    near(scene.triangles[0].v[1].position, {1,0,0}, 0, "Sparse strided POSITION patch one");
    near(scene.triangles[0].v[2].position, {0,1,0}, 0, "Sparse strided POSITION patch two");
    near(scene.triangles[0].v[0].normal, {0,0,1}, 1e-6f, "Missing normals generated");
    require(lengthSquared(Vec3{scene.triangles[0].v[0].tangent.x,scene.triangles[0].v[0].tangent.y,scene.triangles[0].v[0].tangent.z}) > 0.9f, "Missing tangent fallback generated");
    change(json, ",\"indices\":1", "");
    change(json, "\"bufferView\":0,\"componentType\":5126", "\"componentType\":5126");
    write(directory / "sparse no base.gltf", json);
    scene = io::loadGltf(directory / "sparse no base.gltf");
    near(scene.triangles[0].area, 0.5f, 1e-6f, "Sparse-without-base, nonindexed primitive");
    auto tiny = json;
    change(tiny, "\"mesh\":0", "\"mesh\":0,\"scale\":[-1e-7,1e-7,1e-7]");
    write(directory / "tiny transform.gltf", tiny);
    const auto tinyScene = io::loadGltf(directory / "tiny transform.gltf");
    require(tinyScene.triangles.size() == 1, "Well-conditioned tiny mirrored transform is not singular");
    near(tinyScene.triangles[0].geometricNormal, {0,0,1}, 1e-6f, "Tiny transform outward normal");
    bytes[49] = 1; write(directory / "sparse.bin", bytes);
    rejects([&]{ io::loadGltf(directory / "sparse.gltf"); }, "strictly increasing");
    bytes[49] = 3; write(directory / "sparse.bin", bytes);
    rejects([&]{ io::loadGltf(directory / "sparse.gltf"); }, "in range");
    bytes[49] = 2; bytes[84] = 9; write(directory / "sparse.bin", bytes);
    rejects([&]{ io::loadGltf(directory / "sparse.gltf"); }, "index exceeds");
}

// Compare actual Blender GLB/external exports, then use independent manifest
// coordinates to avoid validating two identically wrong mirrored transforms.
void blenderParity() {
#ifdef TEST_SOURCE_DIR
    const fs::path source = fs::path(TEST_SOURCE_DIR) / "scenes" / "fixtures";
    if (!fs::exists(source / "export_contract.glb")) {
        std::cout << "Blender fixture not present; generated I/O fixtures remain mandatory\n";
        return;
    }
    const auto a = io::loadGltf(source / "export_contract.glb");
    const auto b = io::loadGltf(source / "export_contract.gltf");
    require(a.triangles.size() == 1602 && b.triangles.size() == 1602, "Blender triangle counts");
    require(a.materials.size() == 7 && a.images.size() == 5 && a.textures.size() == 6, "Blender owned asset counts");
    require(a.cameras.size() == 1 && a.cameras[0].name == "Reference" && a.lights.size() == 2, "Blender camera/lights");
    near(a.cameras[0].yfov, 0.4958353284f, 1e-6f, "Blender camera FOV");
    near(a.cameras[0].nearClip, 0.05f, 1e-6f, "Blender camera near plane");
    near(a.cameras[0].farClip, 200, 0, "Blender camera far plane");
    require(a.resources.size() == 1 && b.resources.size() == 7, "Blender GLB/external resource tracking");
    bool water = false, foliage = false, emissive = false, transformed = false;
    for (const auto& m : a.materials) {
        if (m.transmission > 0) {
            water = true; near(m.ior, 1.333f, 1e-6f, "Blender water IOR");
            near(m.thickness, 0.8f, 1e-6f, "Blender closed water"); near(m.attenuationDistance, 3, 0, "Blender volume attenuation");
            require(m.normalTexture.texture >= 0 && !m.doubleSided, "Blender water map/sidedness");
        }
        if (m.alphaMode == AlphaMode::Mask) {
            foliage = true; near(m.alphaCutoff, 0.42f, 1e-6f, "Blender foliage cutoff");
            require(m.doubleSided && m.baseColorTexture.texture >= 0, "Blender foliage alpha texture");
        }
        if (maxComponent(m.emission) > 1) { emissive = true; near(m.emission, {6,6,6}, 1e-5f, "Blender emissive strength"); }
        if (m.baseColorTexture.rotation != 0) { transformed = true; near(m.baseColorTexture.rotation, 0.31f, 1e-6f, "Blender UV rotation"); }
    }
    require(water && foliage && emissive && transformed, "Blender material feature parity");
    bool mirrorProbe = false;
    for (std::size_t i = 0; i < a.triangles.size(); ++i) {
        require(a.triangles[i].material == b.triangles[i].material, "GLB/external material identity");
        for (int j = 0; j < 3; ++j) {
            const auto& x = a.triangles[i].v[j]; const auto& y = b.triangles[i].v[j];
            near(x.position, y.position, 1e-6f, "GLB/external world position parity");
            near(x.normal, y.normal, 1e-6f, "GLB/external normal parity");
            near(x.tangent.w, y.tangent.w, 0, "GLB/external tangent parity");
            require(dot(x.normal, a.triangles[i].geometricNormal) > 0, "Blender outward shading/geometric normal agreement");
            if (a.materials[a.triangles[i].material].name == "Normal_Mirrored") {
                near(x.tangent.w, -1, 0, "Blender mirrored node tangent handedness");
                const Vec3 expectedPosition{1.8628146648f,2.3146166801f,-0.7727805972f};
                const Vec3 expectedNormal{0.2633435428f,0.9309852123f,0.2528177798f};
                const Vec3 expectedTangent{-0.1066619083f,-0.2323626280f,0.9667630792f};
                if (length(x.position - expectedPosition) < 3e-6f && length(x.normal - expectedNormal) < 3e-6f &&
                    length(Vec3{x.tangent.x,x.tangent.y,x.tangent.z} - expectedTangent) < 3e-6f) mirrorProbe = true;
            }
        }
    }
    require(mirrorProbe, "World-space Blender mirror probe matches independent manifest values");
    // Embedded and external PNGs must decode to exactly the same channel values.
    for (std::size_t i = 0; i < a.images.size(); ++i) {
        require(a.images[i].pixels.size() == b.images[i].pixels.size(), "Blender embedded/external image size parity");
        for (std::size_t j = 0; j < a.images[i].pixels.size(); ++j) {
            const auto x = a.images[i].pixels[j], y = b.images[i].pixels[j];
            require(x.x == y.x && x.y == y.y && x.z == y.z && x.w == y.w, "Blender embedded/external decoded texel parity");
        }
    }
    std::cout << "Blender import parity: 1602 triangles, 7 materials, 5 images, Reference camera, 2 lights\n";
#endif
}
}

int main() {
    try {
        // The RAII directory owns only generated fixtures; shipped assets are read-only.
        TemporaryDirectory temporary;
        imageTests(temporary.path);
        loadingTests(temporary.path);
        sparseTests(temporary.path);
        blenderParity();
        std::cout << "I/O tests passed: " << checks << " exception-backed checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "I/O test failure after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
