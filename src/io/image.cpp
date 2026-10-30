#include "scene.h"
#include "file.h"

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace {
// stb owns temporary decode buffers, including decompression scratch space.
// An aligned prefix records each allocation without changing stb's payload ABI.
// Decode and release happen on the calling thread, so concurrent loads maintain
// independent budgets rather than contending on shared accounting state.
struct alignas(std::max_align_t) DecodeAllocation { std::size_t bytes; };
thread_local std::size_t decodeBytes = 0;
// Return nullptr through stb's C allocation interface instead of throwing inside
// the codec. The subtraction-based guard bounds outstanding payload bytes safely.
void* decodeAllocate(std::size_t bytes) {
    constexpr std::size_t limit = 512ull * 1024 * 1024;
    if (bytes > limit || decodeBytes > limit - bytes) return nullptr;
    auto* allocation = static_cast<DecodeAllocation*>(std::malloc(sizeof(DecodeAllocation) + bytes));
    if (!allocation) return nullptr;
    allocation->bytes = bytes;
    decodeBytes += bytes;
    return allocation + 1;
}
// Recover the hidden size for both normal completion and partial-decode cleanup.
void decodeFree(void* pointer) {
    if (!pointer) return;
    auto* allocation = static_cast<DecodeAllocation*>(pointer) - 1;
    decodeBytes -= allocation->bytes;
    std::free(allocation);
}
// Allocate before releasing: a failed growth leaves the old buffer valid, as
// realloc requires, and the budget includes the temporary old-plus-new peak.
void* decodeReallocate(void* pointer, std::size_t bytes) {
    if (!pointer) return decodeAllocate(bytes);
    auto* allocation = static_cast<DecodeAllocation*>(pointer) - 1;
    void* replacement = decodeAllocate(bytes);
    if (!replacement) return nullptr;
    std::memcpy(replacement, pointer, std::min(bytes, allocation->bytes));
    decodeFree(pointer);
    return replacement;
}
}

// Compile the pinned codec implementations only in this adapter. Suppress their
// warnings locally, then restore the project's warning policy for our own code.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
// Only the asset-contract formats are enabled. Reading through memory APIs keeps
// filename handling and local-path checks in file.h rather than in stb's stdio.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_HDR
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 16384
#define STBI_MALLOC(size) decodeAllocate(size)
#define STBI_REALLOC(pointer, size) decodeReallocate(pointer, size)
#define STBI_FREE(pointer) decodeFree(pointer)
#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../third_party/stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace cr::io {
namespace {
// stb's output callback cannot return an I/O error. Borrow the owning File's
// handle and latch short writes for finish(), without throwing through the codec.
struct Output {
    std::FILE* file;
    bool failed = false;
    static void write(void* context, void* data, int size) {
        auto& out = *static_cast<Output*>(context);
        if (size < 0 || (size && std::fwrite(data, 1, static_cast<std::size_t>(size), out.file) !=
                                   static_cast<std::size_t>(size))) out.failed = true;
    }
};

// Codec success does not prove bytes reached the filesystem: disk errors can
// surface during fwrite, flushing, or close. Releasing the RAII handle before the
// explicit close prevents a second fclose when the owning wrapper is destroyed.
void finish(detail::File& file, Output& out, int result, const std::filesystem::path& path) {
    const bool flushFailed = std::fflush(file.get()) != 0 || std::ferror(file.get()) != 0;
    const bool closeFailed = std::fclose(file.release()) != 0;
    if (!result || out.failed || flushFailed || closeFailed)
        throw std::runtime_error("Failed writing image '" + detail::pathText(path) + "'");
}

// Check shape and every radiance sample before opening/truncating the output.
// Invalid accumulation values are reported, not silently replaced with black.
std::size_t validateOutput(int width, int height, const std::vector<Vec3>& pixels,
                           const std::string& label, bool hdr) {
    const std::size_t count = detail::imagePixels(width, height, label);
    if (pixels.size() != count) throw std::runtime_error(label + ": pixel count does not match dimensions");
    for (const Vec3 p : pixels) {
        if (!finite(p) || p.x < 0 || p.y < 0 || p.z < 0)
            throw std::runtime_error(label + ": pixels must contain finite, nonnegative linear radiance");
        // RGBE has an 8-bit biased exponent; FLT_MAX would wrap that exponent to zero.
        if (hdr && maxComponent(p) >= std::ldexp(1.0f, 127))
            throw std::runtime_error(label + ": radiance exceeds the Radiance RGBE exponent range");
    }
    return count;
}

// Display conversion is ordered: expose linear radiance, compress highlights
// with an ACES-like rational curve, encode sRGB, then quantize to eight bits.
// This is not a full ACES color-management pipeline. Double precision keeps the
// curve finite over the allowed exposure range even for very bright float input.
unsigned char displayChannel(float value, double exposureScale) {
    const double x = static_cast<double>(value) * exposureScale;
    const double filmic = std::clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
    const float srgb = linearToSrgb(static_cast<float>(filmic));
    return static_cast<unsigned char>(std::lround(std::clamp(srgb, 0.0f, 1.0f) * 255.0f));
}
}

// Input bytes are borrowed only for this call. Force a uniform RGBA layout, copy
// into engine-owned pixels, and release every codec buffer through its deleter.
// The caller explicitly chooses HDR lighting versus encoded material textures.
Image decodeImage(const unsigned char* bytes, std::size_t size, const std::string& label, bool hdr) {
    if (!bytes || !size || size > detail::maxFileBytes || size > static_cast<std::size_t>(INT_MAX))
        throw std::runtime_error(label + ": empty image or encoded image exceeds the allocation limit");
    // stb accepts an int byte count. Check it before narrowing, and disallow its
    // automatic LDR/HDR conversion so loading cannot introduce implicit gamma.
    const int inputSize = static_cast<int>(size);
    const bool isHdr = stbi_is_hdr_from_memory(bytes, inputSize) != 0;
    if (hdr != isHdr)
        throw std::runtime_error(label + (hdr ? ": expected a Radiance .hdr environment" : ": expected a PNG or JPEG image, not HDR"));
    // Probe dimensions before allocating the engine image or fully decompressing
    // it; encoded-byte limits alone do not prevent oversized decoded images.
    int width = 0, height = 0, components = 0;
    if (!stbi_info_from_memory(bytes, inputSize, &width, &height, &components))
        throw std::runtime_error(label + ": cannot decode PNG/JPEG/HDR header: " +
                                 (stbi_failure_reason() ? stbi_failure_reason() : "invalid image"));
    const std::size_t count = detail::imagePixels(width, height, label);
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(count);
    if (hdr) {
        // Radiance HDR decodes directly to linear floats. Do not clamp values
        // above one: those values carry illumination and highlight energy.
        std::unique_ptr<float, decltype(&stbi_image_free)> pixels(
            stbi_loadf_from_memory(bytes, inputSize, &width, &height, &components, 4), &stbi_image_free);
        if (!pixels) throw std::runtime_error(label + ": HDR decode failed");
        for (std::size_t i = 0; i < count; ++i) {
            const float* p = pixels.get() + i * 4;
            if (!finite(Vec3{p[0], p[1], p[2]}) || !std::isfinite(p[3]) ||
                p[0] < 0 || p[1] < 0 || p[2] < 0)
                throw std::runtime_error(label + ": non-finite or negative HDR pixels");
            image.pixels[i] = {p[0], p[1], p[2], p[3]};
        }
    } else if (stbi_is_16_bit_from_memory(bytes, inputSize)) {
        // Preserve 16-bit PNG precision, especially for data maps, instead of
        // reducing to eight bits before converting integer samples to floats.
        std::unique_ptr<stbi_us, decltype(&stbi_image_free)> pixels(
            stbi_load_16_from_memory(bytes, inputSize, &width, &height, &components, 4), &stbi_image_free);
        if (!pixels) throw std::runtime_error(label + ": 16-bit PNG decode failed");
        for (std::size_t i = 0; i < count; ++i) {
            const stbi_us* p = pixels.get() + i * 4;
            image.pixels[i] = {p[0] / 65535.0f, p[1] / 65535.0f, p[2] / 65535.0f, p[3] / 65535.0f};
        }
    } else {
        // PNG/JPEG samples remain in their encoded channel space here; alpha is
        // normalized like the other channels but never receives an sRGB curve.
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
            stbi_load_from_memory(bytes, inputSize, &width, &height, &components, 4), &stbi_image_free);
        if (!pixels) throw std::runtime_error(label + ": PNG/JPEG decode failed: " +
                                             (stbi_failure_reason() ? stbi_failure_reason() : "invalid image"));
        for (std::size_t i = 0; i < count; ++i) {
            const stbi_uc* p = pixels.get() + i * 4;
            image.pixels[i] = {p[0] / 255.0f, p[1] / 255.0f, p[2] / 255.0f, p[3] / 255.0f};
        }
    }
    // Keep encoded color channels and top-left rows. Material slots decide sRGB versus data-map use.
    // One image can serve both color and linear-data slots; neither its channels
    // nor its row order should be irreversibly transformed during asset loading.
    return image;
}

// File-backed and embedded images share the same decoder and validation rules;
// this wrapper adds bounded local file reading and a useful diagnostic label.
Image loadImage(const std::filesystem::path& path, bool hdr) {
    const auto bytes = detail::readFile(path);
    return decodeImage(bytes.data(), bytes.size(), detail::pathText(path), hdr);
}

// PNG is the display deliverable, not a radiometric archive. Exposure is in
// stops (a multiplier of 2^exposure), and conversion never modifies accumulation.
// File creation is deferred until validation and pixel conversion have succeeded.
void writePng(const std::filesystem::path& path, int width, int height,
              const std::vector<Vec3>& linear, float exposure) {
    const auto count = validateOutput(width, height, linear, "PNG output", false);
    if (!std::isfinite(exposure) || exposure < -100 || exposure > 100)
        throw std::runtime_error("PNG exposure must be finite and between -100 and 100 stops");
    const double scale = std::exp2(static_cast<double>(exposure));
    std::vector<unsigned char> pixels(count * 3);
    for (std::size_t i = 0; i < count; ++i) {
        pixels[i * 3] = displayChannel(linear[i].x, scale);
        pixels[i * 3 + 1] = displayChannel(linear[i].y, scale);
        pixels[i * 3 + 2] = displayChannel(linear[i].z, scale);
    }
    auto file = detail::openFile(path, true);
    Output out{file.get()};
    const int result = stbi_write_png_to_func(&Output::write, &out, width, height, 3, pixels.data(), width * 3);
    finish(file, out, result, path);
}

// HDR preserves the unexposed linear signal: no filmic curve, sRGB encoding, or
// exposure multiplier. RGBE stores three mantissas with a shared exponent, so the
// codec still quantizes the signal rather than preserving float bit patterns.
void writeHdr(const std::filesystem::path& path, int width, int height, const std::vector<Vec3>& linear) {
    const auto count = validateOutput(width, height, linear, "HDR output", true);
    // Pack explicitly instead of relying on Vec3 having no padding or treating
    // its member subobjects as an array accepted by the C codec interface.
    std::vector<float> pixels(count * 3);
    for (std::size_t i = 0; i < count; ++i) {
        pixels[i * 3] = linear[i].x;
        pixels[i * 3 + 1] = linear[i].y;
        pixels[i * 3 + 2] = linear[i].z;
    }
    auto file = detail::openFile(path, true);
    Output out{file.get()};
    const int result = stbi_write_hdr_to_func(&Output::write, &out, width, height, 3, pixels.data());
    finish(file, out, result, path);
}
}
