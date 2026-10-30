#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cr::io::detail {
// Encoded files and decoded images have separate limits: a small compressed file
// can expand into a much larger pixel buffer. glTF also applies aggregate budgets.
inline constexpr std::size_t maxFileBytes = 512ull * 1024 * 1024;
inline constexpr std::size_t maxImagePixels = 16ull * 1024 * 1024;
inline constexpr int maxImageDimension = 16384;

// Use UTF-8 for diagnostics without narrowing the native path used for file I/O
// through Windows' current ANSI code page.
inline std::string pathText(const std::filesystem::path& path) {
    const auto s = path.u8string();
    return std::string(s.begin(), s.end());
}

// Reject UNC-style names before even querying file metadata. Both slash styles
// count, so mixing them cannot bypass this lexical local-path policy. This is
// not a sandbox that confines ordinary relative paths to one directory.
inline void requireLocal(const std::filesystem::path& path) {
    const auto text = pathText(path);
    const auto separator = [](char c) { return c == '/' || c == '\\'; };
    if (text.size() >= 2 && separator(text[0]) && separator(text[1]))
        throw std::runtime_error("Only local filesystem paths are supported; network/UNC paths are disabled");
}

// The owning handle closes automatically on exceptions. Image writers explicitly
// close it later so they can report errors from buffered writes and fclose.
using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
// Binary mode prevents newline translation; the Windows wide-character entry
// point preserves spaces and Unicode names without handing paths to the codecs.
inline File openFile(const std::filesystem::path& path, bool writing) {
    requireLocal(path);
#ifdef _WIN32
    std::FILE* file = _wfopen(path.c_str(), writing ? L"wb" : L"rb");
#else
    std::FILE* file = std::fopen(path.c_str(), writing ? "wb" : "rb");
#endif
    if (!file) throw std::runtime_error(std::string(writing ? "Cannot write '" : "Cannot read '") +
                                        pathText(path) + "': " + std::strerror(errno));
    return File(file, &std::fclose);
}

// Resolve an existing regular file to an absolute identity. The glTF loader keeps
// these identities in Scene.resources for the CLI's input-overwrite protection.
inline std::filesystem::path canonicalInput(const std::filesystem::path& path) {
    requireLocal(path);
    std::error_code ec;
    auto result = std::filesystem::canonical(path, ec);
    if (ec || !std::filesystem::is_regular_file(result, ec))
        throw std::runtime_error("Missing or unreadable input file '" + pathText(path) + "'");
    return result;
}

// Bound the allocation before converting the filesystem size or reading bytes.
// A short read is an error rather than partially initialized codec input; format
// interpretation is deliberately left to the glTF and image adapters.
inline std::vector<unsigned char> readFile(const std::filesystem::path& path,
                                           std::size_t limit = maxFileBytes) {
    requireLocal(path);
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) throw std::runtime_error("Cannot read input file '" + pathText(path) + "': " + ec.message());
    if (!size || size > limit)
        throw std::runtime_error("Input file is empty or exceeds the allocation limit: '" + pathText(path) + "'");
    auto file = openFile(path, false);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (std::fread(bytes.data(), 1, bytes.size(), file.get()) != bytes.size())
        throw std::runtime_error("Failed reading input file '" + pathText(path) + "'");
    return bytes;
}

// Short-circuit the nonpositive cases before dividing. Comparing by division
// avoids overflowing width * height while enforcing the decoded-pixel budget.
inline std::size_t imagePixels(int width, int height, const std::string& label) {
    if (width <= 0 || height <= 0 || width > maxImageDimension || height > maxImageDimension ||
        static_cast<std::size_t>(width) > maxImagePixels / static_cast<std::size_t>(height))
        throw std::runtime_error(label + ": image dimensions must be positive, at most 16384 per axis and 16777216 pixels");
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}
}
