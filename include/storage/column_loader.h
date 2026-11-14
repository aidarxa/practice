#pragma once
/** @file @brief Strict numeric decoding at the storage boundary, with no raw-prefix view. */
#include "storage/column.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace db {
/** @brief Validate the complete physical extent before reading or allocating. */
inline std::size_t columnFileBytes(const std::filesystem::path& path, std::size_t rows, std::size_t width) {
    if (!width || rows > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / width)
        throw std::overflow_error("Column file byte size overflow");
    const auto bytes = rows * width;
    if (std::filesystem::file_size(path) != bytes)
        throw std::runtime_error("Invalid column size at " + path.string() + ": expected " + std::to_string(bytes));
    return bytes;
}
/** @brief Read exactly one physical extent, rejecting truncation and extra data. */
template<class T> inline std::vector<T> readExactColumn(const std::filesystem::path& path, std::size_t rows) {
    const auto bytes = columnFileBytes(path, rows, sizeof(T));
    std::vector<T> result(rows);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open column: " + path.string());
    if (bytes && !input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Failed to read column: " + path.string());
    return result;
}
/** @brief Decode a little-endian Int32 without host endianness or signed-shift assumptions. */
inline int32_t decodeInt32LE(const unsigned char* bytes) {
    const uint32_t value = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8U) |
                           (uint32_t(bytes[2]) << 16U) | (uint32_t(bytes[3]) << 24U);
    return value <= uint32_t(INT32_MAX) ? static_cast<int32_t>(value)
        : static_cast<int32_t>(static_cast<int64_t>(value) - (int64_t{1} << 32U));
}
/** @brief Widen only documented ASCII flags. Numeric bytes, whitespace and junk are errors. */
inline int32_t decodeAsciiBoolean(unsigned char value) {
    if (value != '0' && value != '1') throw std::runtime_error("Invalid ASCII boolean in SSB column");
    return value - '0';
}
/** @brief Text requires extension 7-TEXT, and cannot enter a numeric buffer. */
inline void requireNumericStorage(ColumnStorage storage) {
    if (!storage.isNumeric()) throw std::runtime_error("Text columns require extension 7-TEXT; raw Int32 and dictionary views are not supported");
    const auto width = storage.encoding == PhysicalEncoding::AsciiBoolean ? 1U : 4U;
    if (storage.width != width) throw std::invalid_argument("Invalid numeric physical width");
}
/** @brief Load each numeric row correctly; never truncate a file or manufacture a zero tail. */
inline std::vector<int32_t> loadNumericColumn(const std::filesystem::path& path, std::size_t rows, ColumnStorage storage) {
    requireNumericStorage(storage);
    columnFileBytes(path, rows, storage.width);
    if (rows > std::numeric_limits<std::size_t>::max() / sizeof(int32_t)) throw std::overflow_error("Decoded column byte size overflow");
    if (storage.encoding == PhysicalEncoding::AsciiBoolean) {
        const auto bytes = readExactColumn<unsigned char>(path, rows);
        std::vector<int32_t> result;
        result.reserve(rows);
        for (const auto value : bytes) result.push_back(decodeAsciiBoolean(value));
        return result;
    }
    auto result = readExactColumn<int32_t>(path, rows);
    // Reinterpretation as bytes is legal even when int32_t is not native little endian.
    for (auto& value : result) value = decodeInt32LE(reinterpret_cast<const unsigned char*>(&value));
    return result;
}
}
