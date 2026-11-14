#pragma once
/** @file
 * @brief Historical stage-3 raw-prefix fixture, intentionally outside engine headers.
 * @details Preserves the old fixture's contract and evidence. Production numeric
 * loading uses storage/column_loader.h; this is not text/byte SQL support.
 */
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

template<class T>
inline std::vector<T> loadColumnFile(const std::filesystem::path& path, std::size_t rows) {
    if (rows > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::overflow_error("Column file byte size overflow");
    const auto bytes = rows * sizeof(T);
    const auto read_bytes = static_cast<std::size_t>(std::min<std::uintmax_t>(std::filesystem::file_size(path), bytes));
    std::vector<T> values(rows);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open historical column: " + path.string());
    if (read_bytes && !input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(read_bytes)))
        throw std::runtime_error("Failed to read historical column: " + path.string());
    return values;
}
