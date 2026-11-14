#include "storage/text_column.h"
#include "storage/column_loader.h"
#include "core/size_expression.h"
#include <algorithm>
#include <climits>

namespace db {
namespace {
/** @brief Determine the legal UTF-8 sequence size and restricted second-byte range. */
unsigned sequenceBytes(unsigned char first) {
    if (first < 0x80) return 1;
    if (first >= 0xc2 && first <= 0xdf) return 2;
    if (first >= 0xe0 && first <= 0xef) return 3;
    if (first >= 0xf0 && first <= 0xf4) return 4;
    throw std::runtime_error("Invalid UTF-8 leading byte in Text");
}
/** @brief Restrict second bytes to exclude overlong sequences, surrogates and values above U+10FFFF. */
bool validSecondByte(unsigned char first, unsigned char second) {
    switch (first) {
        case 0xe0: return second >= 0xa0;
        case 0xed: return second < 0xa0;
        case 0xf0: return second >= 0x90;
        case 0xf4: return second < 0x90;
        default: return true;
    }
}
void validateSequence(std::string_view value, std::size_t at, unsigned count) {
    if (count > value.size() - at) throw std::runtime_error("Truncated UTF-8 sequence in Text");
    for (unsigned i = 1; i < count; ++i) {
        const auto byte = static_cast<unsigned char>(value[at + i]);
        if (byte < 0x80 || byte > 0xbf) throw std::runtime_error("Invalid UTF-8 continuation in Text");
    }
    if (count == 1) return;
    const auto first = static_cast<unsigned char>(value[at]);
    const auto second = static_cast<unsigned char>(value[at + 1]);
    if (!validSecondByte(first, second))
        throw std::runtime_error("Overlong, surrogate or out-of-range UTF-8 in Text");
}
/** @brief Validate file/offset/bitmap extents before allocating the owning host column. */
uint64_t textColumnExtents(const std::filesystem::path& path, std::size_t rows, ColumnStorage storage,
                           const std::vector<std::uint64_t>& validity) {
    if (storage.encoding != PhysicalEncoding::FixedText || !storage.width || storage.width > INT32_MAX)
        throw std::invalid_argument("Invalid FixedText storage encoding/width");
    if (!validity.empty() && validity.size() != rows / 64 + (rows % 64 != 0))
        throw std::invalid_argument("Text validity extent differs from row count");
    columnFileBytes(path, rows, storage.width);
    const auto offsets = checkedSizeAdd(rows, 1);
    checkedSizeMultiply(offsets, sizeof(std::uint64_t));
    return offsets;
}
/** @brief Append only a valid record's bounded logical prefix, never its physical padding. */
void appendRecord(HostTextColumn& result, const std::vector<char>& record) {
    const auto end = std::find(record.begin(), record.end(), '\0');
    const std::string_view value(record.data(), static_cast<std::size_t>(end - record.begin()));
    validateText(value);
    const auto bytes = checkedSizeAdd(result.bytes.size(), value.size());
    if (bytes > static_cast<std::uint64_t>(INT64_MAX)) throw std::overflow_error("Text column offset overflow");
    result.bytes.insert(result.bytes.end(), value.begin(), value.end());
}
}
void validateText(std::string_view value) {
    if (value.size() > INT32_MAX) throw std::length_error("Text exceeds INT32_MAX bytes");
    for (std::size_t at = 0; at < value.size();) {
        if (value[at] == 0) throw std::runtime_error("Logical NUL is not supported in Text");
        const auto count = sequenceBytes(static_cast<unsigned char>(value[at]));
        validateSequence(value, at, count);
        at += count;
    }
}
HostTextColumn loadTextColumn(const std::filesystem::path& path, std::size_t rows,
                             ColumnStorage storage, const std::vector<std::uint64_t>& validity) {
    const auto offset_count = textColumnExtents(path, rows, storage, validity);
    HostTextColumn result;
    result.offsets.reserve(offset_count);
    result.offsets.push_back(0);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open Text column: " + path.string());
    std::vector<char> record(storage.width);
    for (std::size_t row = 0; row < rows; ++row) {
        if (!input.read(record.data(), record.size())) throw std::runtime_error("Truncated Text column");
        const bool valid = validity.empty() || ((validity[row / 64] >> (row % 64)) & 1U);
        if (valid) {
            appendRecord(result, record);
        }
        result.offsets.push_back(result.bytes.size());
    }
    return result;
}
} // namespace db
