#pragma once
/** @file @brief Strict Text input normalization, independent of SQL and GPU SDK headers. */
#include "core/text.h"
#include "storage/column.h"
#include <filesystem>
#include <string_view>
#include <vector>

namespace db {
/** @brief Reject malformed UTF-8, logical NUL and a value longer than INT32_MAX bytes. */
void validateText(std::string_view value);
/** @brief A normalized owning host column, with one terminal offset and external validity. */
struct HostTextColumn {
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint64_t> offsets;
};
/** @brief Read full fixed-width records; ignore post-NUL tail, never trim spaces.
 * @param validity Optional LSB-first bitmap; masked payload bytes are not interpreted.
 * @throws std::runtime_error On incorrect extents, encoding or bitmap shape.
 */
HostTextColumn loadTextColumn(const std::filesystem::path& path, std::size_t rows,
                             ColumnStorage storage, const std::vector<std::uint64_t>& validity = {});
} // namespace db
