#pragma once
/** @file @brief Streaming, host-only import of unencoded pipe-delimited dbgen tables. */
#include "storage/ssb_dataset.h"

namespace db {
/** @brief Convert all five tables, validate, then publish the complete directory atomically.
 * @throws std::exception For malformed input, overflow, I/O or any existing output.
 * @details Uses bounded row/column buffers. Only its own private staging directory
 * is removed on failure. Linux no-replace rename prevents concurrent overwrite.
 */
SsbDataset importSsbTables(const std::filesystem::path& input, const std::filesystem::path& output,
                           const std::string& scale, const std::string& generator);
} // namespace db
