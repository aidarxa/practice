#pragma once
/** @file @brief Versioned native SSB metadata, independent of SQL and the GPU SDK. */
#include "storage/ssb_schema.h"
#include <cstdint>
#include <filesystem>

namespace db {
inline constexpr const char* ssbManifestName = "dataset.manifest";
inline constexpr const char* ssbDefaultDirectory = "data/ssb/sf10";
/** @brief Immutable dataset identity, validated schema and actual table cardinalities.
 * @details A manifest is mandatory: legacy encoded files cannot be mistaken for Text.
 * Column filenames are derived from the compiled schema, never trusted as paths.
 */
class SsbDataset {
public:
    using RowCounts = std::array<std::uint64_t, ssbTables.size()>;
    SsbDataset(std::filesystem::path directory, std::string scale, std::string generator, RowCounts rows);
    /** @brief Read strict metadata and validate every column's complete physical extent. */
    static SsbDataset open(const std::filesystem::path& directory);
    const std::filesystem::path& directory() const { return directory_; }
    const std::string& scale() const { return scale_; }
    const std::string& generator() const { return generator_; }
    std::uint64_t rowCount(std::string_view table) const;
    std::filesystem::path columnPath(std::string_view column) const;
    /** @brief Exclusively create a manifest; an existing manifest is never replaced. */
    void writeManifest() const;
private:
    std::filesystem::path directory_;
    std::string scale_, generator_;
    RowCounts rows_;
    std::string manifest() const;
};
/** @brief Validate a positive finite decimal scale without locale-dependent syntax. */
void validateSsbScale(const std::string& scale);
} // namespace db
