/** @file @brief Strict native SSB manifest serialization and validation. */
#include "storage/ssb_dataset.h"
#include "storage/column_loader.h"
#include "ssb_parse.h"
#include <cmath>
#include <fcntl.h>
#include <locale>
#include <sstream>
#include <system_error>
#include <unistd.h>

namespace db {
namespace {
constexpr const char* magic = "CRYSTAL_SSB_NATIVE_1";
std::string line(std::istream& input) {
    // Metadata is tiny; bound individual records before allocating untrusted strings.
    char buffer[1024];
    if (!input.getline(buffer, sizeof(buffer))) throw std::runtime_error("Incomplete or oversized SSB manifest record");
    return std::string(buffer, static_cast<std::size_t>(input.gcount()) - 1);
}
std::string field(std::istream& input, const std::string& prefix) {
    const auto value = line(input);
    if (value.compare(0, prefix.size(), prefix) != 0) throw std::runtime_error("Invalid SSB manifest record: " + value);
    return value.substr(prefix.size());
}
void validateGenerator(const std::string& value) {
    if (value.empty() || value.size() > 256 || value.find_first_of("\t\r\n") != std::string::npos || value.find('\0') != std::string::npos)
        throw std::invalid_argument("Invalid SSB generator identity");
}
const char* encodingName(PhysicalEncoding encoding) {
    switch (encoding) {
        case PhysicalEncoding::Int32LE: return "Int32LE";
        case PhysicalEncoding::AsciiBoolean: return "AsciiBoolean";
        case PhysicalEncoding::FixedText: return "FixedText";
    }
    throw std::invalid_argument("Unknown SSB physical encoding");
}
std::string columnRecord(const SsbColumn& column) {
    return "column\t" + std::string(column.name) + "\t" + column.fileName() + "\t" +
        encodingName(column.storage.encoding) + "\t" + std::to_string(column.storage.width);
}
void requireRegular(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)))
        throw std::runtime_error("SSB requires a regular file, not a link: " + path.string());
}
void writeExclusive(const std::filesystem::path& path, const std::string& value) {
    const auto fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "Create " + path.string());
    std::size_t at = 0;
    while (at < value.size()) {
        const auto written = ::write(fd, value.data() + at, value.size() - at);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) { const auto error = errno; ::close(fd); throw std::system_error(error, std::generic_category(), "Write manifest"); }
        at += static_cast<std::size_t>(written);
    }
    if (::close(fd) != 0) throw std::system_error(errno, std::generic_category(), "Close manifest");
}
} // namespace
void validateSsbScale(const std::string& scale) {
    if (scale.empty() || scale.size() > 64 || scale.find_first_not_of("0123456789.") != std::string::npos)
        throw std::invalid_argument("SSB scale must be a positive finite decimal");
    std::istringstream input(scale); input.imbue(std::locale::classic());
    double value = 0;
    input >> std::noskipws >> value;
    if (!input || !input.eof() || !std::isfinite(value) || value <= 0)
        throw std::invalid_argument("SSB scale must be a positive finite decimal");
}
SsbDataset::SsbDataset(std::filesystem::path directory, std::string scale, std::string generator, RowCounts rows)
    : directory_(std::filesystem::canonical(directory)), scale_(std::move(scale)), generator_(std::move(generator)), rows_(rows) {
    validateSsbScale(scale_);
    validateGenerator(generator_);
}
std::uint64_t SsbDataset::rowCount(std::string_view table) const {
    for (std::size_t index = 0; index < ssbTables.size(); ++index)
        if (table == ssbTables[index].name) return rows_[index];
    throw std::invalid_argument("Unknown SSB table: " + std::string(table));
}
std::filesystem::path SsbDataset::columnPath(std::string_view column) const { return directory_ / ssbColumn(column).fileName(); }
std::string SsbDataset::manifest() const {
    std::string result = std::string(magic) + "\nscale\t" + scale_ + "\ngenerator\t" + generator_ + "\n";
    for (const auto& table : ssbTables) result += "table\t" + std::string(table.name) + "\t" + std::to_string(rowCount(table.name)) + "\n";
    for (const auto& column : ssbColumns) result += columnRecord(column) + "\n";
    return result;
}
void SsbDataset::writeManifest() const { writeExclusive(directory_ / ssbManifestName, manifest()); }
SsbDataset SsbDataset::open(const std::filesystem::path& directory) {
    const auto path = directory / ssbManifestName;
    requireRegular(path);
    std::ifstream input(path, std::ios::binary);
    if (line(input) != magic) throw std::runtime_error("Unsupported SSB format; generate a native dataset with bench/prepare_ssb.sh");
    auto scale = field(input, "scale\t");
    auto generator = field(input, "generator\t");
    RowCounts rows{};
    for (std::size_t index = 0; index < ssbTables.size(); ++index)
        rows[index] = ssb_detail::parseInteger<std::uint64_t>(field(input, "table\t" + std::string(ssbTables[index].name) + "\t"));
    SsbDataset dataset(directory, std::move(scale), std::move(generator), rows);
    for (const auto& column : ssbColumns) {
        if (line(input) != columnRecord(column)) throw std::runtime_error("SSB manifest schema mismatch: " + std::string(column.name));
        const auto count = dataset.rowCount(column.table);
        if (count > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("SSB table row count overflow");
        requireRegular(dataset.columnPath(column.name));
        columnFileBytes(dataset.columnPath(column.name), static_cast<std::size_t>(count), column.storage.width);
    }
    if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing SSB manifest records");
    return dataset;
}
} // namespace db
