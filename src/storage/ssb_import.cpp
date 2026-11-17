/** @file @brief Bounded dbgen conversion; no dictionary coding, GPU or SQL dependencies. */
#include "storage/ssb_import.h"
#include "storage/text_column.h"
#include "core/size_expression.h"
#include "ssb_parse.h"
#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <sys/syscall.h>
#include <system_error>
#include <unordered_set>
#include <unistd.h>

namespace db {
namespace {
/** @brief Own only a fresh sibling directory and atomically publish it without replacement. */
class Staging {
public:
    explicit Staging(const std::filesystem::path& output) {
        if (std::filesystem::exists(std::filesystem::symlink_status(output))) throw std::runtime_error("Output already exists: " + output.string());
        auto pattern = (output.parent_path() / ("." + output.filename().string() + ".import-XXXXXX")).string();
        const auto created = ::mkdtemp(pattern.data());
        if (!created) throw std::system_error(errno, std::generic_category(), "Create import staging directory");
        path = created;
    }
    ~Staging() { if (!path.empty()) { std::error_code error; std::filesystem::remove_all(path, error); } }
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
    void publish(const std::filesystem::path& output) {
        // RENAME_NOREPLACE, unlike filesystem::rename, cannot overwrite an empty directory.
        if (::syscall(SYS_renameat2, AT_FDCWD, path.c_str(), AT_FDCWD, output.c_str(), 1U) != 0)
            throw std::system_error(errno, std::generic_category(), "Publish SSB dataset without replacement");
        path.clear();
    }
    std::filesystem::path path;
};
/** @brief Each column owns a bounded write buffer; flush errors cannot be silently ignored. */
class ColumnOutput {
public:
    explicit ColumnOutput(const std::filesystem::path& path, ColumnStorage storage) : file_(path, std::ios::binary), storage_(storage) {
        if (!file_) throw std::runtime_error("Cannot create column: " + path.string());
        bytes_.reserve(bufferBytes);
    }
    void append(std::string_view value) {
        if (bytes_.size() + storage_.width > bufferBytes) flush();
        const auto at = bytes_.size(); bytes_.resize(at + storage_.width, 0);
        if (storage_.encoding == PhysicalEncoding::Int32LE) encodeInteger(value, bytes_.data() + at);
        else if (storage_.encoding == PhysicalEncoding::AsciiBoolean) {
            if (value != "0" && value != "1") throw std::runtime_error("Invalid ASCII boolean");
            bytes_[at] = value.front();
        } else {
            if (value.size() > storage_.width) throw std::runtime_error("Text exceeds physical field width");
            validateText(value);
            std::copy(value.begin(), value.end(), bytes_.begin() + at);
        }
    }
    void close() { flush(); file_.close(); if (!file_) throw std::runtime_error("Failed to close SSB column"); }
private:
    static constexpr std::size_t bufferBytes = 1024 * 1024;
    std::ofstream file_;
    ColumnStorage storage_;
    std::vector<char> bytes_;
    static void encodeInteger(std::string_view value, char* target) {
        const auto word = static_cast<std::uint32_t>(ssb_detail::parseInteger<std::int32_t>(value));
        for (unsigned byte = 0; byte < 4; ++byte) target[byte] = static_cast<char>((word >> (byte * 8U)) & 255U);
    }
    void flush() {
        file_.write(bytes_.data(), static_cast<std::streamsize>(bytes_.size()));
        if (!file_) throw std::runtime_error("Failed to write SSB column");
        bytes_.clear();
    }
};
void convertRow(std::string_view row, std::vector<ColumnOutput>& columns) {
    for (auto& column : columns) {
        const auto separator = row.find('|');
        if (separator == std::string_view::npos) throw std::runtime_error("Missing field or final pipe delimiter");
        column.append(row.substr(0, separator));
        row.remove_prefix(separator + 1);
    }
    if (!row.empty()) throw std::runtime_error("Extra fields after the final pipe delimiter");
}
/** @brief A native manifest may advertise dimension PKs only after uniqueness validation. */
void validatePrimaryKey(const SsbTable& table, std::string_view row, std::unordered_set<std::int32_t>& keys) {
    if (!*table.primary_key) return;
    const auto key = ssb_detail::parseInteger<std::int32_t>(row.substr(0, row.find('|')));
    if (!keys.insert(key).second) throw std::runtime_error("Duplicate dimension primary key");
}
std::uint64_t convertTable(const SsbTable& table, const std::filesystem::path& input, const std::filesystem::path& output) {
    std::ifstream file(input / table.input, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open dbgen table: " + (input / table.input).string());
    std::vector<ColumnOutput> columns;
    for (const auto& column : ssbColumns)
        if (std::string_view(column.table) == table.name) columns.emplace_back(output / column.fileName(), column.storage);
    std::uint64_t rows = 0;
    std::unordered_set<std::int32_t> keys;
    char buffer[4096];
    while (file.getline(buffer, sizeof(buffer))) {
        rows = checkedSizeAdd(rows, 1);
        // gcount includes the consumed newline, but not an absent final newline.
        const auto length = static_cast<std::size_t>(file.gcount()) - (file.eof() ? 0U : 1U);
        try {
            const std::string_view row(buffer, length);
            convertRow(row, columns);
            validatePrimaryKey(table, row, keys);
        }
        catch (const std::exception& error) { throw std::runtime_error(std::string(table.input) + ":" + std::to_string(rows) + ": " + error.what()); }
    }
    if (!file.eof()) throw std::runtime_error(std::string(table.input) + ": read failure or oversized row");
    for (auto& column : columns) column.close();
    return rows;
}
} // namespace
SsbDataset importSsbTables(const std::filesystem::path& input, const std::filesystem::path& output,
                           const std::string& scale, const std::string& generator) {
    validateSsbScale(scale);
    const auto destination = std::filesystem::absolute(output).lexically_normal();
    Staging staging(destination);
    // Validate metadata before spending work on large source tables.
    SsbDataset(staging.path, scale, generator, {});
    SsbDataset::RowCounts rows{};
    for (std::size_t index = 0; index < ssbTables.size(); ++index)
        rows[index] = convertTable(ssbTables[index], input, staging.path);
    SsbDataset(staging.path, scale, generator, rows).writeManifest();
    SsbDataset::open(staging.path);
    staging.publish(destination);
    return SsbDataset::open(destination);
}
} // namespace db
