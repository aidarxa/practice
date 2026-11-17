/** @file @brief Native manual SSB input ownership and host-only extent/cardinality validation. */
#include "runner.h"
#include "crystal/utils.h"
#include <unordered_set>

namespace ssb {
void Dataset::add(const std::string& name, const std::vector<int>& values) {
    if (columns_.count(name) || text_columns_.count(name)) throw std::invalid_argument("Duplicate manual input column: " + name);
    db::requireNumericStorage(db::ssbColumn(name).storage);
    Column column{Buffer<int>(runtime_, values.size()), values};
    column.device.copyFromHost(values);
    columns_.emplace(name, std::move(column));
}
void Dataset::addText(const std::string& name, const std::vector<std::string>& values) {
    if (columns_.count(name) || text_columns_.count(name)) throw std::invalid_argument("Duplicate manual input column: " + name);
    if (db::ssbColumn(name).storage.isNumeric()) throw std::invalid_argument("Manual Text input requires a Text column");
    db::HostTextColumn normalized; normalized.offsets.push_back(0);
    for (const auto& value : values) {
        db::validateText(value);
        const auto end = db::checkedSizeAdd(normalized.bytes.size(), value.size());
        normalized.bytes.insert(normalized.bytes.end(), value.begin(), value.end());
        normalized.offsets.push_back(end);
    }
    auto uploaded = db::uploadTextColumn(runtime_, normalized, normalized.bytes.size() > INT32_MAX);
    text_columns_.emplace(name, TextColumn{std::move(uploaded), values});
}
void Dataset::loadColumn(const std::string& name, const db::SsbDataset& dataset) {
    if (columns_.count(name) || text_columns_.count(name)) return;
    if (hasNullBitmapFile(name, dataset.directory())) throw std::invalid_argument("Manual SSB requires nonnullable input: " + name);
    const auto& column = db::ssbColumn(name);
    const auto rows = dataset.rowCount(column.table);
    if (column.storage.isNumeric()) add(name, db::loadNumericColumn(dataset.columnPath(name), rows, column.storage));
    else {
        const auto data = db::loadTextColumn(dataset.columnPath(name), rows, column.storage);
        std::vector<std::string> values;
        values.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row)
            values.emplace_back(data.bytes.begin() + data.offsets[row], data.bytes.begin() + data.offsets[row + 1]);
        addText(name, values);
    }
}
void Dataset::load(const QuerySpec& spec, const std::filesystem::path& directory) {
    const auto dataset = db::SsbDataset::open(directory);
    if (!spec.group_count) {
        for (const auto* name : {"lo_orderdate", "lo_discount", "lo_quantity", "lo_extendedprice"}) loadColumn(name, dataset);
        return;
    }
    constexpr const char* foreign_keys[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    loadColumn("lo_revenue", dataset);
    if (spec.profit) loadColumn("lo_supplycost", dataset);
    for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& dimension = spec.dimensions[slot];
        if (!*dimension.key) continue;
        for (const auto* name : {dimension.key, dimension.value, foreign_keys[slot]}) loadColumn(name, dataset);
        if (*dimension.filter_column) loadColumn(dimension.filter_column, dataset);
    }
}
ColumnView Dataset::column(const std::string& name) const {
    const auto found = text_columns_.find(name);
    if (found != text_columns_.end()) {
        const auto& device = found->second.device;
        return {nullptr, found->second.host.size(), device.bytes->data(), device.offsets32 ? device.offsets32->data() : nullptr,
            device.offsets64 ? device.offsets64->data() : nullptr, true};
    }
    const auto& column = columns_.at(name);
    return {column.device.data(), column.host.size()};
}
const std::vector<int>& Dataset::host(const std::string& name) const { return columns_.at(name).host; }
Cell Dataset::cell(const std::string& name, std::size_t row) const {
    if (text_columns_.count(name)) return text_columns_.at(name).host.at(row);
    return static_cast<int64_t>(host(name).at(row));
}
std::uint64_t Dataset::cardinality(const std::string& name) const {
    if (text_columns_.count(name)) {
        const auto& values = text_columns_.at(name).host;
        return std::unordered_set<std::string>(values.begin(), values.end()).size();
    }
    const auto& values = host(name);
    return std::unordered_set<int>(values.begin(), values.end()).size();
}
} // namespace ssb
