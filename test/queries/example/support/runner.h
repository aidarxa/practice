#pragma once
/** @file @brief RAII resources and execution interface for the independent manual PCUDA SSB. */
#include "query_spec.h"
#include "core/device_buffer.h"
#include "core/pcuda_scan.h"
#include "core/text_storage.h"
#include "storage/ssb_dataset.h"
#include <filesystem>
#include <map>
#include <string>
#include <variant>

namespace ssb {
using Runtime = std::shared_ptr<db::DeviceRuntime>;
using Word = unsigned long long;
template<class T> using Buffer = db::DynamicDeviceBuffer<T>;
/** @brief POD kernel view; its Dataset owner outlives all dependent stream operations. */
struct ColumnView {
    const int* values = nullptr;
    std::size_t rows = 0;
    const std::uint8_t* bytes = nullptr;
    const std::int32_t* offsets32 = nullptr;
    const std::uint64_t* offsets64 = nullptr;
    bool is_text = false;
    db::TextView text(std::size_t row) const {
        const auto begin = offsets64 ? offsets64[row] : static_cast<std::uint64_t>(offsets32[row]);
        const auto end = offsets64 ? offsets64[row + 1] : static_cast<std::uint64_t>(offsets32[row + 1]);
        return {bytes ? bytes + begin : nullptr, end - begin};
    }
};
using Cell = std::variant<int64_t, std::string>;
/** @brief Native SSB input owner; the handwritten benchmark accepts nonnullable source columns only.
 * @details SQL nullable/3VL acceptance belongs to the real engine fixtures. Manual
 * examples reject a supplied null bitmap rather than quietly treating NULL as zero.
 */
class Dataset {
public:
    explicit Dataset(Runtime runtime) : runtime_(std::move(runtime)) {}
    void add(const std::string& name, const std::vector<int>& values);
    void addText(const std::string& name, const std::vector<std::string>& values);
    void load(const QuerySpec& spec, const std::filesystem::path& directory);
    ColumnView column(const std::string& name) const;
    const std::vector<int>& host(const std::string& name) const;
    Cell cell(const std::string& name, std::size_t row) const;
    std::uint64_t cardinality(const std::string& name) const;
    Runtime runtime() const { return runtime_; }
private:
    struct Column {
        Buffer<int> device;
        std::vector<int> host;
    };
    Runtime runtime_;
    std::map<std::string, Column> columns_;
    struct TextColumn { db::LoadedTextColumn device; std::vector<std::string> host; };
    std::map<std::string, TextColumn> text_columns_;
    void loadColumn(const std::string& name, const db::SsbDataset& dataset);
};
/** @brief Compact CPU result after GPU count/scan/write; keys precede the signed measure. */
using Rows = std::vector<std::vector<Cell>>;
/** @brief Reusable query workspace: build/probe, presence, scan and exact compact result capacity. */
class ManualQuery {
public:
    ManualQuery(Dataset& data, QuerySpec spec);
    /** @brief Rebind only a dimension predicate's values; reuse PHT/result/scan allocations and native kernels. */
    void setFilter(unsigned slot, Filter filter);
    Rows execute();
private:
    struct HashTable {
        Buffer<int> entries;
        Buffer<unsigned> occupied;
        KeyDomain domain;
    };
    Dataset& data_;
    QuerySpec spec_;
    std::size_t groups_;
    std::vector<HashTable> tables_;
    Buffer<Word> sums_, representatives_, presence_, offsets_, result_;
    Buffer<unsigned> failure_;
    db::ExclusiveScan scan_;
    void prepareTables();
    void build();
    void aggregate();
    Rows fetch();
};
/** @brief CLI entry for each original example: optional data directory and trial count. */
int run(unsigned id, int argc, char** argv);
}
