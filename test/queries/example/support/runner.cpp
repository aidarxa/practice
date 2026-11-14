/** @file
 * @brief Native PCUDA manual SSB: dimension builds followed by one fused fact probe/aggregate.
 * @details No SQL parser, optimizer, generated source, module cache or engine
 * execution context is used here. Runtime/scan/device primitives are shared, not
 * the query algorithm. Resources stay in RAII owners on one explicit stream.
 */
#include "runner.h"
#include "crystal/device.h"
#include "crystal/utils.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace ssb {
namespace {
constexpr unsigned threads = 128;
constexpr unsigned items = 4;
constexpr unsigned tile = threads * items;
/** @brief Non-owning PHT view; occupancy supports zero keys and zero payloads. */
struct HashView { int* entries = nullptr; unsigned* occupied = nullptr; int minimum = 0; std::size_t extent = 1; };
struct BuildInput { ColumnView key, value, filter; Filter predicate; HashView table; };
struct FactInput { ColumnView keys[4]; ColumnView revenue, cost; HashView tables[4]; };
struct ScalarInput { ColumnView date, discount, quantity, price; ScalarSpec predicate; };
/** @brief Device grouping never carries host column-name pointers from the handwritten recipe. */
struct DeviceGrouping { std::array<GroupSpec, 3> keys; unsigned count; bool profit; };

/** @brief Checked native launch helper; no default stream and no per-kernel wait. */
template<class F> void launch(const Runtime& runtime, std::size_t rows, F body) {
    if (!rows) return;
    const auto blocks = rows / tile + (rows % tile != 0);
    runtime->validateLaunch({{blocks, 1, 1}, {threads, 1, 1}, 0});
    db::checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, runtime->stream(), body), "manual SSB PCUDA launch");
}
/** @brief Visit the original four striped rows per thread, including partial last tiles. */
template<class F> inline void striped(std::size_t rows, F operation) {
    const auto base = static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.x;
    for (unsigned item = 0; item < items; ++item) {
        const auto row = base + static_cast<std::size_t>(item) * threads;
        if (row < rows) operation(row);
    }
}
void enqueueBuild(const Runtime& runtime, BuildInput input) {
    launch(runtime, input.key.rows, [=] {
        striped(input.key.rows, [&](std::size_t row) {
            if (input.filter.values && !input.predicate.matches(input.filter.values[row])) return;
            crystal::buildPhtEntry<2>(input.key.values[row], input.value.values[row], input.table.entries,
                input.table.extent, input.table.minimum, input.table.occupied);
        });
    });
}
/** @brief All probes and the signed revenue/profit expression remain inside the fact kernel. */
inline bool probeRow(const FactInput& input, std::size_t row, int (&values)[4]) {
    for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& table = input.tables[slot];
        if (!table.entries) continue;
        if (!ProbePHT_2(input.keys[slot].values[row], values[slot], table.entries, table.extent, table.minimum, table.occupied)) return false;
    }
    return true;
}
inline std::size_t groupIndex(const DeviceGrouping& spec, const int (&values)[4]) {
    std::size_t index = 0;
    for (unsigned i = 0; i < spec.count; ++i) {
        const auto& group = spec.keys[i];
        index = index * group.extent + static_cast<std::size_t>(static_cast<int64_t>(values[group.slot]) - group.minimum);
    }
    return index;
}
void enqueueProbe(const Runtime& runtime, FactInput input, const QuerySpec& query, Word* sums, Word* presence) {
    const DeviceGrouping spec{query.groups, query.group_count, query.profit};
    launch(runtime, input.revenue.rows, [=] {
        striped(input.revenue.rows, [&](std::size_t row) {
            int values[4]{};
            if (!probeRow(input, row, values)) return;
            const auto group = groupIndex(spec, values);
            const int64_t value = static_cast<int64_t>(input.revenue.values[row]) - (spec.profit ? static_cast<int64_t>(input.cost.values[row]) : 0);
            db::atomicAggregate<int64_t, db::AggregateOp::Sum>(sums[group], value);
            atomicCAS(presence + group, 0ULL, 1ULL);
        });
    });
}
inline bool scalarSelected(const ScalarInput& input, std::size_t row) {
    return input.predicate.date.matches(input.date.values[row]) &&
        input.predicate.discount.matches(input.discount.values[row]) &&
        input.predicate.quantity.matches(input.quantity.values[row]);
}
void enqueueScalar(const Runtime& runtime, ScalarInput input, Word* output) {
    launch(runtime, input.date.rows, [=] {
        __shared__ int64_t shared[threads];
        int64_t sum = 0;
        striped(input.date.rows, [&](std::size_t row) {
            if (scalarSelected(input, row)) sum += static_cast<int64_t>(input.price.values[row]) * input.discount.values[row];
        });
        const auto total = crystal::blockReduce(sum, shared, threadIdx.x, blockDim.x, crystal::Sum<int64_t>{});
        if (threadIdx.x == 0) db::atomicAggregate<int64_t, db::AggregateOp::Sum>(*output, total);
    });
}
/** @brief Scatter only present groups, even when the signed sum is zero or negative. */
void enqueueCompact(const Runtime& runtime, const QuerySpec& query, std::size_t groups, const Word* sums,
                    const Word* presence, const Word* offsets, Word* output) {
    const DeviceGrouping spec{query.groups, query.group_count, query.profit};
    launch(runtime, groups, [=] {
        striped(groups, [&](std::size_t group) {
            if (!presence[group]) return;
            const auto base = offsets[group] * (spec.count + 1);
            auto remainder = group;
            for (unsigned i = spec.count; i > 0; --i) {
                const auto& key = spec.keys[i - 1];
                output[base + i - 1] = static_cast<Word>(static_cast<int64_t>(remainder % key.extent) + key.minimum);
                remainder /= key.extent;
            }
            output[base + spec.count] = sums[group];
        });
    });
}
uint64_t magnitude(const std::vector<int>& values) {
    uint64_t maximum = 0;
    for (const int64_t value : values) maximum = std::max(maximum, static_cast<uint64_t>(value < 0 ? -value : value));
    return maximum;
}
/** @brief Check the grouping domain of every dimension payload before a device index can use it. */
void validateGroups(const Dataset& data, const QuerySpec& spec) {
    for (unsigned i = 0; i < spec.group_count; ++i) {
        const auto& group = spec.groups[i];
        for (const int64_t value : data.host(spec.dimensions[group.slot].value)) {
            const auto local = value - group.minimum;
            if (local < 0 || static_cast<uint64_t>(local) >= group.extent) throw std::out_of_range("Manual group value exceeds its domain");
        }
    }
}
void validateScalar(const Dataset& data) {
    const auto rows = data.column("lo_orderdate").rows;
    for (const auto* name : {"lo_discount", "lo_quantity", "lo_extendedprice"})
        if (data.column(name).rows != rows) throw std::invalid_argument("Manual scalar input row counts differ");
    validateMeasureBound(rows, magnitude(data.host("lo_extendedprice")), magnitude(data.host("lo_discount")), true);
}
void validateFact(const Dataset& data, const QuerySpec& spec) {
    const auto rows = data.column("lo_revenue").rows;
    if (spec.profit && data.column("lo_supplycost").rows != rows) throw std::invalid_argument("Manual profit input row counts differ");
    validateMeasureBound(rows, magnitude(data.host("lo_revenue")), spec.profit ? magnitude(data.host("lo_supplycost")) : 0, false);
}
/** @brief All dimension/fact extents agree before any device kernel may dereference them. */
KeyDomain validateDimension(const Dataset& data, const DimensionSpec& spec, unsigned slot) {
    constexpr const char* foreign_keys[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    const auto count = data.column(spec.key).rows;
    if (data.column(spec.value).rows != count) throw std::invalid_argument("Manual payload row count differs");
    if (*spec.filter_column && data.column(spec.filter_column).rows != count) throw std::invalid_argument("Manual predicate row count differs");
    if (data.column(foreign_keys[slot]).rows != data.column("lo_revenue").rows) throw std::invalid_argument("Manual foreign key row count differs");
    return keyDomain(data.host(spec.key));
}
std::size_t tableRows(const std::string& table) {
    const std::map<std::string, std::size_t> sizes{{"LINEORDER", LO_LEN}, {"DDATE", D_LEN}, {"SUPPLIER", S_LEN}, {"CUSTOMER", C_LEN}, {"PART", P_LEN}};
    return sizes.at(table);
}
void printRows(const Rows& rows) {
    std::cout << "__SSB_RESULT_BEGIN__\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) std::cout << (i ? " " : "") << row[i];
        std::cout << '\n';
    }
    std::cout << "__SSB_RESULT_END__\nResult count: " << rows.size() << '\n';
}
}

void Dataset::add(const std::string& name, const std::vector<int>& values) {
    if (columns_.count(name)) throw std::invalid_argument("Duplicate manual input column: " + name);
    Column column{Buffer<int>(runtime_, values.size()), values};
    column.device.copyFromHost(values);
    columns_.emplace(name, std::move(column));
}
void Dataset::loadColumn(const std::string& name, const std::filesystem::path& directory) {
    if (columns_.count(name)) return;
    if (hasNullBitmapFile(name, directory)) throw std::invalid_argument("Manual SSB requires nonnullable input: " + name);
    const auto& column = db::ssbColumn(name);
    add(name, db::loadNumericColumn(directory / column.fileName(), tableRows(column.table), column.storage));
}
void Dataset::load(const QuerySpec& spec, const std::filesystem::path& directory) {
    if (!spec.group_count) {
        for (const auto* name : {"lo_orderdate", "lo_discount", "lo_quantity", "lo_extendedprice"}) loadColumn(name, directory);
        return;
    }
    constexpr const char* foreign_keys[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    loadColumn("lo_revenue", directory);
    if (spec.profit) loadColumn("lo_supplycost", directory);
    for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& dimension = spec.dimensions[slot];
        if (!*dimension.key) continue;
        for (const auto* name : {dimension.key, dimension.value, foreign_keys[slot]}) loadColumn(name, directory);
        if (*dimension.filter_column) loadColumn(dimension.filter_column, directory);
    }
}
ColumnView Dataset::column(const std::string& name) const {
    const auto& column = columns_.at(name);
    return {column.device.data(), column.host.size()};
}
const std::vector<int>& Dataset::host(const std::string& name) const { return columns_.at(name).host; }

ManualQuery::ManualQuery(Dataset& data, QuerySpec spec)
    : data_(data), spec_(spec), groups_(groupCapacity(spec)), sums_(data.runtime(), groups_),
      presence_(data.runtime(), groups_), offsets_(data.runtime(), groups_), result_(data.runtime()), scan_(data.runtime()) {
    data.runtime()->requireCapabilities(false, true, threads * sizeof(int64_t));
    if (!spec.group_count) {
        validateScalar(data);
        return;
    }
    validateGroups(data, spec);
    validateFact(data, spec);
    prepareTables();
}
void ManualQuery::prepareTables() {
    for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& dimension = spec_.dimensions[slot];
        const auto domain = *dimension.key ? validateDimension(data_, dimension, slot) : KeyDomain{};
        const auto extent = *dimension.key ? domain.extent : 0;
        tables_.push_back({Buffer<int>(data_.runtime(), db::checkedSizeMultiply(extent, 2)), Buffer<unsigned>(data_.runtime(), extent), domain});
    }
}
void ManualQuery::setFilter(unsigned slot, Filter filter) {
    if (slot >= spec_.dimensions.size() || !*spec_.dimensions[slot].filter_column)
        throw std::invalid_argument("Manual predicate slot is not active");
    spec_.dimensions[slot].filter = filter;
}
void ManualQuery::build() {
    for (unsigned slot = 0; slot < tables_.size(); ++slot) {
        const auto& spec = spec_.dimensions[slot];
        if (!*spec.key) continue;
        auto& table = tables_[slot];
        table.occupied.zero();
        const ColumnView filter = *spec.filter_column ? data_.column(spec.filter_column) : ColumnView{};
        enqueueBuild(data_.runtime(), {data_.column(spec.key), data_.column(spec.value), filter, spec.filter,
            {table.entries.data(), table.occupied.data(), table.domain.minimum, table.domain.extent}});
    }
}
void ManualQuery::aggregate() {
    if (!spec_.group_count) {
        enqueueScalar(data_.runtime(), {data_.column("lo_orderdate"), data_.column("lo_discount"), data_.column("lo_quantity"),
            data_.column("lo_extendedprice"), spec_.scalar}, sums_.data());
        return;
    }
    FactInput input{};
    input.revenue = data_.column("lo_revenue");
    if (spec_.profit) input.cost = data_.column("lo_supplycost");
    constexpr const char* names[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    for (unsigned slot = 0; slot < 4; ++slot) {
        if (!*spec_.dimensions[slot].key) continue;
        const auto& table = tables_[slot];
        input.keys[slot] = data_.column(names[slot]);
        input.tables[slot] = {table.entries.data(), table.occupied.data(), table.domain.minimum, table.domain.extent};
    }
    enqueueProbe(data_.runtime(), input, spec_, sums_.data(), presence_.data());
}
Rows ManualQuery::fetch() {
    if (!spec_.group_count) {
        std::vector<Word> value;
        sums_.copyToHost(value, 1);
        return {{db::aggregateValue<int64_t>(value.front())}};
    }
    scan_.enqueue(presence_.data(), offsets_.data(), groups_, threads);
    const auto count = scan_.finish(); // CPU needs the count before exact output allocation.
    if (count > groups_) throw std::runtime_error("Invalid manual group count");
    const auto cells = db::checkedSizeMultiply(count, spec_.group_count + 1);
    result_.ensureCapacity(cells);
    if (count) enqueueCompact(data_.runtime(), spec_, groups_, sums_.data(), presence_.data(), offsets_.data(), result_.data());
    std::vector<Word> values;
    result_.copyToHost(values, cells);
    Rows rows(count, std::vector<int64_t>(spec_.group_count + 1));
    for (std::size_t row = 0; row < count; ++row)
        for (unsigned col = 0; col <= spec_.group_count; ++col) rows[row][col] = db::aggregateValue<int64_t>(values[row * (spec_.group_count + 1) + col]);
    return rows;
}
Rows ManualQuery::execute() {
    sums_.zero();
    presence_.zero();
    build();
    aggregate();
    return fetch();
}
int run(unsigned id, int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("Usage: ssb_qNN [data-directory [trials]]");
        const auto directory = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path(DATA_DIR);
        std::size_t parsed = 0;
        const auto trials = argc > 2 ? std::stoul(argv[2], &parsed) : 4UL;
        if (!trials || trials > 1000 || (argc > 2 && parsed != std::string(argv[2]).size())) throw std::invalid_argument("Invalid manual SSB trial count");
        Dataset data(std::make_shared<db::DeviceRuntime>(db::loadCrystalConfig()));
        const auto spec = querySpec(id);
        data.load(spec, directory);
        ManualQuery query(data, spec);
        for (unsigned trial = 0; trial < trials; ++trial) {
            const auto start = std::chrono::steady_clock::now();
            const auto rows = query.execute();
            const auto finish = std::chrono::steady_clock::now();
            printRows(rows);
            std::cout << "Trial: " << trial << " Query CPU-ready ms: " << std::chrono::duration<double, std::milli>(finish - start).count() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Manual SSB error: " << error.what() << '\n';
        return 1;
    }
}
}
