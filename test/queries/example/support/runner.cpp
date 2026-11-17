/** @file
 * @brief Independent native PCUDA SSB: dimension row lookup and fused Text fact aggregation.
 * @details No SQL compiler/executor or categorical encoding. Hash representatives
 * are immutable fact row references; equality compares original dimension values.
 */
#include "runner.h"
#include "crystal/device.h"
#include "cli/text_format.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace ssb {
namespace {
constexpr unsigned threads = 128, items = 4, tile = threads * items;
struct HashView { int* entries = nullptr; unsigned* occupied = nullptr; int minimum = 0; std::size_t extent = 1; };
struct BuildInput { ColumnView key, filter; Filter predicate; HashView table; };
struct FactInput { ColumnView keys[4], payloads[4]; ColumnView revenue, cost; HashView tables[4]; };
struct ScalarInput { ColumnView date, discount, quantity, price; ScalarSpec predicate; };
struct DeviceGrouping { std::array<GroupSpec, 3> keys; unsigned count; bool profit; };
template<class F> void launch(const Runtime& runtime, std::size_t rows, F body) {
    if (!rows) return;
    const auto blocks = rows / tile + (rows % tile != 0);
    runtime->validateLaunch({{blocks, 1, 1}, {threads, 1, 1}, 0});
    db::checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(blocks)), dim3(threads), 0, runtime->stream(), body), "manual SSB PCUDA launch");
}
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
            if (input.filter.is_text) { if (!input.predicate.matches(input.filter.text(row))) return; }
            else if (input.filter.values && !input.predicate.matches(input.filter.values[row])) return;
            crystal::buildPhtEntry<2>(input.key.values[row], static_cast<int>(row), input.table.entries,
                input.table.extent, input.table.minimum, input.table.occupied);
        });
    });
}
inline bool probeRow(const FactInput& input, std::size_t row, int (&values)[4]) {
    for (unsigned slot = 0; slot < 4; ++slot) {
        const auto& table = input.tables[slot];
        if (!table.entries) continue;
        if (!ProbePHT_2(input.keys[slot].values[row], values[slot], table.entries, table.extent, table.minimum, table.occupied)) return false;
    }
    return true;
}
/** @brief Hash native values; collisions are resolved by full Text/numeric equality. */
inline Word groupHash(const FactInput& input, const DeviceGrouping& spec, const int (&rows)[4]) {
    Word hash = 14695981039346656037ULL;
    for (unsigned i = 0; i < spec.count; ++i) {
        const auto slot = spec.keys[i].slot;
        const auto column = input.payloads[slot];
        const Word value = column.is_text ? db::hashText(column.text(rows[slot])) : static_cast<Word>(column.values[rows[slot]]);
        hash = (hash ^ value) * 1099511628211ULL;
    }
    return hash;
}
inline bool sameGroup(const FactInput& input, const DeviceGrouping& spec, const int (&left)[4], std::size_t representative) {
    int right[4]{};
    if (!probeRow(input, representative, right)) return false;
    for (unsigned i = 0; i < spec.count; ++i) {
        const auto slot = spec.keys[i].slot;
        const auto column = input.payloads[slot];
        if (column.is_text) { if (db::compareText(column.text(left[slot]), column.text(right[slot])) != 0) return false; }
        else if (column.values[left[slot]] != column.values[right[slot]]) return false;
    }
    return true;
}
void enqueueProbe(const Runtime& runtime, FactInput input, const QuerySpec& query, std::size_t capacity,
                  Word* sums, Word* representatives, Word* presence, unsigned* failure) {
    const DeviceGrouping spec{query.groups, query.group_count, query.profit};
    launch(runtime, input.revenue.rows, [=] {
        striped(input.revenue.rows, [&](std::size_t row) {
            int values[4]{};
            if (!probeRow(input, row, values)) return;
            auto group = groupHash(input, spec, values) % capacity;
            for (std::size_t attempt = 0; attempt < capacity; ++attempt) {
                const auto previous = atomicCAS(representatives + group, 0ULL, static_cast<Word>(row) + 1);
                if (!previous || sameGroup(input, spec, values, previous - 1)) {
                    const int64_t value = static_cast<int64_t>(input.revenue.values[row]) - (spec.profit ? static_cast<int64_t>(input.cost.values[row]) : 0);
                    db::atomicAggregate<int64_t, db::AggregateOp::Sum>(sums[group], value);
                    atomicCAS(presence + group, 0ULL, 1ULL);
                    return;
                }
                group = (group + 1) % capacity;
            }
            atomicExch(failure, 1U);
        });
    });
}
inline bool scalarSelected(const ScalarInput& input, std::size_t row) {
    return input.predicate.date.matches(input.date.values[row]) && input.predicate.discount.matches(input.discount.values[row]) &&
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
/** @brief Compact row references and sums, never reinterpret Text payloads as integer IDs. */
void enqueueCompact(const Runtime& runtime, FactInput input, const QuerySpec& query, std::size_t groups,
                    const Word* sums, const Word* representatives, const Word* presence, const Word* offsets, Word* output) {
    const DeviceGrouping spec{query.groups, query.group_count, query.profit};
    launch(runtime, groups, [=] {
        striped(groups, [&](std::size_t group) {
            if (!presence[group]) return;
            int rows[4]{};
            if (!probeRow(input, representatives[group] - 1, rows)) return;
            const auto base = offsets[group] * (spec.count + 1);
            for (unsigned i = 0; i < spec.count; ++i) output[base + i] = rows[spec.keys[i].slot];
            output[base + spec.count] = sums[group];
        });
    });
}
uint64_t magnitude(const std::vector<int>& values) {
    uint64_t maximum = 0;
    for (const int64_t value : values) maximum = std::max(maximum, static_cast<uint64_t>(value < 0 ? -value : value));
    return maximum;
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
void validateFilter(const Dataset& data, const DimensionSpec& spec, Filter filter) {
    if (data.column(spec.filter_column).is_text != filter.text) throw std::invalid_argument("Manual predicate type differs from source column");
}
KeyDomain validateDimension(const Dataset& data, const DimensionSpec& spec, unsigned slot) {
    constexpr const char* foreign_keys[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    const auto count = data.column(spec.key).rows;
    if (count > INT32_MAX) throw std::overflow_error("Manual PHT row reference exceeds Int32");
    if (data.column(spec.value).rows != count) throw std::invalid_argument("Manual payload row count differs");
    if (*spec.filter_column) {
        if (data.column(spec.filter_column).rows != count) throw std::invalid_argument("Manual predicate row count differs");
        validateFilter(data, spec, spec.filter);
    }
    if (data.column(foreign_keys[slot]).rows != data.column("lo_revenue").rows) throw std::invalid_argument("Manual foreign key row count differs");
    return keyDomain(data.host(spec.key));
}
std::size_t capacity(const Dataset& data, const QuerySpec& spec) {
    if (spec.group_count > spec.groups.size()) throw std::invalid_argument("Invalid manual group count");
    std::array<std::uint64_t, 3> cardinalities{};
    for (unsigned i = 0; i < spec.group_count; ++i) {
        const auto slot = spec.groups[i].slot;
        if (slot >= spec.dimensions.size()) throw std::invalid_argument("Invalid manual group slot");
        cardinalities[i] = data.cardinality(spec.dimensions[slot].value);
    }
    return groupCapacity(spec, cardinalities);
}
FactInput factInput(const Dataset& data, const QuerySpec& spec) {
    FactInput input{}; input.revenue = data.column("lo_revenue");
    if (spec.profit) input.cost = data.column("lo_supplycost");
    constexpr const char* names[]{"lo_orderdate", "lo_suppkey", "lo_custkey", "lo_partkey"};
    for (unsigned slot = 0; slot < 4; ++slot) if (*spec.dimensions[slot].key) {
        input.keys[slot] = data.column(names[slot]); input.payloads[slot] = data.column(spec.dimensions[slot].value);
    }
    return input;
}
void printRows(const Rows& rows) {
    std::cout << "__SSB_RESULT_BEGIN__\n";
    for (const auto& row : rows) {
        std::cout << '[';
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i) std::cout << ',';
            if (std::holds_alternative<std::string>(row[i])) std::cout << db::cli::quoteText(std::get<std::string>(row[i]));
            else std::cout << std::get<int64_t>(row[i]);
        }
        std::cout << "]\n";
    }
    std::cout << "__SSB_RESULT_END__\nResult count: " << rows.size() << '\n';
}
} // namespace
ManualQuery::ManualQuery(Dataset& data, QuerySpec spec)
    : data_(data), spec_(spec), groups_(capacity(data, spec)), sums_(data.runtime(), groups_), representatives_(data.runtime(), groups_),
      presence_(data.runtime(), groups_), offsets_(data.runtime(), groups_), result_(data.runtime()), failure_(data.runtime(), 1), scan_(data.runtime()) {
    data.runtime()->requireCapabilities(false, true, threads * sizeof(int64_t));
    if (!spec.group_count) { validateScalar(data); return; }
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
    if (slot >= spec_.dimensions.size() || !*spec_.dimensions[slot].filter_column) throw std::invalid_argument("Manual predicate slot is not active");
    validateFilter(data_, spec_.dimensions[slot], filter);
    spec_.dimensions[slot].filter = filter;
}
void ManualQuery::build() {
    for (unsigned slot = 0; slot < tables_.size(); ++slot) {
        const auto& spec = spec_.dimensions[slot];
        if (!*spec.key) continue;
        auto& table = tables_[slot]; table.occupied.zero();
        const ColumnView filter = *spec.filter_column ? data_.column(spec.filter_column) : ColumnView{};
        enqueueBuild(data_.runtime(), {data_.column(spec.key), filter, spec.filter,
            {table.entries.data(), table.occupied.data(), table.domain.minimum, table.domain.extent}});
    }
}
void ManualQuery::aggregate() {
    if (!spec_.group_count) {
        enqueueScalar(data_.runtime(), {data_.column("lo_orderdate"), data_.column("lo_discount"), data_.column("lo_quantity"),
            data_.column("lo_extendedprice"), spec_.scalar}, sums_.data());
        return;
    }
    auto input = factInput(data_, spec_);
    for (unsigned slot = 0; slot < tables_.size(); ++slot) if (*spec_.dimensions[slot].key) {
        const auto& table = tables_[slot];
        input.tables[slot] = {table.entries.data(), table.occupied.data(), table.domain.minimum, table.domain.extent};
    }
    enqueueProbe(data_.runtime(), input, spec_, groups_, sums_.data(), representatives_.data(), presence_.data(), failure_.data());
}
Rows ManualQuery::fetch() {
    if (!spec_.group_count) {
        std::vector<Word> value; sums_.copyToHost(value, 1);
        return {{db::aggregateValue<int64_t>(value.front())}};
    }
    scan_.enqueue(presence_.data(), offsets_.data(), groups_, threads);
    const auto count = scan_.finish();
    std::vector<unsigned> failure; failure_.copyToHost(failure, 1);
    if (failure.front() || count > groups_) throw std::runtime_error("Manual group table capacity exceeded");
    const auto cells = db::checkedSizeMultiply(count, spec_.group_count + 1);
    result_.ensureCapacity(cells);
    auto input = factInput(data_, spec_);
    for (unsigned slot = 0; slot < tables_.size(); ++slot) if (*spec_.dimensions[slot].key) {
        const auto& table = tables_[slot]; input.tables[slot] = {table.entries.data(), table.occupied.data(), table.domain.minimum, table.domain.extent};
    }
    if (count) enqueueCompact(data_.runtime(), input, spec_, groups_, sums_.data(), representatives_.data(), presence_.data(), offsets_.data(), result_.data());
    std::vector<Word> values; result_.copyToHost(values, cells);
    Rows rows;
    for (std::size_t row = 0; row < count; ++row) {
        std::vector<Cell> result;
        const auto base = row * (spec_.group_count + 1);
        for (unsigned col = 0; col < spec_.group_count; ++col)
            result.push_back(data_.cell(spec_.dimensions[spec_.groups[col].slot].value, values[base + col]));
        result.emplace_back(db::aggregateValue<int64_t>(values[base + spec_.group_count]));
        rows.push_back(std::move(result));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}
Rows ManualQuery::execute() {
    sums_.zero(); representatives_.zero(); presence_.zero(); failure_.zero();
    build(); aggregate(); return fetch();
}
int run(unsigned id, int argc, char** argv) {
    try {
        if (argc > 3) throw std::invalid_argument("Usage: ssb_qNN [data-directory [trials]]");
        const auto config = db::loadCrystalConfig();
        const auto directory = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path(config.data_dir);
        db::SsbDataset::open(directory); // Reject incompatible input before runtime/GPU allocation.
        std::size_t parsed = 0;
        const auto trials = argc > 2 ? std::stoul(argv[2], &parsed) : 4UL;
        if (!trials || trials > 1000 || (argc > 2 && parsed != std::string(argv[2]).size())) throw std::invalid_argument("Invalid manual SSB trial count");
        Dataset data(std::make_shared<db::DeviceRuntime>(config));
        const auto spec = querySpec(id); data.load(spec, directory); ManualQuery query(data, spec);
        for (unsigned trial = 0; trial < trials; ++trial) {
            const auto start = std::chrono::steady_clock::now(); const auto rows = query.execute(); const auto finish = std::chrono::steady_clock::now();
            printRows(rows);
            std::cout << "Trial: " << trial << " Query CPU-ready ms: " << std::chrono::duration<double, std::milli>(finish - start).count() << '\n';
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "Manual SSB error: " << error.what() << '\n'; return 1; }
}
} // namespace ssb
