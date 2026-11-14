/** @file
 * @brief Independent numeric oracles for runtime parameters on the actual PCUDA SQL path.
 * @details No device helper is used to compute expectations. Assertions remain enabled in Release.
 */
#include "core/execution.h"
#include <cassert>
#include <iostream>
#include <optional>
#include <variant>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

using namespace db;
using Cell = std::optional<std::variant<int64_t, uint64_t, double>>;
using Rows = std::vector<std::vector<Cell>>;

/** @brief One writer owns a tiny catalog, real GPU columns and the process-only SQL cache. */
class ParameterFixture {
public:
    std::shared_ptr<Catalog> catalog = std::make_shared<Catalog>();
    std::shared_ptr<DeviceRuntime> runtime = std::make_shared<DeviceRuntime>();
    ExecutionContext context;
    QueryEngine engine{catalog, std::make_unique<ProcessModuleCache>(),
                       std::make_unique<AdaptiveCppCompiler>(), std::make_unique<DynamicLibraryExecutor>()};

    explicit ParameterFixture(size_t count = 9, bool nullable = false) {
        context.runtime_ = runtime;
        context.result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime, 0);
        context.result_validity_buffer_ = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, 0);
        TableMetadata fact("LINEORDER", {"lo_quantity", "lo_revenue", "lo_suppkey"}, count, true);
        fact.setColumnStats("lo_quantity", {0, static_cast<int64_t>(count ? count - 1 : 0), std::max<size_t>(count, 1)});
        fact.setColumnNullable("lo_revenue", nullable);
        fact.setColumnNullable("lo_suppkey", nullable);
        catalog->pushTableMetadata(fact);
        std::vector<int> quantities(count), revenues(count), keys(count);
        for (size_t i = 0; i < count; ++i) { quantities[i] = i; revenues[i] = static_cast<int>(i % 9) - 4; keys[i] = i % 4; }
        input("lo_quantity", quantities);
        input("lo_revenue", revenues);
        input("lo_suppkey", keys);
        if (nullable) {
            auto& column = context.base_columns_.at("d_lo_revenue");
            std::vector<uint64_t> bitmap((count + 63) / 64, 0);
            for (size_t i = 0; i < count; ++i) if (i % 3) bitmap[i / 64] |= 1ULL << (i % 64);
            column.validity = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, bitmap.size());
            column.validity->copyFromHost(bitmap);
        }
        TableMetadata dimension("SUPPLIER", {"s_suppkey", "s_region"}, 4, false);
        dimension.setColumnStats("s_suppkey", {0, 2, 3}); // duplicates deliberately select MHT
        catalog->pushTableMetadata(dimension);
        input("s_suppkey", {0, 0, 1, 2});
        input("s_region", {3, 5, 7, 9});
    }

    void input(const std::string& name, const std::vector<int>& values) {
        LoadedDeviceColumn column;
        column.values = std::make_unique<DynamicDeviceBuffer<int>>(runtime, values.size());
        column.values->copyFromHost(values);
        context.base_columns_.emplace("d_" + name, std::move(column));
    }

    Rows query(const std::string& sql) {
        std::cout << sql << std::endl;
        engine.executeQuery(sql, &context);
        assert(context.result_is_columnar_);
        Rows rows(context.result_row_count_, std::vector<Cell>(context.result_column_count_));
        for (size_t c = 0; c < context.result_column_count_; ++c) {
            auto& storage = context.result_column_storage_[c];
            std::vector<uint64_t> valid;
            storage.validity->copyToHost(valid, (rows.size() + 63) / 64);
            const auto read = [&](auto* buffer) {
                using T = std::remove_pointer_t<decltype(buffer->data())>;
                std::vector<T> values;
                buffer->copyToHost(values, rows.size());
                for (size_t r = 0; r < rows.size(); ++r)
                    if (!context.result_columns_[c].nullable || (valid[r / 64] & (1ULL << (r % 64))))
                        rows[r][c] = values[r];
            };
            if (storage.type == LogicalType::Int64) read(storage.i64.get());
            else if (storage.type == LogicalType::UInt64) read(storage.u64.get());
            else read(storage.f64.get());
        }
        return rows;
    }

    void noBuild() const {
        const auto& e = context.timing_.module_events;
        assert(e.plan_codegen == 0 && e.module_codegen == 0 && e.compiler == 0 && e.linker == 0 && e.loads == 0 && e.symbols == 0);
    }
    /** @brief A new type schema may lower a physical plan, but a ready fallback emits/builds no module. */
    void noModuleBuild() const {
        const auto& e = context.timing_.module_events;
        assert(e.module_codegen == 0 && e.compiler == 0 && e.linker == 0 && e.loads == 0 && e.symbols == 0);
    }
};

static Cell integer(int64_t value) { return Cell{std::variant<int64_t, uint64_t, double>{value}}; }
static Cell count(uint64_t value) { return Cell{std::variant<int64_t, uint64_t, double>{value}}; }

/** @brief Exact, binary-representable fractional inputs make rounding-independent expectations possible. */
static void numericAggregates() {
    ParameterFixture fixture;
    const auto sql = [](const std::string& increment, int threshold) {
        const auto expr = "lo_revenue + " + increment;
        return "SELECT SUM(" + expr + "),MIN(" + expr + "),MAX(" + expr + "),AVG(" + expr + "),COUNT(" + expr + ") FROM lineorder WHERE lo_quantity >= " + std::to_string(threshold) + " ORDER BY 1";
    };
    assert(fixture.query(sql("0.5", 0)) == Rows({{Cell{4.5}, Cell{-3.5}, Cell{4.5}, Cell{0.5}, count(9)}}));
    assert(fixture.query(sql("-0.5", 0)) == Rows({{Cell{-4.5}, Cell{-4.5}, Cell{3.5}, Cell{-0.5}, count(9)}}));
    fixture.noBuild();
    assert(fixture.query(sql("-0.5", 100)) == Rows({{std::nullopt, std::nullopt, std::nullopt, std::nullopt, count(0)}}));
    fixture.noBuild();
    assert(fixture.query(sql("NULL", 0)) == Rows({{std::nullopt, std::nullopt, std::nullopt, std::nullopt, count(0)}}));
    fixture.noBuild();
    assert(fixture.query(sql("0", 0)) == Rows({{integer(0), integer(-4), integer(4), Cell{0.0}, count(9)}}));
    assert(fixture.query(sql("-10", 0)) == Rows({{integer(-90), integer(-14), integer(-6), Cell{-10.0}, count(9)}}));
    fixture.noBuild();
    assert(fixture.query("SELECT SUM(lo_revenue*lo_revenue*lo_revenue*lo_revenue) FROM lineorder ORDER BY 1") == Rows({{integer(708)}}));
    assert(fixture.query("SELECT MIN(9007199254740993),MAX(9007199254740993) FROM lineorder ORDER BY 1") == Rows({{integer(9007199254740993LL), integer(9007199254740993LL)}}));
    assert(fixture.query("SELECT MIN(NULL),MAX(NULL) FROM lineorder ORDER BY 1") == Rows({{std::nullopt, std::nullopt}}));
    fixture.noBuild();
}

/** @brief Both vector local-group and direct atomic paths must agree with these independent rows. */
static void groupedAggregates() {
    ParameterFixture fixture;
    const auto sql = [](const std::string& increment, int threshold) {
        const auto expr = "lo_revenue+" + increment;
        return "SELECT lo_quantity,SUM(" + expr + "),MIN(" + expr + "),MAX(" + expr + "),AVG(" + expr + ") FROM lineorder GROUP BY lo_quantity HAVING SUM(" + expr + ")>" + std::to_string(threshold) + " ORDER BY 1 LIMIT 100";
    };
    Rows expected;
    for (int64_t i = 5; i < 9; ++i) expected.push_back({integer(i), Cell{static_cast<double>(i) - 3.5}, Cell{static_cast<double>(i) - 3.5}, Cell{static_cast<double>(i) - 3.5}, Cell{static_cast<double>(i) - 3.5}});
    assert(fixture.query(sql("0.5", 1)) == expected);
    assert(fixture.query(sql("0.5", 100)).empty()); fixture.noBuild();
    expected.clear();
    for (int64_t i = 0; i < 9; ++i) expected.push_back({integer(i), Cell{static_cast<double>(i) - 4.5}, Cell{static_cast<double>(i) - 4.5}, Cell{static_cast<double>(i) - 4.5}, Cell{static_cast<double>(i) - 4.5}});
    assert(fixture.query(sql("-0.5", -100)) == expected); fixture.noBuild();
    const auto local = [](const std::string& value) {
        return "SELECT CASE WHEN lo_quantity<4 THEN 0 ELSE 1 END,SUM(lo_revenue+" + value + ") FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN 0 ELSE 1 END ORDER BY 1";
    };
    assert(fixture.query(local("0.5")) == Rows({{integer(0), Cell{-8.0}}, {integer(1), Cell{12.5}}}));
    assert(fixture.query(local("-0.5")) == Rows({{integer(0), Cell{-12.0}}, {integer(1), Cell{7.5}}})); fixture.noBuild();
    assert(fixture.query("SELECT CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END,COUNT(NULL) FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END ORDER BY 1") == Rows({{std::nullopt, count(0)}}));
    assert(fixture.query("SELECT CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END,COUNT(NULL) FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END HAVING COUNT(NULL)=0 ORDER BY 1") == Rows({{std::nullopt, count(0)}}));
    assert(fixture.query("SELECT CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END,COUNT(NULL) FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END HAVING COUNT(NULL)>-1 ORDER BY 1") == Rows({{std::nullopt, count(0)}}));
    assert(fixture.query("SELECT CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END,COUNT(NULL) FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END HAVING COUNT(NULL)+-1<0 ORDER BY 1") == Rows({{std::nullopt, count(0)}}));
}

/** @brief Nonunique joins use item-mode updates; a CPU nested loop defines multiplicity and numbers. */
static void joinedAggregates() {
    ParameterFixture fixture;
    const auto sql = [](const std::string& add) {
        return "SELECT SUM(lo_revenue+" + add + "),MIN(lo_revenue+" + add + "),MAX(lo_revenue+" + add + "),AVG(lo_revenue+" + add + "),COUNT(*) FROM lineorder,supplier WHERE lo_suppkey=s_suppkey ORDER BY 1";
    };
    assert(fixture.query(sql("0.5")) == Rows({{Cell{3.0}, Cell{-3.5}, Cell{4.5}, Cell{0.3}, count(10)}}));
    assert(fixture.query(sql("-0.5")) == Rows({{Cell{-7.0}, Cell{-4.5}, Cell{3.5}, Cell{-0.7}, count(10)}})); fixture.noBuild();
    assert(fixture.query(sql("NULL")) == Rows({{std::nullopt, std::nullopt, std::nullopt, std::nullopt, count(10)}})); fixture.noBuild();
}

/** @brief Nullable/empty inputs must not retain identities, stale values or validity from earlier executions. */
static void nullableInputs() {
    ParameterFixture fixture(9, true);
    const auto sql = [](int threshold) {
        return "SELECT COUNT(*),COUNT(lo_revenue),SUM(lo_revenue),MIN(lo_revenue),MAX(lo_revenue),AVG(lo_revenue) FROM lineorder WHERE lo_quantity>=" + std::to_string(threshold) + " ORDER BY 1";
    };
    assert(fixture.query(sql(0)) == Rows({{count(9), count(6), integer(3), integer(-3), integer(4), Cell{0.5}}}));
    assert(fixture.query(sql(9)) == Rows({{count(0), count(0), std::nullopt, std::nullopt, std::nullopt, std::nullopt}})); fixture.noBuild();
    assert(fixture.query(sql(0)) == Rows({{count(9), count(6), integer(3), integer(-3), integer(4), Cell{0.5}}})); fixture.noBuild();
    assert(fixture.query("SELECT lo_quantity,MIN(lo_revenue) FROM lineorder WHERE lo_quantity<1 GROUP BY lo_quantity ORDER BY 1") == Rows({{integer(0), std::nullopt}}));
    ParameterFixture empty(0);
    assert(empty.query(sql(0)) == Rows({{count(0), count(0), std::nullopt, std::nullopt, std::nullopt, std::nullopt}}));
}

/** @brief All sixteen fractional predicate schemas share fixed output layout and the bounded family. */
static void boundedTypes(bool arithmetic, bool case_value = false) {
    ParameterFixture fixture;
    const auto sql = [arithmetic, case_value](unsigned mask) {
        std::vector<std::string> values;
        const std::string integers[] = {"-1", "9", "-5", "5"};
        const std::string floats[] = {"-0.5", "8.5", "-4.5", "4.5"};
        for (unsigned i = 0; i < 4; ++i)
            values.push_back(arithmetic ? std::to_string(i + 1) + ((mask & (1U << i)) ? ".25" : "") : (mask & (1U << i)) ? floats[i] : integers[i]);
        if (arithmetic)
            return "SELECT SUM(lo_revenue*0.5+" + values[0] + "+" + values[1] + "+" + values[2] + "+" + values[3] +
                (case_value ? "+(CASE WHEN lo_quantity<4 THEN 1 ELSE 1.5 END)/2" : "") + ") FROM lineorder ORDER BY 1";
        return "SELECT COUNT(*) FROM lineorder WHERE lo_quantity>=" + values[0] + " AND lo_quantity<" + values[1] + " AND lo_revenue>" + values[2] + " AND lo_revenue<" + values[3] + " ORDER BY 1";
    };
    size_t firstBuild = 0;
    for (unsigned round = 0; round < 2; ++round) {
        for (unsigned mask = 0; mask < 16; ++mask) {
            unsigned bits = 0;
            for (unsigned i = 0; i < 4; ++i) bits += (mask >> i) & 1;
            const auto expected = arithmetic ? Rows{{Cell{90.0 + 2.25 * bits + (case_value ? 5.75 : 0.0)}}} : Rows{{count(9)}};
            assert(fixture.query(sql(mask)) == expected);
            if (round) fixture.noBuild();
            else if (mask > 8) fixture.noModuleBuild();
            if (!round && !mask) firstBuild = fixture.engine.moduleCounters().compiler;
        }
    }
    assert(fixture.engine.moduleCounters().compiler == firstBuild + 8); // seven further typed + one fallback
    if (case_value) return;
    if (arithmetic) {
        assert(fixture.query("SELECT SUM(lo_revenue*0.5+NULL+NULL+NULL+NULL) FROM lineorder ORDER BY 1") == Rows({{std::nullopt}}));
    } else {
        assert(fixture.query("SELECT COUNT(*) FROM lineorder WHERE lo_quantity>=NULL AND lo_quantity<NULL AND lo_revenue>NULL AND lo_revenue<NULL ORDER BY 1") == Rows({{count(0)}}));
    }
    fixture.noBuild();
}

/** @brief Type changes may create a layout variant; compatible values and typed NULL may not. */
static void outputLayouts() {
    ParameterFixture fixture;
    const auto sql = [](const std::string& value) { return "SELECT lo_revenue+" + value + " FROM lineorder ORDER BY 1 LIMIT 1"; };
    assert(fixture.query(sql("0.5")) == Rows({{Cell{-3.5}}}));
    assert(fixture.query(sql("-0.5")) == Rows({{Cell{-4.5}}})); fixture.noBuild();
    assert(fixture.query(sql("9007199254740993")) == Rows({{integer(9007199254740989LL)}}));
    assert(fixture.query(sql("NULL")) == Rows({{std::nullopt}})); fixture.noBuild();
    assert(fixture.context.result_columns_[0].type == LogicalType::Int64);
    assert(fixture.query(sql("0.5")) == Rows({{Cell{-3.5}}})); fixture.noBuild();
    assert(fixture.query(sql("NULL")) == Rows({{std::nullopt}})); fixture.noBuild();
    assert(fixture.context.result_columns_[0].type == LogicalType::Float64);
    ParameterFixture firstNull;
    assert(firstNull.query("SELECT NULL FROM lineorder ORDER BY 1 LIMIT 1") == Rows({{std::nullopt}}));
    assert(firstNull.query("SELECT 9007199254740993 FROM lineorder ORDER BY 1 LIMIT 1") == Rows({{integer(9007199254740993LL)}}));
    assert(firstNull.query("SELECT NULL FROM lineorder ORDER BY 1 LIMIT 1") == Rows({{std::nullopt}})); firstNull.noBuild();
}

/** @brief A negative COUNT expression stays negative before and after switching to fallback. */
static void boundedCountTypes() {
    ParameterFixture fixture;
    const auto sql = [](unsigned mask, bool nulls = false) {
        std::vector<std::string> values;
        for (unsigned i = 0; i < 4; ++i)
            values.push_back(nulls ? "NULL" : i == 0 ? ((mask & 1) ? "-2.25" : "-2") : ((mask & (1U << i)) ? "0.25" : "0"));
        return "SELECT lo_quantity,COUNT(*) FROM lineorder GROUP BY lo_quantity HAVING COUNT(*)+" + values[0] + "+" + values[1] + "+" + values[2] + "+" + values[3] + "<0 ORDER BY 1 LIMIT 100";
    };
    Rows expected;
    for (int64_t i = 0; i < 9; ++i) expected.push_back({integer(i), count(1)});
    size_t firstBuild = 0;
    for (unsigned round = 0; round < 2; ++round) {
        for (unsigned mask = 0; mask < 16; ++mask) {
            assert(fixture.query(sql(mask)) == expected);
            if (!round && !mask) firstBuild = fixture.engine.moduleCounters().compiler;
            if (round) fixture.noBuild();
            else if (mask > 8) fixture.noModuleBuild();
        }
    }
    assert(fixture.engine.moduleCounters().compiler == firstBuild + 8);
    assert(fixture.query(sql(15, true)).empty()); fixture.noBuild();
}

/** @brief A single prepared projection crosses count/scan capacity and local-selector/full-sort boundaries. */
static void runtimeSizes() {
    ParameterFixture fixture(9001);
    const auto sql = [](size_t selection, size_t limit) {
        return "SELECT lo_quantity,lo_revenue FROM lineorder WHERE lo_quantity<" + std::to_string(selection) + " ORDER BY 1 ASC,2 DESC LIMIT " + std::to_string(limit);
    };
    for (const auto& sizes : {std::pair<size_t, size_t>{1, 1}, {9001, 0}, {9001, 1}, {9001, 2048}, {9001, 2049}, {9001, 4096}, {9001, 4097}, {0, 100}, {9001, 999999}, {2, 999999}}) {
        Rows expected;
        for (size_t i = 0; i < std::min(sizes.first, sizes.second); ++i)
            expected.push_back({integer(i), integer(static_cast<int64_t>(i % 9) - 4)});
        assert(fixture.query(sql(sizes.first, sizes.second)) == expected);
        if (sizes != std::pair<size_t, size_t>{1, 1}) fixture.noBuild();
    }
    const auto desc = [](size_t limit) { return "SELECT lo_quantity FROM lineorder ORDER BY 1 DESC LIMIT " + std::to_string(limit); };
    for (size_t limit : {size_t{0}, size_t{1}, size_t{4096}, size_t{4097}, size_t{10000}}) {
        Rows expected;
        for (size_t i = 0; i < std::min<size_t>(limit, 9001); ++i) expected.push_back({integer(9000 - i)});
        assert(fixture.query(desc(limit)) == expected);
        if (limit) fixture.noBuild();
    }
    ParameterFixture guarded(9001);
    assert(guarded.query(sql(1, 1)) == Rows({{integer(0), integer(-4)}}));
    const auto resident = guarded.runtime->residentDeviceBytes() + guarded.runtime->residentPinnedBytes();
    guarded.context.config_.memory_guard_reserve_fraction = 0;
    guarded.context.config_.memory_guard_reserve_bytes = guarded.runtime->properties().totalGlobalMem - resident - 1024;
    bool rejected = false;
    try { guarded.query(sql(9001, 9001)); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find("Insufficient GPU memory") != std::string::npos; }
    assert(rejected); guarded.noBuild();
    guarded.context.config_ = CrystalConfig{};
    assert(guarded.query(sql(1, 1)) == Rows({{integer(0), integer(-4)}})); guarded.noBuild();
}

/** @brief Domains on either side of the fused-materialization cutoff retain runtime-sized dense output. */
static void denseGroups() {
    for (size_t groups : {size_t{256}, size_t{257}}) {
        ParameterFixture fixture(groups);
        const auto sql = [](const std::string& add, size_t threshold, size_t limit) {
            const auto expr = "lo_revenue+" + add;
            return "SELECT lo_quantity,SUM(" + expr + "),MIN(" + expr + "),MAX(" + expr + "),AVG(" + expr + ") FROM lineorder GROUP BY lo_quantity HAVING lo_quantity<" + std::to_string(threshold) + " ORDER BY 1 LIMIT " + std::to_string(limit);
        };
        for (size_t threshold : {size_t{1}, groups, size_t{0}, groups}) {
            Rows expected;
            for (size_t i = 0; i < threshold; ++i) {
                const auto value = static_cast<double>(i % 9) - 3.5;
                expected.push_back({integer(i), Cell{value}, Cell{value}, Cell{value}, Cell{value}});
            }
            assert(fixture.query(sql("0.5", threshold, 1000)) == expected);
            if (threshold != 1) fixture.noBuild();
        }
    }
    ParameterFixture fixture;
    const auto sql = [](const std::string& a, const std::string& b, const std::string& c) {
        const auto group = "CASE WHEN lo_quantity<3 THEN " + a + " ELSE CASE WHEN lo_quantity<6 THEN " + b + " ELSE " + c + " END END";
        return "SELECT " + group + ",COUNT(*) FROM lineorder GROUP BY " + group + " ORDER BY 1";
    };
    assert(fixture.query(sql("-9007199254740993", "9007199254740993", "0")) == Rows({{integer(-9007199254740993LL), count(3)}, {integer(0), count(3)}, {integer(9007199254740993LL), count(3)}}));
    assert(fixture.query(sql("7", "7", "7")) == Rows({{integer(7), count(9)}})); fixture.noBuild();
    assert(fixture.query(sql("NULL", "7", "7")) == Rows({{integer(7), count(6)}, {std::nullopt, count(3)}})); fixture.noBuild();
    assert(fixture.query(sql("NULL", "NULL", "NULL")) == Rows({{std::nullopt, count(9)}})); fixture.noBuild();
    bool rejected = false;
    try { fixture.query(sql("0.5", "1.5", "2.5")); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find("integer/NULL") != std::string::npos; }
    assert(rejected);
}

/** @brief Convert the independent CSV fixture into owned test columns, without adding production DDL/loading APIs. */
static void nullableDataset() {
    ParameterFixture fixture(12, true);
    std::ifstream csv(std::string(CRYSTAL_TEST_SOURCE_DIR) + "/test/nullable_data/nullable_numbers.csv");
    assert(csv);
    std::string line;
    std::getline(csv, line);
    std::vector<int> groups, values, flags;
    uint64_t value_valid = 0, flag_valid = 0;
    while (std::getline(csv, line)) {
        std::istringstream row(line);
        std::string id, group, value, flag;
        std::getline(row, id, ','); std::getline(row, group, ','); std::getline(row, value, ','); std::getline(row, flag, ',');
        const auto bit = 1ULL << groups.size();
        groups.push_back(std::stoi(group));
        values.push_back(value.empty() ? 0 : std::stoi(value));
        flags.push_back(flag.empty() ? 0 : std::stoi(flag));
        if (!value.empty()) value_valid |= bit;
        if (!flag.empty()) flag_valid |= bit;
    }
    assert(groups.size() == 12);
    fixture.context.base_columns_.at("d_lo_quantity").values->copyFromHost(groups);
    auto& value_column = fixture.context.base_columns_.at("d_lo_revenue");
    auto& flag_column = fixture.context.base_columns_.at("d_lo_suppkey");
    value_column.values->copyFromHost(values);
    value_column.validity->copyFromHost(std::vector<uint64_t>{value_valid});
    flag_column.values->copyFromHost(flags);
    flag_column.validity = std::make_unique<DynamicDeviceBuffer<uint64_t>>(fixture.runtime, 1);
    flag_column.validity->copyFromHost(std::vector<uint64_t>{flag_valid});
    assert(fixture.query("SELECT COUNT(*),COUNT(lo_revenue),SUM(lo_revenue),AVG(lo_revenue) FROM lineorder ORDER BY 1") == Rows({{count(12), count(7), integer(145), Cell{145.0 / 7.0}}}));
    assert(fixture.query("SELECT lo_quantity,COUNT(*),COUNT(lo_revenue),SUM(lo_revenue) FROM lineorder GROUP BY lo_quantity ORDER BY 1") == Rows({{integer(1), count(3), count(2), integer(40)}, {integer(2), count(3), count(2), integer(20)}, {integer(3), count(3), count(2), integer(45)}, {integer(4), count(3), count(1), integer(40)}}));
    assert(fixture.query("SELECT COUNT(*) FROM lineorder WHERE lo_revenue IS NULL ORDER BY 1") == Rows({{count(5)}}));
    assert(fixture.query("SELECT COUNT(*) FROM lineorder WHERE lo_suppkey=1 ORDER BY 1") == Rows({{count(5)}}));
    assert(fixture.query("SELECT lo_quantity,SUM(CASE WHEN lo_revenue IS NULL THEN 1 ELSE 0 END) FROM lineorder GROUP BY lo_quantity ORDER BY 1") == Rows({{integer(1), integer(1)}, {integer(2), integer(1)}, {integer(3), integer(1)}, {integer(4), integer(2)}}));
}

int main(int argc, char** argv) try {
    if (argc > 1) {
        const auto root = std::filesystem::absolute(argv[1]);
        std::filesystem::create_directories(root);
        const auto config = root / "fixture.conf";
        std::ofstream file(config);
        file << "jit_keep_artifacts=on\njit_artifact_dir=" << root.string() << "\n";
        file.close();
        setenv("CRYSTAL_CONFIG", config.c_str(), 1);
        setenv("ACPP_APPDB_DIR", (root / "device-cache").c_str(), 1);
    }
    if (argc > 2 && std::string(argv[2]) == "grouped") {
        groupedAggregates();
        return 0;
    }
    numericAggregates();
    groupedAggregates();
    joinedAggregates();
    nullableInputs();
    boundedTypes(false);
    boundedTypes(true);
    boundedTypes(true, true);
    outputLayouts();
    boundedCountTypes();
    runtimeSizes();
    denseGroups();
    nullableDataset();
    std::cout << "Runtime numeric parameter tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
