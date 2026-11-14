/** @file
 * @brief Deterministic tiny-data tests of the actual engine/plan/legacy GPU boundary.
 * @details Input arrays and expected results are independent of generated device
 * helpers. No SQL DDL or new GPU algorithm is introduced. This executable needs
 * the same compiler/include environment as db_cli and must run alone because the
 * not-yet-migrated compiler still has a process-global PCH path.
 */
#include "core/execution.h"
#include "core/legacy_plan_runtime.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>

using namespace db;
using Cell = std::optional<double>;
using Rows = std::vector<std::vector<Cell>>;

/** @brief Own input allocations with the same RAII wrapper as runtime scratch. */
class Fixture {
public:
    std::shared_ptr<DeviceRuntime> runtime = std::make_shared<DeviceRuntime>();
    ExecutionContext context;
    std::shared_ptr<Catalog> catalog = std::make_shared<Catalog>();
    QueryEngine engine{catalog, std::make_unique<FileBasedQueryCache>(),
                       std::make_unique<AdaptiveCppCompiler>(), std::make_unique<DynamicLibraryExecutor>()};

    Fixture() {
        context.runtime_ = runtime;
        context.config_.reuse_scratch_buffers = true;
        context.result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime, 0);
        context.result_validity_buffer_ = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, 0);
        TableMetadata supplier("SUPPLIER", {"s_suppkey", "s_region", "s_nation"}, 8, false);
        supplier.setColumnStats("s_suppkey", {1, 4, 4}); // explicitly nonunique: choose MHT
        supplier.setColumnNullable("s_suppkey");
        supplier.setColumnStats("s_region", {0, 4, 5});
        supplier.setColumnNullable("s_nation");
        catalog->pushTableMetadata(supplier);
        TableMetadata fact("LINEORDER", {"lo_suppkey", "lo_orderkey", "lo_revenue"}, 6, true);
        fact.setColumnNullable("lo_suppkey");
        catalog->pushTableMetadata(fact);
        input("s_suppkey", {1,1,2,2,3,3,4,4}, 0x7f);
        input("s_region", {0,1,1,2,2,3,3,4});
        input("s_nation", {11,12,21,22,31,32,41,42}, 0x7f);
        input("lo_suppkey", {1,2,3,4,5,0}, 0x1f);
        input("lo_orderkey", {1,2,2,4,4,5});
        input("lo_revenue", {10,20,30,40,50,60});
    }
    void input(const std::string& name, const std::vector<int>& values, std::optional<uint64_t> validity = {}) {
        LoadedDeviceColumn column;
        column.values = std::make_unique<DynamicDeviceBuffer<int>>(runtime, values.size());
        column.values->copyFromHost(values);
        context.loaded_device_bytes_ = checkedSizeAdd(context.loaded_device_bytes_, checkedSizeMultiply(values.size(), sizeof(int)));
        if (validity) {
            auto bitmap = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, 1);
            bitmap->copyFromHost(std::vector<uint64_t>{*validity});
            context.loaded_device_bytes_ = checkedSizeAdd(context.loaded_device_bytes_, sizeof(uint64_t));
            column.validity = std::move(bitmap);
        }
        context.base_columns_.emplace("d_" + name, std::move(column));
    }
    Rows query(const std::string& sql) {
        const auto* nonce = std::getenv("DB_STAGE2_FIXTURE_NONCE");
        std::string spacing = " ";
        for (const char* digit = nonce ? nonce : "0"; *digit; ++digit) {
            const char text[] = {*digit, 0};
            const auto nibble = std::strtoul(text, nullptr, 16);
            for (unsigned bit = 0; bit < 4; ++bit) spacing += (nibble & (1U << bit)) ? '\t' : ' ';
        }
        std::cout << "fixture SQL: " << sql << std::endl;
        engine.executeQuery(sql.substr(0, 6) + spacing + sql.substr(6), &context);
        assert(context.result_is_columnar_);
        Rows result(context.result_row_count_, std::vector<Cell>(context.result_column_count_));
        for (size_t column = 0; column < context.result_column_count_; ++column) {
            auto& storage = context.result_column_storage_[column];
            std::vector<uint64_t> valid;
            storage.validity->copyToHost(valid, (context.result_row_count_ + 63) / 64);
            std::vector<double> f64;
            std::vector<int64_t> i64;
            std::vector<uint64_t> u64;
            if (storage.type == LogicalType::Float64) storage.f64->copyToHost(f64, context.result_row_count_);
            else if (storage.type == LogicalType::Int64) storage.i64->copyToHost(i64, context.result_row_count_);
            else storage.u64->copyToHost(u64, context.result_row_count_);
            for (size_t row = 0; row < result.size(); ++row) {
                if (context.result_columns_[column].nullable && !(valid[row / 64] & (1ULL << (row % 64)))) continue;
                if (!f64.empty()) result[row][column] = f64[row];
                else if (!i64.empty()) result[row][column] = static_cast<double>(i64[row]);
                else result[row][column] = static_cast<double>(u64[row]);
            }
        }
        return result;
    }
};

static void expect(const Rows& actual, const Rows& expected) {
    assert(actual.size() == expected.size());
    for (size_t row = 0; row < actual.size(); ++row) {
        assert(actual[row].size() == expected[row].size());
        for (size_t column = 0; column < actual[row].size(); ++column) {
            assert(actual[row][column].has_value() == expected[row][column].has_value());
            if (actual[row][column]) assert(std::abs(*actual[row][column] - *expected[row][column]) < 1e-9);
        }
    }
}

/** @brief Nested-loop oracle preserves MHT multiplicity and OR duplicate suppression. */
static Rows joinOracle(bool or_join) {
    const int keys[] = {1,2,3,4,5,0}, orders[] = {1,2,2,4,4,5};
    const int build[] = {1,1,2,2,3,3,4,4}, payload[] = {11,12,21,22,31,32,41,42};
    Rows rows;
    for (size_t fact = 0; fact < 6; ++fact) {
        for (size_t dimension = 0; dimension < 7; ++dimension) {
            const bool left = fact < 5 && keys[fact] == build[dimension];
            const bool right = or_join && orders[fact] == build[dimension];
            if (left || right) rows.push_back({double(orders[fact]), double(payload[dimension])});
        }
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

static void incompatibleModuleHasNoEffects(Fixture& fixture) {
    ExecutionPlan plan;
    plan.buffers.push_back({"must_not_allocate", DB_TYPE_UINT64, BufferRole::Scratch, {}});
    plan.kernels.push_back({"bad_schema", {{"value", {DB_ARG_SCALAR, DB_TYPE_UINT64, DB_ACCESS_READ, 0}, SizeExpr(1)}}});
    plan.steps = {EnsureBuffer{"must_not_allocate", 1}, LaunchKernel{0, 1, 1, 0}};
    KernelModuleDescriptorV1 descriptor{1, sizeof(KernelModuleDescriptorV1), DB_BACKEND_LEGACY_SYCL, 0, nullptr, 0};
    std::vector<LegacyKernelEntry> entries{{&descriptor, [](const KernelInvocationV1*, KernelErrorV1*) -> int32_t { std::abort(); }}};
    bool rejected = false;
    try { executeLegacyPlan(plan, {}, entries, fixture.context); }
    catch (const std::runtime_error& error) { rejected = std::string(error.what()).find("argument count") != std::string::npos; }
    assert(rejected && fixture.context.plan_scratch_buffers_.count("must_not_allocate") == 0);
}

int main() try {
    Fixture fixture;
    incompatibleModuleHasNoEffects(fixture);
    const std::string join = "SELECT lo_orderkey, s_nation FROM lineorder, supplier WHERE lo_suppkey=s_suppkey ORDER BY 1,2";
    const std::string or_join = "SELECT lo_orderkey, s_nation FROM lineorder, supplier WHERE (lo_suppkey=s_suppkey OR lo_orderkey=s_suppkey) ORDER BY 1,2";
    expect(fixture.query(join), joinOracle(false));
    expect(fixture.query(or_join), joinOracle(true));
    expect(fixture.query(join), joinOracle(false));
    const std::string aggregates = "SELECT COUNT(*),COUNT(s_nation),SUM(s_nation),AVG(s_nation),MIN(s_nation),MAX(s_nation) FROM supplier";
    expect(fixture.query(aggregates + " ORDER BY 1"), {{8,7,170,170.0/7,11,41}});
    expect(fixture.query(aggregates + " WHERE s_suppkey IS NULL ORDER BY 1"), {{1,0,{}, {}, {}, {}}});
    expect(fixture.query("SELECT s_suppkey,s_nation FROM supplier WHERE s_nation IS NOT NULL ORDER BY 2 DESC LIMIT 3"), {{4,41},{3,32},{3,31}});
    expect(fixture.query("SELECT s_nation FROM supplier WHERE s_nation > 100 ORDER BY 1 LIMIT 0"), {});
    fixture.context.config_.reuse_scratch_buffers = false;
    expect(fixture.query(or_join), joinOracle(true));
    fixture.context.config_.reuse_scratch_buffers = true;
    const auto reserve = fixture.context.config_.memory_guard_reserve_bytes;
    fixture.context.config_.memory_guard_reserve_bytes = fixture.runtime->properties().totalGlobalMem;
    bool budget_rejected = false;
    try { fixture.query(join); } catch (const std::exception& error) {
        budget_rejected = std::string(error.what()).find("Insufficient GPU memory") != std::string::npos;
    }
    assert(budget_rejected);
    fixture.context.config_.memory_guard_reserve_bytes = std::numeric_limits<size_t>::max();
    bool rejected = false;
    try { fixture.query(join); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
    fixture.context.config_.memory_guard_reserve_bytes = reserve;
    const auto fraction = fixture.context.config_.memory_guard_reserve_fraction;
    fixture.context.config_.memory_guard_reserve_fraction = std::numeric_limits<double>::quiet_NaN();
    bool bad_fraction = false;
    try { fixture.query(join); } catch (const std::exception& error) {
        bad_fraction = std::string(error.what()).find("invalid memory guard") != std::string::npos;
    }
    assert(bad_fraction);
    fixture.context.config_.memory_guard_reserve_fraction = fraction;
    fixture.context.config_.memory_guard_enabled = false;
    expect(fixture.query(join), joinOracle(false));
    fixture.context.config_.memory_guard_enabled = true;
    expect(fixture.query(join), joinOracle(false));
    std::cout << "GPU plan fixtures passed: MHT, OR multiplicity, nullable aggregates, projection, sorting, reuse, memory/ABI rejection\n";
} catch (const std::exception& error) {
    std::cerr << "GPU fixture failure: " << error.what() << '\n';
    return 1;
}
