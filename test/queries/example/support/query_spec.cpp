/** @file @brief Manual SSB constants and pure host-side safety checks. */
#include "query_spec.h"
#include "core/size_expression.h"
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace ssb {
namespace {
constexpr DimensionSpec dateAll{"d_datekey", "d_year", "", {}};
constexpr DimensionSpec dateRange{"d_datekey", "d_year", "d_year", {1992, 1997}};
constexpr DimensionSpec dateLate{"d_datekey", "d_year", "d_year", {1997, 1998, true}};
constexpr DimensionSpec dateMonth{"d_datekey", "d_year", "d_yearmonthnum", {199712, 199712}};
constexpr GroupSpec year{Date}, brand{Part}, nationS{Supplier}, nationC{Customer}, cityS{Supplier}, cityC{Customer};
constexpr DimensionSpec supplier(const char* region) {
    return {"s_suppkey", "s_nation", "s_region", {region, region}};
}
constexpr DimensionSpec customer(const char* region) {
    return {"c_custkey", "c_nation", "c_region", {region, region}};
}
constexpr DimensionSpec supplierCity{"s_suppkey", "s_city", "s_city", {"UNITED KI1", "UNITED KI5", true}};
constexpr DimensionSpec customerCity{"c_custkey", "c_city", "c_city", {"UNITED KI1", "UNITED KI5", true}};
constexpr DimensionSpec manufacturer{"p_partkey", "p_category", "p_mfgr", {"MFGR#1", "MFGR#2", true}};
constexpr std::array<QuerySpec, 13> queries{{
    {11, {{19930101, 19931231}, {1, 3}, {0, 24}}},
    {12, {{19940101, 19940131}, {4, 6}, {26, 35}}},
    {13, {{19940204, 19940210}, {5, 7}, {26, 35}}},
    {21, {}, {dateAll, supplier("AMERICA"), {}, {"p_partkey", "p_brand1", "p_category", {"MFGR#12", "MFGR#12"}}}, {year, brand}, 2},
    {22, {}, {dateAll, supplier("ASIA"), {}, {"p_partkey", "p_brand1", "p_brand1", {"MFGR#2221", "MFGR#2228"}}}, {year, brand}, 2},
    {23, {}, {dateAll, supplier("EUROPE"), {}, {"p_partkey", "p_brand1", "p_brand1", {"MFGR#2221", "MFGR#2221"}}}, {year, brand}, 2},
    {31, {}, {dateRange, supplier("ASIA"), customer("ASIA"), {}}, {year, nationC, nationS}, 3},
    {32, {}, {dateRange, {"s_suppkey", "s_city", "s_nation", {"UNITED STATES", "UNITED STATES"}}, {"c_custkey", "c_city", "c_nation", {"UNITED STATES", "UNITED STATES"}}, {}}, {year, cityC, cityS}, 3},
    {33, {}, {dateRange, supplierCity, customerCity, {}}, {year, cityC, cityS}, 3},
    {34, {}, {dateMonth, supplierCity, customerCity, {}}, {year, cityC, cityS}, 3},
    {41, {}, {dateAll, supplier("AMERICA"), customer("AMERICA"), manufacturer}, {year, nationC}, 2, true},
    {42, {}, {dateLate, supplier("AMERICA"), customer("AMERICA"), manufacturer}, {year, nationS, {Part}}, 3, true},
    {43, {}, {dateLate, {"s_suppkey", "s_city", "s_nation", {"UNITED STATES", "UNITED STATES"}}, customer("AMERICA"), {"p_partkey", "p_brand1", "p_category", {"MFGR#14", "MFGR#14"}}}, {year, cityS, brand}, 3, true}
}};
}

QuerySpec querySpec(unsigned id) {
    for (const auto& query : queries) if (query.id == id) return query;
    throw std::invalid_argument("Unknown manual SSB query");
}
std::size_t groupCapacity(const QuerySpec& spec, const std::array<std::uint64_t, 3>& cardinalities) {
    if (spec.group_count > spec.groups.size()) throw std::invalid_argument("Invalid manual group count");
    uint64_t count = 1;
    for (unsigned i = 0; i < spec.group_count; ++i) {
        const auto& group = spec.groups[i];
        if (group.slot >= spec.dimensions.size()) throw std::invalid_argument("Invalid manual group slot");
        count = db::checkedSizeMultiply(count, std::max<std::uint64_t>(cardinalities[i], 1));
    }
    if (spec.group_count) count = db::checkedSizeMultiply(count, 2);
    if (count > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Manual group extent overflow");
    return static_cast<std::size_t>(count);
}
KeyDomain keyDomain(const std::vector<int>& keys) {
    if (keys.empty()) return {};
    std::unordered_set<int> unique;
    for (const auto key : keys)
        if (!unique.insert(key).second) throw std::invalid_argument("Manual PHT requires unique dimension keys");
    const auto bounds = std::minmax_element(keys.begin(), keys.end());
    const auto extent = static_cast<uint64_t>(static_cast<int64_t>(*bounds.second) - *bounds.first) + 1;
    if (extent > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Manual PHT domain overflow");
    return {*bounds.first, static_cast<std::size_t>(extent)};
}
void validateMeasureBound(uint64_t rows, uint64_t left, uint64_t right, bool product) {
    const auto per_row = product ? db::checkedSizeMultiply(left, right) : db::checkedSizeAdd(left, right);
    if (db::checkedSizeMultiply(rows, per_row) > static_cast<uint64_t>(INT64_MAX))
        throw std::overflow_error("Manual SSB signed aggregate bound exceeds Int64");
}
}
