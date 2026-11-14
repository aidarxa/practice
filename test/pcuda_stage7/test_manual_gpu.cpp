/** @file @brief Independent tiny CPU oracles for manual PCUDA algorithms, including reuse and zero keys. */
#include "support/runner.h"
#include <cassert>
#include <iostream>

template<class F> static void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
static void scalar(const ssb::Runtime& runtime) {
    ssb::Dataset data(runtime);
    std::vector<int> date(1031), discount(1031), quantity(1031), price(1031);
    int64_t expected = 0;
    for (std::size_t row = 0; row < date.size(); ++row) {
        date[row] = row % 2 ? 19930101 : 19940101;
        discount[row] = row % 5;
        quantity[row] = row % 30;
        price[row] = static_cast<int>(row % 19) - 9;
        if (row % 2 && discount[row] >= 1 && discount[row] <= 3 && quantity[row] < 25)
            expected += static_cast<int64_t>(price[row]) * discount[row];
    }
    data.add("lo_orderdate", date); data.add("lo_discount", discount);
    data.add("lo_quantity", quantity); data.add("lo_extendedprice", price);
    rejects([&] { data.add("lo_orderdate", date); });
    ssb::ManualQuery query(data, ssb::querySpec(11));
    for (unsigned repeat = 0; repeat < 4; ++repeat) assert(query.execute() == ssb::Rows{{expected}});
    auto spec = ssb::querySpec(11);
    spec.scalar.date = {1, 2};
    ssb::ManualQuery empty(data, spec);
    assert(empty.execute() == ssb::Rows{{0}});
}
static ssb::QuerySpec groupedSpec() {
    ssb::QuerySpec spec{31, {}};
    spec.dimensions[ssb::Supplier] = {"s_suppkey", "s_nation", "s_region", {1, 1}};
    spec.groups[0] = {ssb::Supplier, -1, 4};
    spec.group_count = 1;
    return spec;
}
static void grouped(const ssb::Runtime& runtime) {
    ssb::Dataset data(runtime);
    data.add("s_suppkey", {-1, 0, 1, 2}); data.add("s_nation", {-1, 0, 1, 2}); data.add("s_region", {1, 1, 1, 1});
    data.add("lo_suppkey", {0, -1, 1, 0, -1, 7, 2}); data.add("lo_revenue", {3, 4, -2, -3, -7, 99, 0});
    data.add("lo_supplycost", {1, 1, 1, 1, 1, 1, 0});
    auto spec = groupedSpec();
    ssb::ManualQuery query(data, spec);
    const ssb::Rows expected{{-1, -3}, {0, 0}, {1, -2}, {2, 0}};
    for (unsigned repeat = 0; repeat < 4; ++repeat) assert(query.execute() == expected);
    const auto resident = runtime->residentDeviceBytes();
    query.setFilter(ssb::Supplier, {2, 2});
    assert(query.execute().empty());
    query.setFilter(ssb::Supplier, {1, 1});
    assert(query.execute() == expected);
    assert(runtime->residentDeviceBytes() == resident);
    rejects([&] { query.setFilter(4, {}); });
    rejects([&] { query.setFilter(ssb::Date, {}); });
    spec.profit = true;
    ssb::ManualQuery profit(data, spec);
    assert(profit.execute() == ssb::Rows({{-1, -5}, {0, -2}, {1, -3}, {2, 0}}));
    spec.dimensions[ssb::Supplier].filter = {2, 2};
    ssb::ManualQuery empty(data, spec);
    for (unsigned repeat = 0; repeat < 2; ++repeat) assert(empty.execute().empty());
    assert(query.execute() == expected); // another query must not corrupt retained workspace.
    spec = groupedSpec(); spec.groups[0].minimum = 0;
    rejects([&] { ssb::ManualQuery invalid(data, spec); });
}
static void emptyAndErrors(const ssb::Runtime& runtime) {
    ssb::Dataset empty(runtime);
    for (const auto* name : {"s_suppkey", "s_nation", "s_region", "lo_suppkey", "lo_revenue"}) empty.add(name, {});
    ssb::ManualQuery no_rows(empty, groupedSpec());
    assert(no_rows.execute().empty());
    ssb::Dataset duplicate(runtime);
    duplicate.add("s_suppkey", {0, 0}); duplicate.add("s_nation", {0, 1}); duplicate.add("s_region", {1, 1});
    duplicate.add("lo_suppkey", {0}); duplicate.add("lo_revenue", {1});
    rejects([&] { ssb::ManualQuery invalid(duplicate, groupedSpec()); });
    ssb::Dataset mismatched(runtime);
    mismatched.add("lo_orderdate", {19930101}); mismatched.add("lo_discount", {});
    mismatched.add("lo_quantity", {1}); mismatched.add("lo_extendedprice", {1});
    rejects([&] { ssb::ManualQuery invalid(mismatched, ssb::querySpec(11)); });
    ssb::Dataset no_dimension(runtime);
    for (const auto* name : {"s_suppkey", "s_nation", "s_region"}) no_dimension.add(name, {});
    no_dimension.add("lo_suppkey", {0}); no_dimension.add("lo_revenue", {5});
    ssb::ManualQuery absent(no_dimension, groupedSpec());
    assert(absent.execute().empty());
}
int main() {
    const auto runtime = std::make_shared<db::DeviceRuntime>();
    scalar(runtime);
    grouped(runtime);
    emptyAndErrors(runtime);
    runtime->wait();
    assert(runtime->residentDeviceBytes() == 0 && runtime->residentPinnedBytes() == 0);
    std::cout << "Manual PCUDA tiny-data checks passed\n";
}
