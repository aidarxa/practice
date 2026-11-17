/** @file @brief Independent boundary checks of manual grouping, predicates and signed size bounds. */
#include "support/query_spec.h"
#include <cassert>
#include <climits>
#include <iostream>
#include <limits>

template<class F> static void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
#ifdef SSB_COMBINED_TEST
int manualSpecChecks() {
#else
int main() {
#endif
    for (const auto id : {11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43}) {
        const auto query = ssb::querySpec(id);
        assert(query.id == static_cast<unsigned>(id));
        assert(ssb::groupCapacity(query, {7, 25, 25}) > 0);
    }
    rejects([] { ssb::querySpec(0); });
    assert(ssb::groupCapacity(ssb::querySpec(33), {7, 250, 250}) == 2 * 7 * 250 * 250);
    assert(ssb::groupCapacity(ssb::querySpec(43), {7, 250, 1000}) == 2 * 7 * 250 * 1000);
    auto spec = ssb::querySpec(33);
    spec.group_count = 4; rejects([&] { ssb::groupCapacity(spec, {1, 1, 1}); });
    spec = ssb::querySpec(33); spec.groups[0].slot = 4; rejects([&] { ssb::groupCapacity(spec, {1, 1, 1}); });
    spec = ssb::querySpec(33);
    rejects([&] { ssb::groupCapacity(spec, {UINT64_MAX, UINT64_MAX, UINT64_MAX}); });
    rejects([&] { ssb::groupCapacity(spec, {std::uint64_t{1} << 63, 2, 1}); });
    assert(ssb::groupCapacity(spec, {0, 0, 0}) == 2);
    const ssb::Filter text{"ASIA", "EUROPE", true};
    const auto view = [](const char* value) { return db::TextView{reinterpret_cast<const std::uint8_t*>(value), std::char_traits<char>::length(value)}; };
    assert(text.matches(view("ASIA")) && text.matches(view("EUROPE")) && !text.matches(view("ASIA ")));
    const auto range = ssb::Filter{2, 4};
    assert(!range.matches(1) && range.matches(2) && range.matches(3) && range.matches(4) && !range.matches(5));
    const auto either = ssb::Filter{2, 4, true};
    assert(!either.matches(1) && either.matches(2) && !either.matches(3) && either.matches(4) && !either.matches(5));
    assert(ssb::querySpec(11).scalar.date.matches(19931231));
    assert(!ssb::querySpec(11).scalar.date.matches(19940101));
    const auto empty = ssb::keyDomain({});
    assert(empty.minimum == 0 && empty.extent == 1);
    const auto domain = ssb::keyDomain({2, -2, 0});
    assert(domain.minimum == -2 && domain.extent == 5);
    const auto wide = ssb::keyDomain({INT_MIN, INT_MAX});
    assert(wide.minimum == INT_MIN && wide.extent == uint64_t{1} << 32);
    rejects([] { ssb::keyDomain({0, 0}); });
    ssb::validateMeasureBound(1, INT64_MAX, 0, false);
    ssb::validateMeasureBound(1, INT64_MAX, 1, true);
    ssb::validateMeasureBound(0, 7, 3, true);
    rejects([] { ssb::validateMeasureBound(1, INT64_MAX, 1, false); });
    rejects([] { ssb::validateMeasureBound(2, INT64_MAX, 1, true); });
    rejects([] { ssb::validateMeasureBound(1, UINT64_MAX, 2, true); });
    rejects([] { ssb::validateMeasureBound(1, UINT64_MAX, 1, false); });
    std::cout << "Manual SSB specification checks passed\n";
    return 0;
}
