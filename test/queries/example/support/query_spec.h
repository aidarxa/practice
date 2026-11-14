#pragma once
/** @file @brief Handwritten SSB recipes, independent of SQL AST, optimizer and codegen. */
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ssb {
/** @brief Dimension slots are Date/Supplier/Customer/Part, matching fact foreign keys. */
enum Slot : unsigned { Date, Supplier, Customer, Part };
/** @brief A fixed manual predicate; no SQL expression interpreter is involved. */
struct Filter {
    int low = 0;
    int high = 0;
    bool either = false;
    bool matches(int value) const { return either ? value == low || value == high : value >= low && value <= high; }
};
/** @brief Empty key means this query does not build/probe the dimension. */
struct DimensionSpec {
    const char* key = "";
    const char* value = "";
    const char* filter_column = "";
    Filter filter;
};
/** @brief Mixed-radix grouping is collision-free, with an explicit checked domain. */
struct GroupSpec { unsigned slot = 0; int minimum = 0; unsigned extent = 1; };
struct ScalarSpec { Filter date; Filter discount; Filter quantity; };
/** @brief One of the thirteen existing handwritten queries, not a generated physical plan. */
struct QuerySpec {
    unsigned id;
    ScalarSpec scalar;
    std::array<DimensionSpec, 4> dimensions{};
    std::array<GroupSpec, 3> groups{};
    unsigned group_count = 0;
    bool profit = false;
};
/** @brief Return the fixed manual recipe; Q1.1 deliberately retains the reference's 1993-only window. */
QuerySpec querySpec(unsigned id);
/** @brief Check group products before allocation, with no implicit modulo collisions. */
std::size_t groupCapacity(const QuerySpec& spec);
/** @brief Validate and describe a unique perfect-hash key domain, allowing key zero/negative keys. */
struct KeyDomain { int minimum = 0; std::size_t extent = 1; };
KeyDomain keyDomain(const std::vector<int>& keys);
/** @brief Reject aggregate bounds that could overflow signed Int64 before enqueueing. */
void validateMeasureBound(uint64_t rows, uint64_t left_magnitude, uint64_t right_magnitude, bool product);
}
