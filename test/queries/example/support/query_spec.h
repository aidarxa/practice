#pragma once
/** @file @brief Handwritten SSB recipes, independent of SQL AST, optimizer and codegen. */
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "core/text.h"

namespace ssb {
/** @brief Dimension slots are Date/Supplier/Customer/Part, matching fact foreign keys. */
enum Slot : unsigned { Date, Supplier, Customer, Part };
/** @brief A fixed manual predicate; no SQL expression interpreter is involved. */
struct Filter {
    int low = 0;
    int high = 0;
    bool either = false;
    bool text = false;
    std::array<std::uint8_t, 32> low_text{}, high_text{};
    unsigned low_bytes = 0, high_bytes = 0;
    constexpr Filter() = default;
    constexpr Filter(int first, int last, bool alternatives = false) : low(first), high(last), either(alternatives) {}
    constexpr Filter(const char* first, const char* last, bool alternatives = false) : either(alternatives), text(true) {
        while (first[low_bytes]) {
            if (low_bytes == low_text.size()) throw std::invalid_argument("Manual predicate is too long");
            low_text[low_bytes] = first[low_bytes]; ++low_bytes;
        }
        while (last[high_bytes]) {
            if (high_bytes == high_text.size()) throw std::invalid_argument("Manual predicate is too long");
            high_text[high_bytes] = last[high_bytes]; ++high_bytes;
        }
    }
    bool matches(int value) const { return either ? value == low || value == high : value >= low && value <= high; }
    bool matches(db::TextView value) const {
        const auto first = db::compareText(value, {low_text.data(), low_bytes});
        const auto last = db::compareText(value, {high_text.data(), high_bytes});
        return either ? first == 0 || last == 0 : first >= 0 && last <= 0;
    }
};
/** @brief Empty key means this query does not build/probe the dimension. */
struct DimensionSpec {
    const char* key = "";
    const char* value = "";
    const char* filter_column = "";
    Filter filter;
};
/** @brief Group by a native dimension value; there is no categorical numeric domain. */
struct GroupSpec { unsigned slot = 0; };
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
std::size_t groupCapacity(const QuerySpec& spec, const std::array<std::uint64_t, 3>& cardinalities);
/** @brief Validate and describe a unique perfect-hash key domain, allowing key zero/negative keys. */
struct KeyDomain { int minimum = 0; std::size_t extent = 1; };
KeyDomain keyDomain(const std::vector<int>& keys);
/** @brief Reject aggregate bounds that could overflow signed Int64 before enqueueing. */
void validateMeasureBound(uint64_t rows, uint64_t left_magnitude, uint64_t right_magnitude, bool product);
}
