#pragma once
/** @file
 * @brief Immutable, non-owning Text values shared by host tests and PCUDA kernels.
 * @details Bytes are validated UTF-8 at the input boundary. Ordering is unsigned
 * bytewise, case-sensitive and non-normalizing. Validity is deliberately external.
 * A view never allocates and its owner must outlive every reader, including fetch.
 */
#include <cstdint>
#include <cstddef>
#include "core/module_abi_v2.h"

namespace db {
/** @brief Sixteen-byte borrowed byte range; a zero length is not SQL NULL. */
using TextView = DbTextViewV2;
static_assert(sizeof(TextView) == 16 && alignof(TextView) == 8, "64-bit Text view layout");
/** @brief Compare complete values, never reading padding or requiring a terminator. */
inline int compareText(TextView left, TextView right) {
    const auto common = left.bytes < right.bytes ? left.bytes : right.bytes;
    for (std::uint64_t i = 0; i < common; ++i) {
        if (left.data[i] < right.data[i]) return -1;
        if (left.data[i] > right.data[i]) return 1;
    }
    return left.bytes < right.bytes ? -1 : left.bytes > right.bytes ? 1 : 0;
}
/** @brief Equality rejects different lengths before accessing character bytes. */
inline bool equalText(TextView left, TextView right) {
    return left.bytes == right.bytes && compareText(left, right) == 0;
}
inline bool safe_eq(TextView a, TextView b) { return equalText(a, b); }
inline bool safe_neq(TextView a, TextView b) { return !equalText(a, b); }
inline bool safe_lt(TextView a, TextView b) { return compareText(a, b) < 0; }
inline bool safe_lte(TextView a, TextView b) { return compareText(a, b) <= 0; }
inline bool safe_gt(TextView a, TextView b) { return compareText(a, b) > 0; }
inline bool safe_gte(TextView a, TextView b) { return compareText(a, b) >= 0; }
/** @brief Content hash is only a candidate filter; callers must verify equality. */
inline std::uint64_t hashText(TextView value) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::uint64_t i = 0; i < value.bytes; ++i) hash = (hash ^ value.data[i]) * 1099511628211ULL;
    return hash;
}
/** @brief Decode a checked compact column offset without specializing on values. */
inline TextView textAt(const std::uint8_t* bytes, const std::uint64_t* offsets, std::uint64_t row) {
    const auto length = offsets[row + 1] - offsets[row];
    return {length ? bytes + offsets[row] : nullptr, length};
}
inline TextView textAt(const std::uint8_t* bytes, const std::int32_t* offsets, std::uint64_t row) {
    const auto length = static_cast<std::uint64_t>(offsets[row + 1] - offsets[row]);
    return {length ? bytes + offsets[row] : nullptr, length};
}
} // namespace db
