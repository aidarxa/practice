#pragma once
/** @file
 * @brief Bit-preserving aggregate states over the unchanged UInt64 storage ABI.
 * @details Numeric types are existing Int64/Float64, not new SQL types. Validity and
 * group presence are independent of neutral values. Integer SQL overflow remains
 * outside the representable-input contract; no integer is routed through double.
 */
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace db {
enum class AggregateOp { Sum, Minimum, Maximum };

/** @brief Reinterpret one eight-byte state without an aliasing-violating pointer cast. */
template<class T> inline T aggregateValue(unsigned long long bits) {
    static_assert(sizeof(T) == sizeof(bits), "eight-byte aggregate state required");
    T result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
template<class T> inline unsigned long long aggregateBits(T value) {
    static_assert(sizeof(T) == sizeof(unsigned long long), "eight-byte aggregate state required");
    unsigned long long result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

/** @brief Typed identities must work for negative values, fractions and empty reductions. */
template<class T, AggregateOp op> inline T aggregateIdentity() {
    if constexpr (op == AggregateOp::Sum) return T{0};
    if constexpr (op == AggregateOp::Minimum) {
        if constexpr (std::is_floating_point_v<T>) return std::numeric_limits<T>::infinity();
        else return std::numeric_limits<T>::max();
    }
    if constexpr (op == AggregateOp::Maximum) {
        if constexpr (std::is_floating_point_v<T>) return -std::numeric_limits<T>::infinity();
        else return std::numeric_limits<T>::lowest();
    }
}
/** @brief The same typed merge serves register accumulation and portable block reduction. */
template<class T, AggregateOp op> struct AggregateMerge {
    T operator()(T a, T b) const {
        if constexpr (op == AggregateOp::Minimum) return a < b ? a : b;
        if constexpr (op == AggregateOp::Maximum) return a > b ? a : b;
        if constexpr (op == AggregateOp::Sum) return a + b;
    }
};
} // namespace db
