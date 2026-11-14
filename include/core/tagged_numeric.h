#pragma once
/** @file @brief Exact Int64/Float64 fallback payloads; validity remains an independent ABI bit. */
#include "core/module_abi.h"
#include <type_traits>
#include <limits>

namespace db {
/** @brief Device-local number. Unsigned is an internal aggregate operand, not a new SQL parameter type. */
struct TaggedNumeric {
    KernelScalarV1 value{};
    uint32_t type = DB_TYPE_INT64;
    TaggedNumeric() = default;
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    TaggedNumeric(T input) {
        if constexpr (std::is_unsigned_v<T>) { type = DB_TYPE_UINT64; value.u64 = input; }
        else value.i64 = input;
    }
    template<class T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    TaggedNumeric(T input) : type(DB_TYPE_FLOAT64) { value.f64 = input; }
    template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    explicit operator T() const {
        if (type == DB_TYPE_FLOAT64) return static_cast<T>(value.f64);
        if (type == DB_TYPE_UINT64) return static_cast<T>(value.u64);
        return static_cast<T>(value.i64);
    }
};
/** @brief Copy a validated ABI value into a closure-safe payload; invalid payloads are neutral. */
template<class Argument> inline TaggedNumeric taggedParameter(const Argument& argument) {
    TaggedNumeric result;
    if (argument.valid) {
        result.type = argument.schema.type;
        if (result.type==DB_TYPE_FLOAT64) result.value.f64=argument.value.f64;
        else if (result.type==DB_TYPE_UINT64) result.value.u64=argument.value.u64;
        else result.value.i64=argument.value.i64;
    }
    return result;
}
enum class NumericOp { Add, Subtract, Multiply, Divide, Equal, Less, LessEqual, GreaterEqual };
template<NumericOp op, class T> inline auto numericOperation(T a, T b) {
    if constexpr (op == NumericOp::Add) return a + b;
    if constexpr (op == NumericOp::Subtract) return a - b;
    if constexpr (op == NumericOp::Multiply) return a * b;
    if constexpr (op == NumericOp::Divide) return a / b;
    if constexpr (op == NumericOp::Equal) return a == b;
    if constexpr (op == NumericOp::Less) return a < b;
    if constexpr (op == NumericOp::LessEqual) return a <= b;
    if constexpr (op == NumericOp::GreaterEqual) return a >= b;
}
template<NumericOp op>
using NumericResult = std::conditional_t<(op >= NumericOp::Equal),bool,TaggedNumeric>;

/** @brief Exact integer shape for mixed COUNT/Int64 operands, without double or a wider SQL type.
 * @details An unsigned COUNT is not permission to turn a negative SQL operand into
 * a huge positive number. Magnitudes also handle INT64_MIN without negating it.
 * As with ordinary scalar arithmetic, intermediate results must be representable.
 */
struct IntegerMagnitude {
    uint64_t magnitude;
    bool negative;
    IntegerMagnitude(uint64_t value, bool sign) : magnitude(value), negative(sign && value != 0) {}
};
inline IntegerMagnitude integerMagnitude(TaggedNumeric input) {
    if (input.type == DB_TYPE_UINT64) return {input.value.u64, false};
    const auto value = input.value.i64;
    const auto bits = static_cast<uint64_t>(value);
    return {value < 0 ? uint64_t{0} - bits : bits, value < 0};
}
inline bool integerLess(IntegerMagnitude a, IntegerMagnitude b) {
    if (a.negative != b.negative) return a.negative;
    return a.negative ? a.magnitude > b.magnitude : a.magnitude < b.magnitude;
}
inline IntegerMagnitude integerAdd(IntegerMagnitude a, IntegerMagnitude b) {
    if (a.negative == b.negative) return {a.magnitude + b.magnitude, a.negative};
    if (a.magnitude < b.magnitude) { const auto temporary = a; a = b; b = temporary; }
    return {a.magnitude - b.magnitude, a.negative};
}
/** @brief Magnitude operations reuse the same compile-time dispatcher as native numbers. */
inline IntegerMagnitude operator+(IntegerMagnitude a, IntegerMagnitude b) { return integerAdd(a, b); }
inline IntegerMagnitude operator-(IntegerMagnitude a, IntegerMagnitude b) {
    return integerAdd(a, {b.magnitude, !b.negative});
}
inline IntegerMagnitude operator*(IntegerMagnitude a, IntegerMagnitude b) {
    return {a.magnitude * b.magnitude, a.negative != b.negative};
}
inline IntegerMagnitude operator/(IntegerMagnitude a, IntegerMagnitude b) {
    return {a.magnitude / b.magnitude, a.negative != b.negative};
}
inline bool operator==(IntegerMagnitude a, IntegerMagnitude b) { return a.magnitude == b.magnitude && a.negative == b.negative; }
inline bool operator<(IntegerMagnitude a, IntegerMagnitude b) { return integerLess(a, b); }
inline bool operator<=(IntegerMagnitude a, IntegerMagnitude b) { return !integerLess(b, a); }
inline bool operator>=(IntegerMagnitude a, IntegerMagnitude b) { return !integerLess(a, b); }
inline TaggedNumeric taggedInteger(IntegerMagnitude value) {
    if (!value.negative) return TaggedNumeric(value.magnitude);
    if (value.magnitude == (uint64_t{1} << 63)) return TaggedNumeric(std::numeric_limits<int64_t>::min());
    return TaggedNumeric(-static_cast<int64_t>(value.magnitude));
}
template<NumericOp op> inline NumericResult<op> numericInteger(TaggedNumeric a, TaggedNumeric b) {
    const auto left = integerMagnitude(a), right = integerMagnitude(b);
    if constexpr (op >= NumericOp::Equal) return numericOperation<op>(left, right);
    else return taggedInteger(numericOperation<op>(left, right));
}

template<NumericOp op> inline NumericResult<op> numericPromoted(TaggedNumeric a, TaggedNumeric b) {
    if (a.type == DB_TYPE_FLOAT64 || b.type == DB_TYPE_FLOAT64)
        return numericOperation<op>(static_cast<double>(a), static_cast<double>(b));
    if (a.type == DB_TYPE_UINT64 || b.type == DB_TYPE_UINT64)
        return numericInteger<op>(a, b);
    return numericOperation<op>(static_cast<int64_t>(a), static_cast<int64_t>(b));
}
inline TaggedNumeric operator+(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Add>(a,b); }
inline TaggedNumeric operator-(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Subtract>(a,b); }
inline TaggedNumeric operator*(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Multiply>(a,b); }
inline TaggedNumeric operator/(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Divide>(a,b); }
inline bool operator==(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Equal>(a,b); }
inline bool operator!=(TaggedNumeric a, TaggedNumeric b) { return !(a == b); }
inline bool operator<(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::Less>(a,b); }
inline bool operator>(TaggedNumeric a, TaggedNumeric b) { return b < a; }
inline bool operator<=(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::LessEqual>(a,b); }
inline bool operator>=(TaggedNumeric a, TaggedNumeric b) { return numericPromoted<NumericOp::GreaterEqual>(a,b); }
} // namespace db
