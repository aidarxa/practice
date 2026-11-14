#pragma once
#include <cstdint>
#include <type_traits>
#include "core/tagged_numeric.h"
/** @file @brief Promoted scalar arithmetic and bit-preserving codecs; no GPU/host I/O API.
 * @details Promotion prevents Int32 intermediate overflow, not arbitrary Int64 SQL
 * overflow. Division requires a nonzero divisor and a representable quotient.
 * Device atomic operations are deliberately separate in core/device_atomics.h.
 */
#include <cstring>

namespace db {

// 1. Механизм расширения типов (Type Promotion)
template <typename T> struct PromoteType { using type = T; }; // Fallback
template <> struct PromoteType<int32_t>  { using type = int64_t; };
template <> struct PromoteType<uint32_t> { using type = uint64_t; };
template <> struct PromoteType<float>    { using type = double; };

/** @brief Mixed unsigned COUNT / signed SQL arithmetic uses exact tagged integer operations.
 * @details The internal tag is not a new SQL parameter/output type. Equal-sign
 * integral and ordinary floating operands keep their native promoted type.
 */
template<class A, class B, bool mixed = std::is_integral_v<A> && std::is_integral_v<B> && (std::is_signed_v<A> != std::is_signed_v<B>)>
struct NumericCommon { using type = std::common_type_t<A, B>; };
template<class A, class B> struct NumericCommon<A, B, true> { using type = TaggedNumeric; };

/** @brief Central promotion policy, shared by arithmetic and comparisons. */
template <typename T1,typename T2>
using PromotedCommon = typename NumericCommon<typename PromoteType<T1>::type,typename PromoteType<T2>::type>::type;

// 2. Безопасные арифметические операции (принимают РАЗНЫЕ типы T1 и T2)
template <typename T1, typename T2>
inline auto safe_add(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) + static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_sub(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) - static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_mul(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) * static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_div(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) / static_cast<Common>(b);
}

// 3. Безопасные сравнения (возвращают int 1 или 0 для масок)
template <typename T1, typename T2>
inline int safe_lt(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) < static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_gt(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) > static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_lte(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) <= static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_gte(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) >= static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_eq(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) == static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_neq(T1 a, T2 b) {
    using Common = PromotedCommon<T1,T2>;
    return static_cast<Common>(a) != static_cast<Common>(b) ? 1 : 0;
}


inline unsigned long long bit_cast_double_to_ull(double value) {
    static_assert(sizeof(value) == sizeof(unsigned long long), "64-bit double representation required");
    unsigned long long bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline double bit_cast_ull_to_double(unsigned long long value) {
    double bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline int bitmap_valid_at(const uint64_t* bitmap, unsigned long long bit_idx) {
    if (bitmap == nullptr) return 1;
    return static_cast<int>((bitmap[bit_idx >> 6] >> (bit_idx & 63ULL)) & 1ULL);
}

} // namespace db
