#pragma once
#include <cstdint>
#include <type_traits>

namespace db {

// 1. Механизм расширения типов (Type Promotion)
template <typename T> struct PromoteType { using type = T; }; // Fallback
template <> struct PromoteType<int32_t>  { using type = int64_t; };
template <> struct PromoteType<uint32_t> { using type = uint64_t; };
template <> struct PromoteType<float>    { using type = double; };

// 2. Безопасные арифметические операции (принимают РАЗНЫЕ типы T1 и T2)
template <typename T1, typename T2>
inline auto safe_add(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>; // Находим общий безопасный тип
    return static_cast<Common>(a) + static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_sub(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) - static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_mul(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) * static_cast<Common>(b);
}

template <typename T1, typename T2>
inline auto safe_div(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) / static_cast<Common>(b);
}

// 3. Безопасные сравнения (возвращают int 1 или 0 для масок)
template <typename T1, typename T2>
inline int safe_lt(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) < static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_gt(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) > static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_lte(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) <= static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_gte(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) >= static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_eq(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) == static_cast<Common>(b) ? 1 : 0;
}

template <typename T1, typename T2>
inline int safe_neq(T1 a, T2 b) {
    using P1 = typename PromoteType<T1>::type;
    using P2 = typename PromoteType<T2>::type;
    using Common = std::common_type_t<P1, P2>;
    return static_cast<Common>(a) != static_cast<Common>(b) ? 1 : 0;
}

} // namespace db