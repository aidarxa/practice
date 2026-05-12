#pragma once
#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>

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


inline unsigned long long bit_cast_double_to_ull(double value) {
    union Bits { double d; unsigned long long u; };
    Bits bits;
    bits.d = value;
    return bits.u;
}

inline double bit_cast_ull_to_double(unsigned long long value) {
    union Bits { double d; unsigned long long u; };
    Bits bits;
    bits.u = value;
    return bits.d;
}

// Atomic helpers used by generated aggregate kernels.
inline unsigned long long atomic_fetch_add_ull(unsigned long long& target, unsigned long long value) {
    sycl::atomic_ref<unsigned long long,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> at(target);
    return at.fetch_add(value);
}

inline void atomic_add_ull(unsigned long long& target, unsigned long long value) {
    sycl::atomic_ref<unsigned long long,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> at(target);
    at.fetch_add(value);
}

inline void atomic_min_ull(unsigned long long& target, unsigned long long value) {
    sycl::atomic_ref<unsigned long long,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> at(target);
    unsigned long long observed = at.load();
    while (value < observed && !at.compare_exchange_weak(observed, value)) {}
}

inline void atomic_max_ull(unsigned long long& target, unsigned long long value) {
    sycl::atomic_ref<unsigned long long,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> at(target);
    unsigned long long observed = at.load();
    while (value > observed && !at.compare_exchange_weak(observed, value)) {}
}

inline void atomic_set_valid_bit(uint64_t* bitmap, unsigned long long bit_idx) {
    if (bitmap == nullptr) return;
    sycl::atomic_ref<uint64_t,
                     sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> at(bitmap[bit_idx >> 6]);
    at.fetch_or(1ULL << (bit_idx & 63ULL));
}

inline int bitmap_valid_at(const uint64_t* bitmap, unsigned long long bit_idx) {
    if (bitmap == nullptr) return 1;
    return static_cast<int>((bitmap[bit_idx >> 6] >> (bit_idx & 63ULL)) & 1ULL);
}

} // namespace db