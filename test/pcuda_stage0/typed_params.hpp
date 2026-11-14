#pragma once

#include "pch.hpp"
#include <type_traits>

namespace param_probe {
#ifdef STAGE0_FOUR_PARAMS
constexpr unsigned parameterCount = 4;
#else
constexpr unsigned parameterCount = 3;
#endif
constexpr unsigned tagIdentity = 1u << parameterCount;
enum : unsigned { integer = 0, floating = 1 };
struct Scalar {
    union { long long i; double d; } value;
    unsigned tag;
    unsigned valid;
};
struct Parameters { Scalar slots[parameterCount]; };
static_assert(std::is_trivial<Scalar>::value && std::is_standard_layout<Scalar>::value, "POD test values");
static_assert(sizeof(Scalar) == 16, "unexpected experimental layout");
using Launch = pcudaError_t (*)(pcudaStream_t, const Scalar*, Scalar*, std::size_t, Parameters);

template <class T> T get(Scalar scalar) {
    if constexpr (std::is_same<T, double>::value) return scalar.value.d;
    else return scalar.value.i;
}

template <class T> Scalar pack(T value, bool valid) {
    Scalar scalar{};
    scalar.valid = valid;
    if constexpr (std::is_same<T, double>::value) {
        scalar.tag = floating;
        scalar.value.d = valid ? value : 0.0;
    } else {
        scalar.tag = integer;
        scalar.value.i = valid ? value : 0;
    }
    return scalar;
}

template <unsigned Mask>
Scalar typed(Scalar input, Parameters parameters) {
    using A = std::conditional_t<(Mask & 1) != 0, double, long long>;
    using B = std::conditional_t<(Mask & 2) != 0, double, long long>;
    using C = std::conditional_t<(Mask & 4) != 0, double, long long>;
    using D = std::conditional_t<(Mask & 8) != 0, double, long long>;
    using Prefix = decltype((input.value.i + A{}) * B{} + C{});
    using R = std::conditional_t<parameterCount == 4, decltype(Prefix{} + D{}), Prefix>;
    bool valid = input.valid != 0;
    for (const auto& slot : parameters.slots) valid = valid && slot.valid;
    if (!valid) return pack(R{}, false);
    const auto prefix = (input.value.i + get<A>(parameters.slots[0])) * get<B>(parameters.slots[1]) +
                        get<C>(parameters.slots[2]);
    if constexpr (parameterCount == 4) return pack(prefix + get<D>(parameters.slots[3]), true);
    else return pack(prefix, true);
}

inline Scalar binary(Scalar a, Scalar b, bool multiply) {
    const bool fp = a.tag == floating || b.tag == floating;
    const bool valid = a.valid && b.valid;
    if (!valid) return fp ? pack(0.0, false) : pack(0LL, false);
    if (fp) {
        const double left = a.tag == floating ? a.value.d : static_cast<double>(a.value.i);
        const double right = b.tag == floating ? b.value.d : static_cast<double>(b.value.i);
        return pack(multiply ? left * right : left + right, true);
    }
    return pack(multiply ? a.value.i * b.value.i : a.value.i + b.value.i, true);
}
} // namespace param_probe
