#pragma once

#include "pch.hpp"
#include <limits>

namespace stage0 {
constexpr unsigned maxBlock = 256;
using Word = unsigned long long;
using SignedWord = long long;

// Test-only signatures. No allocations, waiting, or host execution plan in modules.
using ScanLaunch = pcudaError_t (*)(pcudaStream_t, const Word*, Word*, Word*, std::size_t, unsigned);
using AddLaunch = pcudaError_t (*)(pcudaStream_t, Word*, const Word*, std::size_t, unsigned);
using ReduceLaunch = pcudaError_t (*)(pcudaStream_t, const SignedWord*, SignedWord*, std::size_t, unsigned);
using UnsignedReduceLaunch = pcudaError_t (*)(pcudaStream_t, const Word*, Word*, std::size_t, unsigned);

struct AtomicState {
    unsigned count32;
    unsigned claims32;
    int key32;
    int exchange32;
    Word sum64;
    Word min64;
    Word max64;
    Word bits64;
    Word cas64;
};
using AtomicLaunch = pcudaError_t (*)(pcudaStream_t, AtomicState*, unsigned, unsigned, bool);

inline bool validCollective(pcudaStream_t stream, std::size_t count, unsigned block) {
    return stream && block > 0 && block <= maxBlock &&
           count <= std::numeric_limits<std::size_t>::max() / sizeof(Word) &&
           count / block + (count % block != 0) <= std::numeric_limits<unsigned>::max();
}

// All threads, including padded lanes, must call each collective.
template <class T, class Op>
inline T blockReduce(T value, T* shared, unsigned tid, unsigned block, Op op) {
    shared[tid] = value;
    __syncthreads();
    for (unsigned stride = 1; stride < block; stride *= 2) {
        if (tid % (2 * stride) == 0 && tid + stride < block)
            shared[tid] = op(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    const T result = shared[0];
    __syncthreads(); // shared may be reused by the next collective
    return result;
}

template <class T> struct Plus {
    T operator()(T a, T b) const { return a + b; }
};
template <class T> struct Minimum {
    T operator()(T a, T b) const { return a < b ? a : b; }
};
template <class T> struct Maximum {
    T operator()(T a, T b) const { return a > b ? a : b; }
};
} // namespace stage0
