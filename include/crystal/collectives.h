#pragma once
/** @file
 * @brief Work-group collectives implemented using public PCUDA shared memory/barriers.
 * @details Every thread of a 1D block must call in uniform control flow, including
 * padded lanes (supply the operation identity). Scratch has at least block entries.
 * The final barrier permits immediate scratch reuse. No warp/subgroup size assumed.
 */
#include <pcuda.hpp>
#include <limits>

namespace crystal {
template <typename T> struct Sum { T operator()(T a,T b) const { return a+b; } };
template <typename T> struct Minimum { T operator()(T a,T b) const { return a<b?a:b; } };
template <typename T> struct Maximum { T operator()(T a,T b) const { return a>b?a:b; } };

/** @brief Broadcast a commutative associative reduction, including non-power-of-two blocks. */
template <typename T,typename Op>
inline T blockReduce(T value,T* scratch,unsigned tid,unsigned block,Op op) {
    scratch[tid] = value;
    __syncthreads();
    for (unsigned active=block;active>1;active=(active+1)/2) {
        const auto half = (active+1)/2;
        if (tid < active/2) scratch[tid] = op(scratch[tid],scratch[tid+half]);
        __syncthreads();
    }
    const T result = scratch[0];
    __syncthreads();
    return result;
}

/** @brief Inclusive Hillis-Steele scan converted to exclusive with an explicit identity.
 * @pre block <= 1024; op is associative and preserves the supplied identity.
 * Arithmetic bounds are a caller contract; CheckedSum supplies overflow detection.
 */
template <typename T,typename Op>
inline T blockExclusiveScan(T value,T* scratch,unsigned tid,unsigned block,T identity,Op op) {
    scratch[tid] = value;
    __syncthreads();
    for (unsigned offset=1;offset<block;offset*=2) {
        const T left = tid>=offset ? scratch[tid-offset] : identity;
        const T right = scratch[tid];
        __syncthreads();
        scratch[tid] = op(left,right);
        __syncthreads();
    }
    const T result = tid ? scratch[tid-1] : identity;
    __syncthreads();
    return result;
}

/** @brief Mark overflow of nonnegative 64-bit counts; output must be discarded on error. */
struct CheckedSum {
    int* overflow;
    unsigned long long operator()(unsigned long long a,unsigned long long b) const {
        if (b > std::numeric_limits<unsigned long long>::max()-a) atomicExch(overflow,1);
        return a+b;
    }
};
}
