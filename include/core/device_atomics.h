#pragma once
/** @file
 * @brief Public PCUDA device-scope relaxed integer atomics for aggregate storage.
 * @details Runtime checks atomic64 support before launch. Cross-kernel visibility
 * comes from the one ordered stream, not from these relaxed operations. Bitmap
 * words are raw aligned device allocations, viewed as PCUDA's unsigned 64-bit word.
 */
#include <pcuda.hpp>
#include <cstdint>
#include "core/aggregate_numeric.h"

namespace db {
inline unsigned long long atomic_fetch_add_ull(unsigned long long& target, unsigned long long value) { return atomicAdd(&target,value); }
inline void atomic_add_ull(unsigned long long& target, unsigned long long value) { atomicAdd(&target,value); }
inline void atomic_min_ull(unsigned long long& target, unsigned long long value) { atomicMin(&target,value); }
inline void atomic_max_ull(unsigned long long& target, unsigned long long value) { atomicMax(&target,value); }
/** @brief Relaxed, device-scope CAS on raw storage, with signed/Float64 comparison semantics.
 * @details Sum of integer states can use native bitwise-equivalent atomicAdd.
 * Float64 does not require native FP64 atomicAdd, only checked FP64/atomic64 support.
 * The initial load is atomic too; all concurrent state access uses this protocol.
 */
template<class T, AggregateOp op> inline void atomicAggregate(unsigned long long& target, T value) {
    if constexpr (op == AggregateOp::Sum && std::is_integral_v<T>) {
        atomicAdd(&target, aggregateBits(value));
    } else {
        auto observed = atomicCAS(&target, 0ULL, 0ULL);
        for (;;) {
            const auto desired = aggregateBits(AggregateMerge<T, op>{}(aggregateValue<T>(observed), value));
            const auto previous = atomicCAS(&target, observed, desired);
            if (previous == observed) return;
            observed = previous;
        }
    }
}
/** @brief Set a valid bit atomically; null denotes a nonnullable output. */
inline void atomic_set_valid_bit(std::uint64_t* bitmap, unsigned long long bit) {
    static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long), "PCUDA 64-bit word required");
    static_assert(alignof(std::uint64_t) >= alignof(unsigned long long), "PCUDA bitmap alignment required");
    if (bitmap) atomicOr(reinterpret_cast<unsigned long long*>(bitmap) + (bit >> 6), 1ULL << (bit & 63));
}
}
