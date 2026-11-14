#pragma once

#include "collectives.hpp"

namespace stage0 {
template <class T>
pcudaError_t launchReduction(pcudaStream_t stream, const T* input, T* output,
                             std::size_t count, unsigned block) {
    if (!validCollective(stream, count, block) || (count && (!input || !output)))
        return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = static_cast<unsigned>(count / block + (count % block != 0));
    if (grid > std::numeric_limits<std::size_t>::max() / (3 * sizeof(T)))
        return pcudaErrorInvalidValue;
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        __shared__ T shared[maxBlock];
        const unsigned tid = threadIdx.x;
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
        const T value = i < count ? input[i] : T{0};
        const auto sum = blockReduce(value, shared, tid, block, Plus<T>{});
        const auto min = blockReduce(i < count ? value : std::numeric_limits<T>::max(),
                                     shared, tid, block, Minimum<T>{});
        const auto max = blockReduce(i < count ? value : std::numeric_limits<T>::lowest(),
                                     shared, tid, block, Maximum<T>{});
        if (tid == 0) {
            const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 3;
            output[base] = sum;
            output[base + 1] = min;
            output[base + 2] = max;
        }
    });
}
} // namespace stage0
