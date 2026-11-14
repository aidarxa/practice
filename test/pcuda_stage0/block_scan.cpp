#include "collectives.hpp"

extern "C" pcudaError_t launch_block_scan(pcudaStream_t stream, const stage0::Word* input,
        stage0::Word* output, stage0::Word* sums, std::size_t count, unsigned block) {
    using namespace stage0;
    if (!validCollective(stream, count, block) || (count && (!input || !output || !sums)))
        return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = static_cast<unsigned>(count / block + (count % block != 0));
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        __shared__ Word shared[maxBlock];
        const unsigned tid = threadIdx.x;
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
        const Word value = i < count ? input[i] : Word{0};
        shared[tid] = value;
        __syncthreads();
        // Hillis-Steele: supports arbitrary block sizes, no subgroup assumptions.
        for (unsigned offset = 1; offset < block; offset *= 2) {
            const Word add = tid >= offset ? shared[tid - offset] : Word{0};
            __syncthreads(); // finish reads before overwriting shared
            shared[tid] += add;
            __syncthreads();
        }
        if (i < count) output[i] = shared[tid] - value;
        if (tid == block - 1) sums[blockIdx.x] = shared[tid];
    });
}
