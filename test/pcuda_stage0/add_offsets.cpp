#include "collectives.hpp"

extern "C" pcudaError_t launch_add_offsets(pcudaStream_t stream, stage0::Word* output,
        const stage0::Word* offsets, std::size_t count, unsigned block) {
    if (!stage0::validCollective(stream, count, block) || (count && (!output || !offsets)))
        return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = static_cast<unsigned>(count / block + (count % block != 0));
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count) output[i] += offsets[blockIdx.x];
    });
}
