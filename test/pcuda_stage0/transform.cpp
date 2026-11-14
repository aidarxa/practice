#include "probe.hpp"

namespace transform_probe {
pcudaError_t launch(pcudaStream_t stream, std::uint32_t* data, std::size_t count,
                    unsigned block, std::uint32_t value) {
    if (!validProbeLaunch(stream, data, count, block)) return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = static_cast<unsigned>(count / block + (count % block != 0));
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count) data[i] = data[i] * std::uint32_t{3} + value;
    });
}
}

extern "C" pcudaError_t launch_probe(pcudaStream_t stream, std::uint32_t* data,
                                    std::size_t count, unsigned block, std::uint32_t value) {
    return transform_probe::launch(stream, data, count, block, value);
}
