#include "reduce_kernel.hpp"

extern "C" pcudaError_t launch_block_reduce(pcudaStream_t stream, const stage0::SignedWord* input,
        stage0::SignedWord* output, std::size_t count, unsigned block) {
    return stage0::launchReduction(stream, input, output, count, block);
}
