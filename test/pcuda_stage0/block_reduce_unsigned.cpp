#include "reduce_kernel.hpp"

extern "C" pcudaError_t launch_block_reduce_unsigned(pcudaStream_t stream, const stage0::Word* input,
        stage0::Word* output, std::size_t count, unsigned block) {
    return stage0::launchReduction(stream, input, output, count, block);
}
