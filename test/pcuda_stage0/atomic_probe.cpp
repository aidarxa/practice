#include "collectives.hpp"

extern "C" pcudaError_t launch_atomics(pcudaStream_t stream, stage0::AtomicState* state,
                                       unsigned count, unsigned block, bool atomic64) {
    if (!stage0::validCollective(stream, count, block) || (count && !state))
        return pcudaErrorInvalidValue;
    if (!count) return pcudaSuccess;
    const auto grid = count / block + (count % block != 0);
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count) {
            atomicAdd(&state->count32, 1u);
            if (atomicCAS(&state->key32, 0, 77) == 0) atomicAdd(&state->claims32, 1u);
            atomicExch(&state->exchange32, 91);
            if (atomic64) {
                const stage0::Word value = (stage0::Word{1} << 54) + i;
                atomicAdd(&state->sum64, stage0::Word{1});
                atomicMin(&state->min64, value);
                atomicMax(&state->max64, value);
                atomicOr(&state->bits64, stage0::Word{1} << (i % 64));
                atomicCAS(&state->cas64, stage0::Word{0}, stage0::Word{99});
            }
        }
    });
}
