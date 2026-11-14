#include "tier.hpp"
#include <limits>

#ifndef JIT_TIER
#error JIT_TIER must distinguish the two module/kernel identities
#endif

template <int Tier>
struct FilterProject {
    TierArgs args;
    void operator()() const {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= args.count) return;
        // WHERE comparison above 2^53, validity and signed integer arithmetic.
        // The host restricts offset so this arithmetic cannot overflow.
        args.output[i] = args.valid[i] && args.values[i] >= args.threshold
            ? (args.values[i] % 997) * 7 + args.offset
            : std::numeric_limits<long long>::min();
    }
};

extern "C" pcudaError_t tier_launch(pcudaStream_t stream, TierArgs args) {
    constexpr unsigned block = 128;
    constexpr std::size_t maxRows = 64 * 1024 * 1024;
    if (!stream || !args.values || !args.valid || !args.output || args.count > maxRows ||
        args.offset < -1000000 || args.offset > 1000000) return pcudaErrorInvalidValue;
    // A zero-row launch deliberately still submits ONE empty block: this is an
    // explicit preparation operation, not a promise that count=0 never compiles.
    const auto grid = static_cast<unsigned>(args.count / block + (args.count % block != 0));
    return pcudaParallelFor(grid ? grid : 1, block, 0, stream, FilterProject<JIT_TIER>{args});
}
