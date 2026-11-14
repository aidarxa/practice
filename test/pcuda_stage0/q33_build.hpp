#pragma once

#include "q33.hpp"

namespace q33 {
template <class Identity, bool Date>
pcudaError_t build(pcudaStream_t stream, BuildArgs args) {
    if (!stream || !args.keys || !args.values || !args.table || !args.slots || !validCount(args.count))
        return pcudaErrorInvalidValue;
    if (!args.count) return pcudaSuccess;
    constexpr unsigned tile = block * items;
    const auto grid = static_cast<unsigned>(args.count / tile + (args.count % tile != 0));
    return pcudaParallelFor(grid, block, 0, stream, [=]() {
        const std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.x;
        for (unsigned item = 0; item < items; ++item) {
            const auto i = base + static_cast<std::size_t>(item) * block;
            if (i < args.count) {
                const int value = args.values[i];
                const bool selected = Date ? (value >= args.first && value <= args.last)
                                           : (value == args.first || value == args.last);
                const auto slot = static_cast<long long>(args.keys[i]) - args.key_min;
                if (selected && slot >= 0 && static_cast<unsigned long long>(slot) < args.slots)
                    args.table[slot] = value; // unique dimension keys verified by host
            }
        }
    });
}
} // namespace q33
