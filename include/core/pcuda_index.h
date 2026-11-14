#pragma once
/** @file @brief Device-only 1D index vocabulary, mapped directly to native PCUDA builtins. */
#include <pcuda.hpp>
#include <cstdint>
namespace db {
/** @brief No queue/group objects: each accessor reads the current native launch dimensions. */
struct KernelIndex {
    uint64_t get_global_linear_id() const { return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; }
    unsigned get_local_linear_id() const { return threadIdx.x; }
    unsigned get_group_linear_id() const { return blockIdx.x; }
    unsigned get_group_range(unsigned) const { return gridDim.x; }
};
}
