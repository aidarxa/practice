#include "core/pcuda_services.h"
#include "core/size_expression.h"
#include <pcuda.hpp>
#include <algorithm>
#include <stdexcept>

namespace db {
void compactSparseRows(DeviceRuntime& runtime, const unsigned long long* source,
                       unsigned long long* target, unsigned long long* count,
                       size_t physical_rows, size_t tuple_size) {
    if (!count || !tuple_size) throw std::runtime_error("invalid sparse compaction shape/count");
    checkedSizeMultiply(checkedSizeMultiply(physical_rows, tuple_size), sizeof(unsigned long long));
    runtime.requireCapabilities(false, true, 0);
    runtime.fillAsync(count, 0, sizeof(*count));
    if (!physical_rows) return;
    if (!source || !target) throw std::runtime_error("null sparse compaction storage");
    const auto block = static_cast<unsigned>(std::min({128, runtime.properties().maxThreadsPerBlock,
                                                      runtime.properties().maxThreadsDim[0]}));
    if (!block) throw std::runtime_error("GPU has no supported block size");
    const auto grid = physical_rows / block + (physical_rows % block != 0);
    runtime.validateLaunch({{grid,1,1},{block,1,1},0});
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(grid)), dim3(block), 0, runtime.stream(), [=] {
        const auto row = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row >= physical_rows) return;
        const auto base = row * tuple_size;
        bool present = false;
        for (size_t column = 0; column < tuple_size; ++column) present |= source[base + column] != 0;
        if (!present) return;
        const auto output = atomicAdd(count, 1ULL) * tuple_size;
        for (size_t column = 0; column < tuple_size; ++column) target[output + column] = source[base + column];
    }), "compact sparse PCUDA result");
}
}
