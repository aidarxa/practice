#pragma once
/** @file @brief Runtime-owned high-water workspace for checked, arbitrary-depth PCUDA scan. */
#include "core/device_buffer.h"
#include "core/scan_layout.h"

namespace db {
/** @brief Single-writer scan session: enqueue all levels, then finish at a CPU consumption boundary.
 * @details Kernels never allocate. Inputs/outputs must remain alive through finish;
 * exact in-place scan is supported, partially overlapping ranges are not. A pending
 * operation must be finished before reuse. Overflow invalidates the entire output.
 * Destruction still drains via RAII buffers if the caller exits on an exception.
 */
class ExclusiveScan {
public:
    explicit ExclusiveScan(std::shared_ptr<DeviceRuntime> runtime);
    /** @brief Validate the complete hierarchy before allocation/submission, then enqueue without waits. */
    void enqueue(const unsigned long long* input,unsigned long long* output,uint64_t count,unsigned block=256);
    /** @brief Wait for final status/total; throw rather than silently use overflowed counts. */
    unsigned long long finish();
    uint64_t scratchCapacity() const { return scratch_.capacity(); }
private:
    std::shared_ptr<DeviceRuntime> runtime_;
    DynamicDeviceBuffer<unsigned long long> scratch_;
    DynamicDeviceBuffer<int> overflow_;
    ScanLayout layout_;
    bool pending_ = false;
    bool submitted_ = false;
    void validate(const unsigned long long* input,unsigned long long* output,unsigned block) const;
    void scanLevels(const unsigned long long* input,unsigned long long* output,unsigned block);
    void addLevels(unsigned long long* output,unsigned block);
};
}
