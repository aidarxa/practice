#pragma once
/** @file @brief Prebuilt prefix services; SQL module compilation contains no universal scan kernels. */
#include "core/pcuda_scan.h"
namespace db {
/** @brief Single-writer reusable service state, with budgeted Int32 conversion scratch. */
class PlanPrefixService {
public:
    explicit PlanPrefixService(std::shared_ptr<DeviceRuntime> runtime);
    /** @brief Exclusive prefix of Int32 or UInt64 counts; return checked total after the required CPU boundary.
     * @details Int32 counts are widened without signed arithmetic. Any negative input
     * necessarily makes the unsigned total exceed INT32_MAX and is rejected before
     * narrowing offsets. Query row materialization stays entirely on the GPU.
     */
    uint64_t run(void* destination,const void* source,uint32_t type,uint64_t count);
private:
    std::shared_ptr<DeviceRuntime> runtime_;
    ExclusiveScan scan_;
    DynamicDeviceBuffer<unsigned long long> input_;
    DynamicDeviceBuffer<unsigned long long> output_;
    void widen(const int* source,uint64_t count);
    void narrow(int* destination,uint64_t count);
};
}
