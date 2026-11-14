#include "core/pcuda_plan_services.h"
#include <pcuda.hpp>
#include <climits>
#include <stdexcept>
namespace db {
namespace {
KernelLaunchConfigV1 conversionLaunch(uint64_t count) {
    return {{count/256+(count%256!=0),1,1},{256,1,1},0};
}
}
PlanPrefixService::PlanPrefixService(std::shared_ptr<DeviceRuntime> runtime)
    : runtime_(std::move(runtime)),scan_(runtime_),input_(runtime_,0),output_(runtime_,0) {}
void PlanPrefixService::widen(const int* source,uint64_t count) {
    if(!count) return;
    const auto launch=conversionLaunch(count);runtime_->validateLaunch(launch);
    auto* destination=input_.data();
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(launch.grid[0])),dim3(256),0,runtime_->stream(),[=] {
        const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        if(i<count) destination[i]=static_cast<unsigned>(source[i]);
    }),"prefix count widening");
}
void PlanPrefixService::narrow(int* destination,uint64_t count) {
    if(!count) return;
    const auto launch=conversionLaunch(count);runtime_->validateLaunch(launch);
    const auto* source=output_.data();
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(launch.grid[0])),dim3(256),0,runtime_->stream(),[=] {
        const uint64_t i=static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        if(i<count) destination[i]=static_cast<int>(source[i]);
    }),"prefix offset narrowing");
}
uint64_t PlanPrefixService::run(void* destination,const void* source,uint32_t type,uint64_t count) {
    if(count && (!destination || !source)) throw std::invalid_argument("null prefix input/output");
    if(type==DB_TYPE_UINT64) {
        scan_.enqueue(static_cast<const unsigned long long*>(source),static_cast<unsigned long long*>(destination),count);
        return scan_.finish();
    }
    if(type!=DB_TYPE_INT32) throw std::runtime_error("prebuilt prefix requires Int32/UInt64 counts");
    input_.ensureCapacity(count);output_.ensureCapacity(count);
    widen(static_cast<const int*>(source),count);
    scan_.enqueue(input_.data(),output_.data(),count);
    const auto total=scan_.finish();
    if(total>INT32_MAX) throw std::overflow_error("negative/overflowed Int32 prefix counts");
    narrow(static_cast<int*>(destination),count);
    return total;
}
}
