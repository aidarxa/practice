#include "core/pcuda_scan.h"
#include "crystal/collectives.h"

namespace db {
namespace {
/** @brief Per-block device body, separated from host submission and resource policy. */
inline void scanBlock(const unsigned long long* input,unsigned long long* output,unsigned long long* totals,
                       int* overflow,uint64_t count,unsigned long long* shared) {
    const auto tid=threadIdx.x;
    const auto row=static_cast<unsigned long long>(blockIdx.x)*blockDim.x+tid;
    const auto value=row<count?input[row]:0ULL;
    const crystal::CheckedSum add{overflow};
    const auto prefix=crystal::blockExclusiveScan(value,shared,tid,blockDim.x,0ULL,add);
    if (row<count) output[row]=prefix;
    if (tid==blockDim.x-1) totals[blockIdx.x]=add(prefix,value);
}
/** @brief Uniform block scan: padded lanes contribute zero and still reach every barrier. */
void scanLevel(DeviceRuntime& runtime,const unsigned long long* input,unsigned long long* output,
               unsigned long long* totals,int* overflow,const ScanLevel& level,unsigned block) {
    const auto count=level.items;
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(level.blocks)),dim3(block),0,runtime.stream(),[=] {
        __shared__ unsigned long long shared[1024];
        scanBlock(input,output,totals,overflow,count,shared);
    }),"PCUDA exclusive block scan");
}
void addLevel(DeviceRuntime& runtime,unsigned long long* output,const unsigned long long* offsets,
              int* overflow,const ScanLevel& level,unsigned block) {
    const auto count=level.items;
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(level.blocks)),dim3(block),0,runtime.stream(),[=] {
        const auto row=static_cast<unsigned long long>(blockIdx.x)*blockDim.x+threadIdx.x;
        if (row<count) output[row]=crystal::CheckedSum{overflow}(output[row],offsets[blockIdx.x]);
    }),"PCUDA add scan block offsets");
}
}

ExclusiveScan::ExclusiveScan(std::shared_ptr<DeviceRuntime> runtime)
    : runtime_(std::move(runtime)),scratch_(runtime_),overflow_(runtime_,1) {}

void ExclusiveScan::validate(const unsigned long long* input,unsigned long long* output,unsigned block) const {
    if (!layout_.levels.empty() && (!input || !output)) throw std::invalid_argument("null scan input/output");
    runtime_->requireCapabilities(false,false,1024*sizeof(unsigned long long));
    for (const auto& level:layout_.levels) runtime_->validateLaunch({{level.blocks,1,1},{block,1,1},0});
}
void ExclusiveScan::enqueue(const unsigned long long* input,unsigned long long* output,uint64_t count,unsigned block) {
    if (pending_) throw std::logic_error("finish pending scan before reuse");
    layout_=makeScanLayout(count,block);
    validate(input,output,block);
    scratch_.ensureCapacity(layout_.scratch_elements);
    // Mark pending before the first submission: an API failure must not permit
    // reuse while earlier work still references these workspace buffers.
    pending_=true;
    submitted_=false;
    overflow_.zero();
    scanLevels(input,output,block);
    addLevels(output,block);
    submitted_=true;
}
void ExclusiveScan::scanLevels(const unsigned long long* input,unsigned long long* output,unsigned block) {
    auto* scratch=scratch_.data();
    for (const auto& level:layout_.levels) {
        scanLevel(*runtime_,input,output,scratch+level.offset,overflow_.data(),level,block);
        input=scratch+level.offset;
        output=scratch+level.offset;
    }
}
void ExclusiveScan::addLevels(unsigned long long* output,unsigned block) {
    auto* scratch=scratch_.data();
    for (size_t i=layout_.levels.size();i>1;--i) {
        const auto child=i-2;
        auto* target=child?scratch+layout_.levels[child-1].offset:output;
        addLevel(*runtime_,target,scratch+layout_.levels[child].offset,overflow_.data(),layout_.levels[child],block);
    }
}
unsigned long long ExclusiveScan::finish() {
    if (!pending_) throw std::logic_error("no pending scan");
    // Both copies follow the entire hierarchy; a single consumption wait suffices.
    RuntimeAllocation staging(runtime_,sizeof(unsigned long long)+sizeof(int),MemoryKind::Pinned);
    auto* total=static_cast<unsigned long long*>(staging.data());
    auto* error=reinterpret_cast<int*>(total+1);
    *total=0;
    runtime_->copyAsync(error,overflow_.data(),sizeof(int));
    if (!layout_.levels.empty()) runtime_->copyAsync(total,scratch_.data()+layout_.levels.back().offset,sizeof(*total));
    runtime_->wait();
    pending_=false;
    if (!submitted_) throw std::runtime_error("PCUDA scan submission did not complete; output is invalid");
    if (*error) throw std::overflow_error("PCUDA exclusive scan count overflow");
    return *total;
}
}
