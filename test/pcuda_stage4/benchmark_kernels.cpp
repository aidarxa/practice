/** @file @brief Controlled old-header/SYCL baseline versus production PCUDA primitives. */
#include "benchmark_api.h"
#include <cstdint>
#include <stdexcept>
#include "crystal/load.h"
#include "crystal/pred.h"
#include "crystal/join.h"
#ifdef STAGE4_BASELINE
#include <sycl/sycl.hpp>
#else
#include "crystal/collectives.h"
#endif
#include "core/pcuda_runtime.h"

using Word=unsigned long long;
constexpr unsigned threads=128,items=4;
template <typename F> static void map(const BenchCall& call,size_t groups,F body) {
#ifdef STAGE4_BASELINE
    auto& queue=*static_cast<sycl::queue*>(call.legacy_queue);
    const auto event=queue.parallel_for(sycl::nd_range<1>(groups*threads,threads),[=](sycl::nd_item<1> it) {
        body(static_cast<unsigned>(it.get_local_linear_id()),static_cast<size_t>(it.get_group_linear_id()));
    });
    if (event.template get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::submitted)
        throw std::runtime_error("baseline requires instant submission");
#else
    db::checkPcudacall(pcudaParallelFor(dim3(groups),dim3(threads),0,static_cast<pcudaStream_t>(call.stream),[=] {
        body(threadIdx.x,static_cast<size_t>(blockIdx.x));
    }),"benchmark map");
#endif
}
static void loadPred(const BenchCall& c) {
    const auto groups=c.count/(threads*items)+(c.count%(threads*items)!=0);
    map(c,groups,[=](unsigned tid,size_t group) {
        const auto base=group*threads*items;
        const auto count=c.count-base<threads*items?c.count-base:threads*items;
        int values[items],flags[items];
        BlockLoad<int,threads,items>(c.input+base,tid,base,values,count);
        BlockPredGT<int,threads,items>(tid,values,flags,7,count);
        for (unsigned i=0;i<items;++i) if (tid+i*threads<count) c.output[base+tid+i*threads]=flags[i]?values[i]:0;
    });
}
static void pht(const BenchCall& c) {
    map(c,c.key_count/(threads*items),[=](unsigned tid,size_t group) {
        const auto base=group*threads*items;
        int keys[items],values[items],flags[items];
        BlockLoad<int,threads,items>(c.keys+base,tid,base,keys,threads*items);
        InitFlags<threads,items>(flags);
        for (unsigned i=0;i<items;++i) { values[i]=keys[i]*3; flags[i]=keys[i]%3!=0; }
#ifdef STAGE4_BASELINE
        BlockBuildSelectivePHT_2<int,int,threads,items>(tid,keys,values,flags,c.table,c.key_count,0,threads*items);
#else
        BlockBuildSelectivePHT_2<int,int,threads,items>(tid,keys,values,flags,c.table,c.key_count,0,threads*items,c.occupied);
#endif
    });
    map(c,c.count/(threads*items)+(c.count%(threads*items)!=0),[=](unsigned tid,size_t group) {
        const auto base=group*threads*items;
        const auto count=c.count-base<threads*items?c.count-base:threads*items;
        int keys[items],values[items]={},flags[items];
        BlockLoad<int,threads,items>(c.input+base,tid,base,keys,count);
        InitFlags<threads,items>(flags);
#ifdef STAGE4_BASELINE
        BlockProbeAndPHT_2<int,int,threads,items>(tid,keys,values,flags,c.table,c.key_count,0,count);
#else
        BlockProbeAndPHT_2<int,int,threads,items>(tid,keys,values,flags,c.table,c.key_count,0,count,c.occupied);
#endif
        for (unsigned i=0;i<items;++i) if (tid+i*threads<count) c.output[base+tid+i*threads]=flags[i]?static_cast<Word>(values[i]):UINT64_MAX;
    });
}
static void collective(const BenchCall& c,bool scan) {
    const auto groups=c.count/threads+(c.count%threads!=0);
#ifdef STAGE4_BASELINE
    auto& queue=*static_cast<sycl::queue*>(c.legacy_queue);
    const auto event=queue.parallel_for(sycl::nd_range<1>(groups*threads,threads),[=](sycl::nd_item<1> it) {
        const auto row=it.get_global_linear_id();
        const Word value=row<c.count?c.input[row]:0ULL;
        if (scan) {
            const auto prefix=sycl::exclusive_scan_over_group(it.get_group(),value,sycl::plus<Word>{});
            if (row<c.count) c.output[row]=prefix;
        } else {
            const auto sum=sycl::reduce_over_group(it.get_group(),value,sycl::plus<Word>{});
            if (it.get_local_linear_id()==0) c.output[it.get_group_linear_id()]=sum;
        }
    });
    if (event.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::submitted)
        throw std::runtime_error("baseline collective requires instant submission");
#else
    db::checkPcudacall(pcudaParallelFor(dim3(groups),dim3(threads),0,static_cast<pcudaStream_t>(c.stream),[=] {
        __shared__ Word shared[threads];
        const auto row=static_cast<Word>(blockIdx.x)*blockDim.x+threadIdx.x;
        const Word value=row<c.count?c.input[row]:0ULL;
        if (scan) {
            const auto prefix=crystal::blockExclusiveScan(value,shared,threadIdx.x,blockDim.x,0ULL,crystal::Sum<Word>{});
            if (row<c.count) c.output[row]=prefix;
        } else {
            const auto sum=crystal::blockReduce(value,shared,threadIdx.x,blockDim.x,crystal::Sum<Word>{});
            if (threadIdx.x==0) c.output[blockIdx.x]=sum;
        }
    }),"benchmark collective");
#endif
}
void benchmarkLaunch(unsigned operation,const BenchCall& call) {
    switch (operation) {
        case 0: loadPred(call); break;
        case 1: pht(call); break;
        case 2: collective(call,false); break;
        case 3: collective(call,true); break;
        default: throw std::invalid_argument("unknown primitive benchmark");
    }
}
