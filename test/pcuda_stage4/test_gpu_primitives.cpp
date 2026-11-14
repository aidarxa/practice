/** @file @brief Native PCUDA primitive checks against independent scalar CPU oracles. */
#include "crystal/device.h"
#include "core/pcuda_scan.h"
#include <algorithm>
#include <iostream>
#include <numeric>

using Word = unsigned long long;
template <typename T> using Buffer = db::DynamicDeviceBuffer<T>;
using Runtime = std::shared_ptr<db::DeviceRuntime>;
static size_t checks=0;
static void require(bool value,const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template <typename F> static void rejects(F operation,const char* message) {
    bool rejected=false;
    try { operation(); } catch (const std::exception&) { rejected=true; }
    require(rejected,message);
}

template <typename T> static void collective(Runtime runtime,unsigned block,size_t count) {
    std::vector<T> input(count);
    for (size_t i=0;i<count;++i) {
        if constexpr (std::is_unsigned<T>::value) input[i]=9007199254740993ULL+i%7;
        else input[i]=static_cast<T>(static_cast<int>(i%17)-8);
    }
    const auto groups=count/block+(count%block!=0);
    Buffer<T> source(runtime,count),result(runtime,groups*3),prefix(runtime,count);
    source.copyFromHost(input);
    runtime->requireCapabilities(std::is_floating_point<T>::value,false,1024*sizeof(T));
    if (groups) {
        runtime->validateLaunch({{groups,1,1},{block,1,1},0});
        auto* data=source.data(); auto* out=result.data(); auto* scan=prefix.data();
        db::checkPcudacall(pcudaParallelFor(dim3(groups),dim3(block),0,runtime->stream(),[=] {
            __shared__ T shared[1024];
            const auto tid=threadIdx.x;
            const auto row=static_cast<Word>(blockIdx.x)*blockDim.x+tid;
            const T value=row<count?data[row]:T{};
            const auto sum=crystal::blockReduce(value,shared,tid,blockDim.x,crystal::Sum<T>{});
            const auto low=crystal::blockReduce(row<count?value:std::numeric_limits<T>::max(),shared,tid,blockDim.x,crystal::Minimum<T>{});
            const auto high=crystal::blockReduce(row<count?value:std::numeric_limits<T>::lowest(),shared,tid,blockDim.x,crystal::Maximum<T>{});
            const auto exclusive=crystal::blockExclusiveScan(value,shared,tid,blockDim.x,T{},crystal::Sum<T>{});
            if (row<count) scan[row]=exclusive;
            if (tid==0) { out[blockIdx.x*3]=sum; out[blockIdx.x*3+1]=low; out[blockIdx.x*3+2]=high; }
        }),"collective fixture");
    }
    std::vector<T> actual,scanned;
    result.copyToHost(actual,groups*3); prefix.copyToHost(scanned,count);
    for (size_t group=0;group<groups;++group) {
        const auto begin=group*block,end=std::min(count,begin+block);
        T sum{},low=std::numeric_limits<T>::max(),high=std::numeric_limits<T>::lowest();
        for (size_t row=begin;row<end;++row) {
            require(scanned[row]==sum,"block exclusive sum");
            sum+=input[row]; low=std::min(low,input[row]); high=std::max(high,input[row]);
        }
        require(actual[group*3]==sum && actual[group*3+1]==low && actual[group*3+2]==high,"SUM/MIN/MAX and scratch reuse");
    }
}

static void scan(Runtime runtime) {
    db::ExclusiveScan workspace(runtime);
    rejects([&] { workspace.finish(); },"finish idle scan");
    rejects([&] { workspace.enqueue(nullptr,nullptr,1); },"null scan storage");
    rejects([&] { workspace.enqueue(nullptr,nullptr,UINT64_MAX); },"scan byte overflow before allocation");
    uint64_t retained=0;
    for (const auto shape:std::vector<std::pair<size_t,unsigned>>{{0,256},{1,256},{255,256},{256,256},{257,256},
            {65535,256},{65536,256},{65537,256},{16777217,256},{1000,3},{2305,48},{19,256}}) {
        const auto count=shape.first; const auto block=shape.second;
        std::vector<Word> input(count);
        for (size_t i=0;i<count;++i) input[i]=i%5;
        Buffer<Word> source(runtime,count),out(runtime,count);
        source.copyFromHost(input);
        workspace.enqueue(source.data(),out.data(),count,block);
        rejects([&] { workspace.enqueue(source.data(),out.data(),count,block); },"pending scan cannot be overwritten");
        const auto total=workspace.finish();
        require(workspace.scratchCapacity()>=retained,"scan high-water reuse");
        retained=workspace.scratchCapacity();
        std::vector<Word> actual; out.copyToHost(actual,count);
        Word expected=0;
        for (size_t i=0;i<count;++i) { require(actual[i]==expected,"hierarchical exclusive scan"); expected+=input[i]; }
        require(total==expected,"hierarchical total");
        // Exact aliasing, including a second operation on the retained workspace.
        if (count<70000) {
            workspace.enqueue(source.data(),source.data(),count,block);
            require(workspace.finish()==expected,"in-place scan total");
            source.copyToHost(actual,count);
            Word prefix=0;
            for (size_t i=0;i<count;++i) { require(actual[i]==prefix,"in-place scan values"); prefix+=input[i]; }
        }
    }
    Buffer<Word> large(runtime,2),out(runtime,2);
    large.copyFromHost(std::vector<Word>{UINT64_MAX,1});
    workspace.enqueue(large.data(),out.data(),2,3);
    rejects([&] { workspace.finish(); },"overflow must invalidate output");
    large.copyFromHost(std::vector<Word>{UINT64_MAX-1,1});
    workspace.enqueue(large.data(),out.data(),2,3);
    require(workspace.finish()==UINT64_MAX,"exact maximum sum is valid");
}

/** @brief Striped load/gather/predicate/validity check on a non-power-of-two block. */
static void tiles(Runtime runtime) {
    constexpr unsigned block=48,items=4;
    constexpr size_t count=197;
    std::vector<Word> input(count);
    std::iota(input.begin(),input.end(),9007199254740993ULL);
    Buffer<Word> source(runtime,count),out(runtime,count);
    Buffer<int> masks(runtime,count);
    Buffer<uint64_t> validity(runtime,4);
    source.copyFromHost(input);
    validity.copyFromHost(std::vector<uint64_t>{0xaaaaaaaaaaaaaaaaULL,UINT64_MAX,1,0});
    auto* data=source.data(); auto* values=out.data(); auto* flags_out=masks.data(); auto* bitmap=validity.data();
    db::checkPcudacall(pcudaParallelFor(dim3(2),dim3(block),0,runtime->stream(),[=] {
        const auto tid=threadIdx.x;
        const size_t base=static_cast<size_t>(blockIdx.x)*block*items;
        const auto size=count-base<block*items?count-base:block*items;
        Word loaded[items],gathered[items]; size_t rows[items]; int flags[items],valid[items];
        BlockLoad<Word,block,items>(data+base,tid,base,loaded,size);
        BlockMakeRowIds<block,items>(tid,base,rows,size);
        InitFlags<block,items>(flags);
        BlockGather<Word,block,items>(data,tid,rows,flags,gathered,size);
        BlockLoadValidity<block,items>(bitmap,tid,base,valid,size);
        BlockApplyValidityAnd<block,items>(tid,flags,valid,size);
        BlockPredAGT<Word,block,items>(tid,loaded,flags,9007199254740994ULL,size);
        for (unsigned i=0;i<items;++i) {
            const auto lane=tid+i*block;
            if (lane<size) { values[base+lane]=gathered[i]; flags_out[base+lane]=flags[i]; }
        }
    }),"striped fixture");
    std::vector<Word> actual; std::vector<int> flags;
    out.copyToHost(actual,count); masks.copyToHost(flags,count);
    require(actual==input,"exact 64-bit striped load/gather");
    const uint64_t bits[]={0xaaaaaaaaaaaaaaaaULL,UINT64_MAX,1,0};
    for (size_t i=0;i<count;++i) require(flags[i]==static_cast<int>(((bits[i/64]>>(i%64))&1) && i>1),"validity/predicate mask");
}

static void pht(Runtime runtime,bool include_zero) {
    constexpr unsigned block=48,items=4,length=8,count=197;
    std::vector<int> keys(count),values(count);
    const int domain[]={-3,-2,0,1,3};
    for (size_t i=0;i<count;++i) { keys[i]=domain[i%5]; values[i]=keys[i]+100; }
    Buffer<int> source(runtime,count),payload(runtime,count),key_table(runtime,length),pair_table(runtime,length*2),out(runtime,12);
    Buffer<unsigned> occupied1(runtime,length),occupied2(runtime,length);
    source.copyFromHost(keys); payload.copyFromHost(values);
    occupied1.zero(); occupied2.zero(); key_table.zero(); pair_table.zero();
    auto* k=source.data(); auto* v=payload.data(); auto* h1=key_table.data(); auto* h2=pair_table.data();
    auto* o1=occupied1.data(); auto* o2=occupied2.data(); auto* result=out.data();
    db::checkPcudacall(pcudaParallelFor(dim3(2),dim3(block),0,runtime->stream(),[=] {
        const auto tid=threadIdx.x; const auto base=static_cast<size_t>(blockIdx.x)*block*items;
        const auto size=count-base<block*items?count-base:block*items;
        int keys[items],vals[items],flags[items];
        BlockLoad<int,block,items>(k+base,tid,base,keys,size);
        BlockLoad<int,block,items>(v+base,tid,base,vals,size);
        InitFlags<block,items>(flags);
        if (!include_zero) BlockPredANEq<int,block,items>(tid,keys,flags,0,size);
        BlockBuildSelectivePHT_1<int,block,items>(tid,keys,flags,h1,length,-3,size,o1);
        BlockBuildSelectivePHT_2<int,int,block,items>(tid,keys,vals,flags,h2,length,-3,size,o2);
    }),"PHT build fixture");
    db::checkPcudacall(pcudaParallelFor(dim3(1),dim3(6),0,runtime->stream(),[=] {
        const int probes[]={-3,-1,0,3,INT32_MIN,INT32_MAX};
        const auto tid=threadIdx.x;
        int value=-1;
        result[tid*2]=ProbePHT_1(probes[tid],h1,length,-3,o1);
        result[tid*2+1]=ProbePHT_2(probes[tid],value,h2,length,-3,o2)?value:-1;
    }),"PHT probe fixture");
    std::vector<int> actual; out.copyToHost(actual,12);
    require(actual==std::vector<int>({1,97,0,-1,include_zero?1:0,include_zero?100:-1,1,103,0,-1,0,-1}),"PHT occupied/absent zero, signed keys and probe aliases");
}

static void mht(Runtime runtime) {
    constexpr unsigned count=1001,length=8,block=48,items=4;
    std::vector<int> input(count); for (unsigned i=0;i<count;++i) input[i]=static_cast<int>(i%5)-2;
    Buffer<int> keys(runtime,count),counts(runtime,length),offsets(runtime,length),position(runtime,length),directory(runtime,length*3),payload(runtime,count);
    Buffer<Word> wide_counts(runtime,length),wide_offsets(runtime,length);
    keys.copyFromHost(input); counts.zero(); position.zero(); directory.zero();
    auto* k=keys.data(); auto* c=counts.data(); auto* off=offsets.data(); auto* pos=position.data(); auto* dir=directory.data(); auto* pay=payload.data();
    auto* wc=wide_counts.data(); auto* wo=wide_offsets.data();
    db::checkPcudacall(pcudaParallelFor(dim3(6),dim3(block),0,runtime->stream(),[=] {
        const auto base=static_cast<size_t>(blockIdx.x)*block*items;
        const auto size=count-base<block*items?count-base:block*items;
        int values[items],flags[items];
        BlockLoad<int,block,items>(k+base,threadIdx.x,base,values,size);
        InitFlags<block,items>(flags);
        BlockBuildMHT_Count<int,block,items>(threadIdx.x,values,flags,c,length,-2,size);
    }),"MHT count fixture");
    db::checkPcudacall(pcudaParallelFor(dim3(1),dim3(length),0,runtime->stream(),[=] { wc[threadIdx.x]=c[threadIdx.x]; }),"widen MHT counts fixture");
    db::ExclusiveScan scan(runtime); scan.enqueue(wc,wo,length,3);
    require(scan.finish()==count,"MHT GPU-resident counts/scan total");
    db::checkPcudacall(pcudaParallelFor(dim3(1),dim3(length),0,runtime->stream(),[=] { off[threadIdx.x]=static_cast<int>(wo[threadIdx.x]); }),"narrow bounded MHT offsets fixture");
    db::checkPcudacall(pcudaParallelFor(dim3(6),dim3(block),0,runtime->stream(),[=] {
        const auto base=static_cast<size_t>(blockIdx.x)*block*items;
        const auto size=count-base<block*items?count-base:block*items;
        int values[items],rows[items],flags[items];
        BlockLoad<int,block,items>(k+base,threadIdx.x,base,values,size);
        BlockMakeRowIds<block,items>(threadIdx.x,base,rows,size);
        InitFlags<block,items>(flags);
        BlockBuildMHT_Write<int,int,block,items>(threadIdx.x,values,rows,flags,dir,off,pos,c,pay,length,-2,size);
    }),"MHT write fixture");
    std::vector<int> actual,entries; payload.copyToHost(actual,count); directory.copyToHost(entries,length*3);
    size_t begin=0;
    for (int key=-2;key<=2;++key) {
        std::vector<int> expected;
        for (size_t row=0;row<input.size();++row) if (input[row]==key) expected.push_back(row);
        const auto slot=static_cast<size_t>(key+2);
        require(entries[slot*3]==key && entries[slot*3+1]==static_cast<int>(begin) && entries[slot*3+2]==static_cast<int>(expected.size()),"MHT one-writer directory");
        std::vector<int> bucket(actual.begin()+begin,actual.begin()+begin+expected.size());
        std::sort(bucket.begin(),bucket.end()); require(bucket==expected,"MHT multiplicity and exact payload positions");
        begin+=expected.size();
    }
}

static void atomics(Runtime runtime) {
    runtime->requireCapabilities(true,true,0);
    Buffer<Word> state(runtime,6); Buffer<uint64_t> bitmap(runtime,3); Buffer<long long> key64(runtime,1);
    state.copyFromHost(std::vector<Word>{0,UINT64_MAX,0,0,0,0}); bitmap.zero(); key64.zero();
    auto* s=state.data(); auto* bits=bitmap.data(); auto* key=key64.data();
    db::checkPcudacall(pcudaParallelFor(dim3(17),dim3(48),0,runtime->stream(),[=] {
        const Word row=static_cast<Word>(blockIdx.x)*blockDim.x+threadIdx.x;
        db::atomic_add_ull(s[0],9007199254740993ULL+row);
        db::atomic_min_ull(s[1],9007199254740993ULL+row);
        db::atomic_max_ull(s[2],9007199254740993ULL+row);
        db::atomic_set_valid_bit(bits,row%128);
        db::atomic_set_valid_bit(nullptr,row);
        if (row==0) db::atomic_set_valid_bit(bits,130);
        if (crystal::compareExchangeKey(key,0LL,-9007199254740993LL)==0)
            s[5]=db::atomic_fetch_add_ull(s[4],1);
        if (row==0) s[3]=db::bit_cast_double_to_ull(db::bit_cast_ull_to_double(0x8000000000000000ULL));
    }),"atomic fixture");
    std::vector<Word> actual; std::vector<uint64_t> valid;
    state.copyToHost(actual,6); bitmap.copyToHost(valid,3);
    constexpr Word count=17*48;
    require(actual==std::vector<Word>({9007199254740993ULL*count+count*(count-1)/2,9007199254740993ULL,9007199254740993ULL+count-1,0x8000000000000000ULL,1,0}),"64-bit atomics and double bits");
    std::vector<long long> claimed; key64.copyToHost(claimed,1);
    require(claimed==std::vector<long long>({-9007199254740993LL}),"64-bit CAS preserves signed key bits");
    require(valid==std::vector<uint64_t>({UINT64_MAX,UINT64_MAX,4}),"bitmap atomic OR contention and exact sparse bit");
}

int main() try {
    auto runtime=std::make_shared<db::DeviceRuntime>();
    std::cout<<runtime->properties().name<<'\n';
    for (unsigned block:{1,3,31,48,127,128,255,256,1023}) {
        for (size_t count:{size_t(0),size_t(1),size_t(block-1),size_t(block),size_t(block+1),size_t(4097)}) {
            collective<long long>(runtime,block,count);
            collective<Word>(runtime,block,count);
            collective<double>(runtime,block,count);
        }
    }
    tiles(runtime); scan(runtime);
    for (int repeat=0;repeat<3;++repeat) { pht(runtime,false); pht(runtime,true); mht(runtime); atomics(runtime); }
    runtime->wait();
    require(runtime->residentDeviceBytes()==0 && runtime->residentPinnedBytes()==0,"all primitive resources released");
    std::cout<<"primitive oracle assertions: "<<checks<<" passed\n";
} catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
