/** @file @brief Native chrono timing with independent oracles strictly outside measured intervals. */
#include "benchmark_api.h"
#include "core/device_buffer.h"
#ifdef STAGE4_BASELINE
#include "core/legacy_queue.h"
#endif
#include <chrono>
#include <iostream>
#include <numeric>

using Word=unsigned long long;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
static void check(unsigned op,const std::vector<int>& input,const std::vector<Word>& actual) {
    std::vector<Word> expected(actual.size());
    Word prefix=0;
    for (size_t i=0;i<input.size();++i) {
        if (op==0) expected[i]=input[i]>7?input[i]:0;
        else if (op==1) expected[i]=input[i]%3?input[i]*3:UINT64_MAX;
        else if (op==2) expected[i/128]+=input[i];
        else { if (i%128==0) prefix=0; expected[i]=prefix; prefix+=input[i]; }
    }
    if (actual!=expected) throw std::runtime_error("primitive benchmark CPU oracle mismatch");
}
int main(int argc,char** argv) try {
    const unsigned rotation=argc>1?std::stoul(argv[1])%4:0;
    const bool verify=argc>2 && std::string(argv[2])=="verify";
    const auto process_start=Clock::now();
    auto runtime=std::make_shared<db::DeviceRuntime>();
    runtime->requireCapabilities(false,false,128*sizeof(Word));
    runtime->validateLaunch({{8193,1,1},{128,1,1},0});
    void* legacy_handle=nullptr;
#ifdef STAGE4_BASELINE
    db::LegacyQueue legacy(runtime); legacy_handle=legacy.handle();
#endif
    const auto startup_end=Clock::now();
    constexpr size_t count=1048589,keys=65536;
    std::vector<int> host(count),host_keys(keys);
    for (size_t i=0;i<count;++i) host[i]=i%keys+1;
    std::iota(host_keys.begin(),host_keys.end(),1);
    db::DynamicDeviceBuffer<int> input(runtime,count),key_input(runtime,keys),table(runtime);
    db::DynamicDeviceBuffer<unsigned> occupied(runtime);
    db::DynamicDeviceBuffer<Word> output(runtime);
    input.copyFromHost(host); key_input.copyFromHost(host_keys);
    const auto data_ready=Clock::now();
    std::cout<<"{\"startup_ms\":"<<ms(process_start,startup_end)<<",\"data_load_ms\":"<<ms(startup_end,data_ready)<<",\"count\":"<<count<<"}\n";
    for (unsigned order=0;order<4;++order) {
        const auto op=(order+rotation)%4;
        for (unsigned repeat=0;repeat<9;++repeat) {
            const auto begin=Clock::now();
            const auto rows=op==2?count/128+(count%128!=0):count;
            output.ensureCapacity(rows);
            if (op==1) {
                table.ensureCapacity(keys*2); table.zero();
#ifndef STAGE4_BASELINE
                occupied.ensureCapacity(keys); occupied.zero();
#endif
            }
            const BenchCall call{runtime->stream(),legacy_handle,input.data(),key_input.data(),table.data(),occupied.data(),output.data(),count,keys};
            benchmarkLaunch(op,call);
            runtime->wait(); // timing boundary, not an extra wait after each constituent kernel
            const auto executed=Clock::now();
            std::vector<Word> actual; output.copyToHost(actual,rows);
            const auto ready=Clock::now();
            std::cout<<"{\"op\":"<<op<<",\"repeat\":"<<repeat<<",\"execute_ms\":"<<ms(begin,executed)<<",\"fetch_ms\":"<<ms(executed,ready)<<",\"summary_ms\":"<<ms(begin,ready)<<"}\n";
            if (verify) check(op,host,actual);
        }
    }
} catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
