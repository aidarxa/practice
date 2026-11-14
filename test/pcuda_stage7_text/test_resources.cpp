/** @file @brief Native PCUDA Text offset, fetch-window, masked-payload and recovery contracts. */
#include "core/text_fetch.h"
#include "core/text_storage.h"
#include <climits>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition) { if(!condition) throw std::runtime_error("Text resource contract mismatch"); }
template<class F> void rejects(F operation) {
    bool rejected=false;
    try { operation(); } catch(const std::exception&) { rejected=true; }
    require(rejected);
}
void fetchWindows(const std::shared_ptr<db::DeviceRuntime>& runtime) {
    db::DynamicDeviceBuffer<uint8_t> bytes(runtime,3);
    bytes.copyFromHost({'a','b','c'});
    db::DynamicDeviceBuffer<db::TextView> views(runtime,4);
    db::DynamicDeviceBuffer<uint64_t> validity(runtime,1);
    // Invalid masked payload must not be read/validated or included in a prefix sum.
    const std::vector<db::TextView> input={{nullptr,0},{reinterpret_cast<const uint8_t*>(1),UINT64_MAX},{bytes.data(),1},{bytes.data()+1,2}};
    views.copyFromHost(input); validity.copyFromHost({13});
    db::TextFetch fetch(runtime);
    require(fetch.read(views.data(),validity.data(),4,0,4)==std::vector<std::string>({"","","a","bc"}));
    require(fetch.read(views.data(),validity.data(),4,1,3)==std::vector<std::string>({"","a","bc"}));
    require(fetch.read(nullptr,nullptr,0,0,0).empty());
    rejects([&] { fetch.read(views.data(),validity.data(),4,3,2); });
    rejects([&] { fetch.read(nullptr,nullptr,1,0,1); });
    rejects([&] { fetch.read(views.data(),nullptr,4,1,1); });
    auto malformed=input; malformed[0]={nullptr,1}; views.copyFromHost(malformed);
    rejects([&] { fetch.read(views.data(),validity.data(),4,0,1); });
    views.copyFromHost(input);
    require(fetch.read(views.data(),validity.data(),4,2,2)==std::vector<std::string>({"a","bc"}));
}
void uploads(const std::shared_ptr<db::DeviceRuntime>& runtime) {
    db::HostTextColumn input{{'a','b','c'},{0,1,1,3}};
    auto narrow=db::uploadTextColumn(runtime,input,false,{5});
    auto wide=db::uploadTextColumn(runtime,input,true,{5});
    std::vector<int32_t> offsets32; std::vector<uint64_t> offsets64;
    narrow.offsets32->copyToHost(offsets32,4); wide.offsets64->copyToHost(offsets64,4);
    require(offsets32==std::vector<int32_t>({0,1,1,3}) && offsets64==input.offsets);
    require(!narrow.offsets64 && !wide.offsets32);
    rejects([&] { db::uploadTextColumn(runtime,{{'a'},{1,1}},false); });
    rejects([&] { db::uploadTextColumn(runtime,{{'a'},{0,2}},false); });
    rejects([&] { db::uploadTextColumn(runtime,{{'a'},{0,1,0,1}},false); });
    rejects([&] { db::uploadTextColumn(runtime,input,false,{0,0}); });
    const db::CrystalConfig old{}; // this fixture creates its runtime with the same explicit default policy
    auto guard=old; guard.memory_guard_reserve_bytes=runtime->properties().totalGlobalMem;
    auto guarded=std::make_shared<db::DeviceRuntime>(guard);
    rejects([&] { db::uploadTextColumn(guarded,input,false); });
    auto recovered=db::uploadTextColumn(runtime,input,false);
    require(recovered.byte_count==3);
}
}
int main() {
    try {
        auto runtime=std::make_shared<db::DeviceRuntime>();
        fetchWindows(runtime); uploads(runtime);
        runtime->wait();
        std::cout<<"Native Text resources, narrow/wide offsets, windows, masked payload, memory guard and recovery passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
