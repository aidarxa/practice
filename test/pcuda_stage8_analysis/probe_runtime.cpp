/** @file @brief HIP host API intervals; durations are not claimed to be GPU kernel times. */
#include "probe_log.h"
#include "probe_symbols.h"
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

namespace {
/** @brief HIP is a local backend dependency, not necessarily visible through RTLD_NEXT. */
template<class Function> Function original(const char* name) {
    return db::s2probe::original<Function>(name,CRYSTAL_HIP_RUNTIME);
}
template<class Function,class... Arguments>
auto observed(const char* kind,std::uint64_t bytes,Function function,Arguments... arguments) {
    db::s2probe::Span interval(kind,bytes);
    const auto status=function(arguments...);
    interval.finish(static_cast<int>(status));
    return status;
}
}
extern "C" hipError_t hipModuleLoadData(hipModule_t* module,const void* image) {
    static auto function=original<decltype(&hipModuleLoadData)>("hipModuleLoadData");
    return observed("hip_module_load",0,function,module,image);
}
extern "C" hipError_t hipModuleLoadDataEx(hipModule_t* module,const void* image,unsigned count,hipJitOption* options,void** values) {
    static auto function=original<decltype(&hipModuleLoadDataEx)>("hipModuleLoadDataEx");
    return observed("hip_module_load_ex",0,function,module,image,count,options,values);
}
extern "C" hiprtcResult hiprtcLinkComplete(hiprtcLinkState state,void** binary,size_t* size) {
    // LD_DEBUG=bindings confirms all four stock HIPRTC link entries bind to this HIP DSO.
    // Choosing the other libhiprtc implementation would mix incompatible opaque states.
    static auto function=original<decltype(&hiprtcLinkComplete)>("hiprtcLinkComplete");
    return observed("hiprtc_complete",0,function,state,binary,size);
}
extern "C" hipError_t hipMalloc(void** pointer,size_t size) {
    using Function=hipError_t(*)(void**,size_t);
    static auto function=original<Function>("hipMalloc");
    return observed("hip_malloc",size,function,pointer,size);
}
extern "C" hipError_t hipFree(void* pointer) {
    static auto function=original<decltype(&hipFree)>("hipFree");
    return observed("hip_free",0,function,pointer);
}
extern "C" hipError_t hipHostMalloc(void** pointer,size_t size,unsigned flags) {
    using Function=hipError_t(*)(void**,size_t,unsigned);
    static auto function=original<Function>("hipHostMalloc");
    return observed("hip_host_malloc",size,function,pointer,size,flags);
}
extern "C" hipError_t hipHostFree(void* pointer) {
    static auto function=original<decltype(&hipHostFree)>("hipHostFree");
    return observed("hip_host_free",0,function,pointer);
}
extern "C" hipError_t hipMemsetAsync(void* pointer,int value,size_t size,hipStream_t stream) {
    static auto function=original<decltype(&hipMemsetAsync)>("hipMemsetAsync");
    return observed("hip_memset_async",size,function,pointer,value,size,stream);
}
extern "C" hipError_t hipMemcpyAsync(void* destination,const void* source,size_t size,hipMemcpyKind kind,hipStream_t stream) {
    static auto function=original<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
    return observed("hip_memcpy_async",size,function,destination,source,size,kind,stream);
}
extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
    static auto function=original<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
    return observed("hip_stream_wait",0,function,stream);
}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* result,hipModule_t module,const char* name) {
    static auto function=original<decltype(&hipModuleGetFunction)>("hipModuleGetFunction");
    db::s2probe::Span interval("hip_module_function",0,name?name:"");
    const auto status=function(result,module,name);
    interval.finish(static_cast<int>(status));
    return status;
}
