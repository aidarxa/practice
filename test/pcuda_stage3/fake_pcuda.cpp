#include "fake_pcuda.h"
#include "../../src/core/pcuda_compat.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
std::vector<std::string> log;
std::map<void*, bool> memory;
std::vector<std::function<void()>> pending;
int backend = 0, platform = 0, device = 0;
bool allocation_failure = false, copy_failure = false, wait_failure = false, persistent_failure = false;
bool duplicate_gpu = false, fp64 = true, atomic64 = true;
bool null_allocation = false, empty_platform = false;
std::size_t stream_count = 0;
pcudaStream_t last_stream = nullptr;
pcudaError_t allocate(void** pointer, size_t bytes, bool pinned) {
    log.push_back(pinned ? "allocate-pinned" : "allocate-device");
    if (std::exchange(allocation_failure, false)) return pcudaErrorMemoryAllocation;
    if (std::exchange(null_allocation, false)) { *pointer = nullptr; return pcudaSuccess; }
    *pointer = std::malloc(bytes);
    if (!*pointer) return pcudaErrorMemoryAllocation;
    memory[*pointer] = pinned;
    return pcudaSuccess;
}
pcudaError_t release(void* pointer, bool pinned) {
    if (!pending.empty()) throw std::runtime_error("free before wait");
    if (memory.at(pointer) != pinned) throw std::runtime_error("wrong allocator release");
    log.push_back(pinned ? "free-pinned" : "free-device");
    memory.erase(pointer);
    std::free(pointer);
    return pcudaSuccess;
}
}
namespace fake_pcuda {
void reset() { log.clear(); }
void failAllocation() { allocation_failure = true; }
void nullAllocation() { null_allocation = true; }
void failCopy() { copy_failure = true; }
void failWait(bool persistent) { wait_failure = true; persistent_failure = persistent; }
void duplicateGpu(bool enabled) { duplicate_gpu = enabled; }
void emptyFirstPlatform(bool enabled) { empty_platform = enabled; }
void capabilities(bool has_fp64, bool has_atomic64) { fp64 = has_fp64; atomic64 = has_atomic64; }
const std::vector<std::string>& events() { return log; }
std::size_t allocations() { return memory.size(); }
}
namespace db::detail {
bool pcudaDeviceIsGpu(int b, int, int) { return b >= 2; }
std::string takePcudaruntimeErrors() { return {}; }
void waitPcudastream(pcudaStream_t stream) {
    if (!stream) throw std::runtime_error("default stream used");
    log.push_back("wait");
    if (wait_failure) {
        wait_failure = persistent_failure;
        throw std::runtime_error("injected wait failure");
    }
    for (const auto& operation : pending) operation();
    pending.clear();
}
}
extern "C" {
pcudaError_t pcudaGetBackendCount(int* count) { *count = duplicate_gpu ? 4 : 3; return pcudaSuccess; }
pcudaError_t pcudaSetBackend(int b) { backend = b; platform = 0; return b == 0 || (b == 2 && empty_platform) ? pcudaErrorNoDevice : pcudaSuccess; }
pcudaError_t pcudaGetPlatformCount(int* count) { *count = backend == 2 && empty_platform ? 2 : 1; return pcudaSuccess; }
pcudaError_t pcudaSetPlatform(int p) { platform = p; return backend == 0 || (backend == 2 && empty_platform && p == 0) ? pcudaErrorNoDevice : pcudaSuccess; }
pcudaError_t pcudaSetDevice(int d) { device = d; return pcudaSuccess; }
pcudaError_t pcudaGetDeviceCount(int* count) { *count = 1; return pcudaSuccess; }
pcudaError_t pcudaSetDeviceExt(int b, int p, int d) { backend = b; platform = p; device = d; return pcudaSuccess; }
pcudaError_t pcudaGetDeviceProperties(pcudaDeviceProp* properties, int) {
    *properties = {};
    std::strcpy(properties->name, backend >= 2 ? "Test GPU" : "Test CPU");
    properties->totalGlobalMem = 1ULL << 30;
    properties->sharedMemPerBlock = 32768;
    properties->maxThreadsPerBlock = 1024;
    std::fill_n(properties->maxThreadsDim, 3, 1024);
    std::fill_n(properties->maxGridSize, 3, INT32_MAX);
    properties->pcudaHasFp64 = fp64;
    properties->pcudaHasAtomic64 = atomic64;
    return pcudaSuccess;
}
pcudaError_t pcudaStreamCreate(pcudaStream_t* stream) {
    log.push_back("create-stream");
    *stream = reinterpret_cast<pcudaStream_t>(++stream_count);
    last_stream = *stream;
    return pcudaSuccess;
}
pcudaError_t pcudaStreamDestroy(pcudaStream_t) { log.push_back("destroy-stream"); return pcudaSuccess; }
pcudaError_t pcudaAllocateDevice(void** pointer, size_t bytes) { return allocate(pointer, bytes, false); }
pcudaError_t pcudaAllocateHost(void** pointer, size_t bytes) { return allocate(pointer, bytes, true); }
pcudaError_t pcudaFree(void* pointer) { return release(pointer, false); }
pcudaError_t pcudaFreeHost(void* pointer) { return release(pointer, true); }
pcudaError_t pcudaMemcpyAsync(void* destination, const void* source, size_t bytes, pcudaMemcpyKind, pcudaStream_t stream) {
    if (!stream) throw std::runtime_error("default stream copy");
    log.push_back("copy");
    if (std::exchange(copy_failure, false)) return pcudaErrorInvalidValue;
    pending.push_back([=] { std::memcpy(destination, source, bytes); });
    return pcudaSuccess;
}
pcudaError_t pcudaMemsetAsync(void* destination, int value, size_t bytes, pcudaStream_t stream) {
    if (!stream) throw std::runtime_error("default stream fill");
    log.push_back("fill");
    pending.push_back([=] { std::memset(destination, value, bytes); });
    return pcudaSuccess;
}
pcudaError_t pcudaPeekAtLastError() { return pcudaSuccess; }
const char* pcudaGetErrorString(pcudaError_t) { return "injected API failure"; }
}
