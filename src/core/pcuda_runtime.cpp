#include "core/pcuda_runtime.h"
#include "core/memory_guard.h"
#include "core/size_expression.h"
#include "pcuda_compat.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace db {
namespace {
std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}
bool present(pcudaError_t status) {
    if (status == pcudaErrorNoDevice) return false;
    checkPcudacall(status, "enumerate PCUDA topology");
    return true;
}
void appendPlatformGpus(int backend, int platform, const std::string& name, std::vector<DeviceAddress>& devices) {
    if (!present(pcudaSetPlatform(platform))) return;
    int count = 0;
    if (!present(pcudaGetDeviceCount(&count))) return;
    for (int device = 0; device < count; ++device) {
        if (!detail::pcudaDeviceIsGpu(backend, platform, device)) continue;
        pcudaDeviceProp properties{};
        checkPcudacall(pcudaGetDeviceProperties(&properties, device), "read device properties");
        if (lowercase(properties.name).find(name) != std::string::npos) devices.push_back({backend,platform,device});
    }
}
void appendBackendGpus(int backend, const std::string& name, std::vector<DeviceAddress>& devices) {
    // NoDevice can mean only that platform 0 is empty; later platforms still exist.
    present(pcudaSetBackend(backend));
    int count = 0;
    if (!present(pcudaGetPlatformCount(&count))) return;
    for (int platform = 0; platform < count; ++platform) appendPlatformGpus(backend, platform, name, devices);
}
std::vector<DeviceAddress> enumerateGpus(const std::string& name) {
    std::vector<DeviceAddress> devices;
    int count = 0;
    if (!present(pcudaGetBackendCount(&count))) return devices;
    for (int backend = 0; backend < count; ++backend) appendBackendGpus(backend, lowercase(name), devices);
    return devices;
}
DeviceAddress selectGpu(const std::string& name) {
    const auto devices = enumerateGpus(name);
    if (devices.empty()) throw std::runtime_error("No GPU matches device_name: " + name);
    if (!name.empty() && devices.size() != 1) throw std::runtime_error("Ambiguous GPU device_name: " + name);
    return devices.front();
}
void fatalCleanup(const char* message) noexcept {
    std::fprintf(stderr, "Unsafe PCUDA resource teardown: %s\n", message);
    std::abort();
}
}
void checkPcudacall(pcudaError_t status, const char* operation) {
    if (status == pcudaSuccess) return;
    const auto pending = detail::takePcudaruntimeErrors();
    throw std::runtime_error(std::string(operation) + ": " + pcudaGetErrorString(status) + " " + pending);
}
DeviceRuntime::DeviceRuntime(const CrystalConfig& config) : policy_(config) {
    deviceMemoryBudget(0, policy_);
    address_ = selectGpu(config.device_name);
    activate();
    checkPcudacall(pcudaGetDeviceProperties(&properties_, address_.device), "read selected GPU properties");
    checkPcudacall(pcudaStreamCreate(&stream_), "create explicit PCUDA stream");
    if (!stream_) throw std::runtime_error("PCUDA returned a null stream");
}
DeviceRuntime::~DeviceRuntime() noexcept {
    waitNoThrow();
    try { activate(); checkPcudacall(pcudaStreamDestroy(stream_), "destroy PCUDA stream"); }
    catch (const std::exception& error) { fatalCleanup(error.what()); }
}
void DeviceRuntime::activate() const {
    present(pcudaSetBackend(address_.backend));
    checkPcudacall(pcudaSetPlatform(address_.platform), "activate PCUDA platform");
    checkPcudacall(pcudaSetDevice(address_.device), "activate PCUDA device");
}
void DeviceRuntime::setMemoryPolicy(const CrystalConfig& config) {
    checkDeviceMemoryRequest(properties_.totalGlobalMem, checkedSizeAdd(device_bytes_, pinned_bytes_), 0, config);
    policy_ = config;
}
void DeviceRuntime::wait() const { detail::waitPcudastream(stream_); }
void DeviceRuntime::waitNoThrow() const noexcept {
    try { wait(); }
    catch (const std::exception& error) { fatalCleanup(error.what()); }
    catch (...) { fatalCleanup("unknown completion error"); }
}
void DeviceRuntime::copyAsync(void* destination, const void* source, size_t bytes) {
    if (!bytes) return;
    if (!destination || !source) throw std::runtime_error("null PCUDA copy pointer");
    checkPcudacall(pcudaMemcpyAsync(destination, source, bytes, pcudaMemcpyDefault, stream_), "PCUDA copy");
}
void DeviceRuntime::fillAsync(void* destination, uint8_t value, size_t bytes) {
    if (!bytes) return;
    if (!destination) throw std::runtime_error("null PCUDA fill pointer");
    checkPcudacall(pcudaMemsetAsync(destination, value, bytes, stream_), "PCUDA memset");
    checkPcudacall(pcudaPeekAtLastError(), "PCUDA memset submission");
}
void* DeviceRuntime::allocate(size_t bytes, MemoryKind kind) {
    checkDeviceMemoryRequest(properties_.totalGlobalMem, checkedSizeAdd(device_bytes_, pinned_bytes_), bytes, policy_);
    if (!bytes) return nullptr;
    activate();
    void* pointer = nullptr;
    const auto status = kind == MemoryKind::Device ? pcudaMalloc(&pointer, bytes) : pcudaMallocHost(&pointer, bytes);
    checkPcudacall(status, "allocate PCUDA memory");
    if (!pointer) throw std::runtime_error("allocate PCUDA memory returned null");
    auto& resident = kind == MemoryKind::Device ? device_bytes_ : pinned_bytes_;
    resident += bytes; // peak sum checked above, including old storage during growth
    return pointer;
}
void DeviceRuntime::release(void* pointer, size_t bytes, MemoryKind kind) noexcept {
    if (!pointer) return;
    waitNoThrow();
    try {
        activate();
        checkPcudacall(kind == MemoryKind::Device ? pcudaFree(pointer) : pcudaFreeHost(pointer), "free PCUDA memory");
        auto& resident = kind == MemoryKind::Device ? device_bytes_ : pinned_bytes_;
        resident -= bytes;
    } catch (const std::exception& error) { fatalCleanup(error.what()); }
}
void DeviceRuntime::requireCapabilities(bool fp64, bool atomic64, uint64_t shared_bytes) const {
    if (fp64 && !properties_.pcudaHasFp64) throw std::runtime_error("GPU lacks required FP64");
    if (atomic64 && !properties_.pcudaHasAtomic64) throw std::runtime_error("GPU lacks required atomic64");
    if (shared_bytes > properties_.sharedMemPerBlock) throw std::runtime_error("GPU shared memory limit exceeded");
}
void DeviceRuntime::validateLaunch(const KernelLaunchConfigV1& launch) const {
    uint64_t threads = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!launch.block[axis] || launch.block[axis] > static_cast<uint64_t>(properties_.maxThreadsDim[axis]))
            throw std::runtime_error("GPU block dimension limit exceeded");
        if (!launch.grid[axis] || launch.grid[axis] > static_cast<uint64_t>(properties_.maxGridSize[axis]))
            throw std::runtime_error("GPU grid dimension limit exceeded");
        threads = checkedSizeMultiply(threads, launch.block[axis]);
    }
    if (threads > static_cast<uint64_t>(properties_.maxThreadsPerBlock)) throw std::runtime_error("GPU block volume limit exceeded");
    requireCapabilities(false, false, launch.dynamic_shared_bytes);
}
RuntimeAllocation::RuntimeAllocation(std::shared_ptr<DeviceRuntime> runtime, size_t bytes, MemoryKind kind)
    : runtime_(std::move(runtime)), bytes_(bytes), kind_(kind) {
    if (!runtime_) throw std::runtime_error("allocation requires a PCUDA runtime");
    data_ = runtime_->allocate(bytes, kind);
}
RuntimeAllocation::~RuntimeAllocation() noexcept { if (runtime_) runtime_->release(data_, bytes_, kind_); }
void RuntimeAllocation::swap(RuntimeAllocation& other) noexcept {
    runtime_.swap(other.runtime_);
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
    std::swap(kind_, other.kind_);
}
RuntimeAllocation::RuntimeAllocation(RuntimeAllocation&& other) noexcept { swap(other); }
RuntimeAllocation& RuntimeAllocation::operator=(RuntimeAllocation&& other) noexcept {
    RuntimeAllocation replacement(std::move(other));
    swap(replacement);
    return *this;
}
}
