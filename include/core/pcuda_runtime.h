#pragma once
/** @file
 * @brief PCUDA resource domain: one device, one explicit stream and one byte ledger.
 * @details A session is single-writer. Buffers share ownership of its runtime, not
 * a global allocator/stream. No SQL, SYCL object or device helper enters this API.
 */
#include <AdaptiveCpp/pcuda/pcuda_runtime.hpp>
#include "core/config.h"
#include "core/module_abi.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace db {
/** @brief Runtime enumeration coordinates, never assumed to have fixed indices. */
struct DeviceAddress { int backend; int platform; int device; };
/** @brief Both kinds count against the conservative session budget (also on integrated GPUs). */
enum class MemoryKind { Device, Pinned };
/** @brief Translate an API failure into a diagnostic without leaving a sticky known error. */
void checkPcudacall(pcudaError_t status, const char* operation);

class RuntimeAllocation;
/** @brief Own an explicit ordered stream; allocations keep it alive until their safe release. */
class DeviceRuntime {
public:
    explicit DeviceRuntime(const CrystalConfig& config = {});
    ~DeviceRuntime() noexcept;
    DeviceRuntime(const DeviceRuntime&) = delete;
    DeviceRuntime& operator=(const DeviceRuntime&) = delete;
    pcudaStream_t stream() const { return stream_; }
    const DeviceAddress& address() const { return address_; }
    const pcudaDeviceProp& properties() const { return properties_; }
    uint64_t residentDeviceBytes() const { return device_bytes_; }
    uint64_t residentPinnedBytes() const { return pinned_bytes_; }
    /** @brief Validate new policy, but do not evict retained high-water allocations. */
    void setMemoryPolicy(const CrystalConfig& config);
    /** @brief Select this domain before allocation/free; PCUDA device selection is thread-local. */
    void activate() const;
    /** @brief Borrow transfer pointers until wait; kernels must never dereference CPU-only pointers. */
    void copyAsync(void* destination, const void* source, size_t bytes);
    void fillAsync(void* destination, uint8_t value, size_t bytes);
    /** @brief Check native completion and asynchronous errors at a CPU/resource boundary. */
    void wait() const;
    /** @brief Terminate rather than release code/memory after unconfirmed completion. */
    void waitNoThrow() const noexcept;
    void requireCapabilities(bool fp64, bool atomic64, uint64_t shared_bytes) const;
    void validateLaunch(const KernelLaunchConfigV1& launch) const;
private:
    friend class RuntimeAllocation;
    DeviceAddress address_{};
    pcudaDeviceProp properties_{};
    pcudaStream_t stream_ = nullptr;
    CrystalConfig policy_;
    uint64_t device_bytes_ = 0;
    uint64_t pinned_bytes_ = 0;
    void* allocate(size_t bytes, MemoryKind kind);
    void release(void* pointer, size_t bytes, MemoryKind kind) noexcept;
};

/** @brief Move-only allocation; construction and destruction update the same resident ledger. */
class RuntimeAllocation {
public:
    RuntimeAllocation() = default;
    RuntimeAllocation(std::shared_ptr<DeviceRuntime> runtime, size_t bytes, MemoryKind kind = MemoryKind::Device);
    ~RuntimeAllocation() noexcept;
    RuntimeAllocation(const RuntimeAllocation&) = delete;
    RuntimeAllocation& operator=(const RuntimeAllocation&) = delete;
    RuntimeAllocation(RuntimeAllocation&& other) noexcept;
    RuntimeAllocation& operator=(RuntimeAllocation&& other) noexcept;
    void* data() const { return data_; }
    size_t bytes() const { return bytes_; }
private:
    std::shared_ptr<DeviceRuntime> runtime_;
    void* data_ = nullptr;
    size_t bytes_ = 0;
    MemoryKind kind_ = MemoryKind::Device;
    void swap(RuntimeAllocation& other) noexcept;
};
}
