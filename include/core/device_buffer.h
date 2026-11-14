#pragma once
/** @file @brief Reusable PCUDA buffers with checked capacity and safe transfer lifetimes. */
#include "core/pcuda_runtime.h"
#include "core/memory_guard.h"
#include "core/size_expression.h"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace db {
/** @brief Typed high-water allocation, owned by a single stream's resource domain.
 * @details Growth does not preserve contents (result/scratch contract). A failed
 * growth keeps old storage alive; the budget includes both allocations until the
 * old one is drained and released. Large buffers do not receive 1.5x headroom.
 */
template<class T> class DynamicDeviceBuffer {
    static_assert(std::is_trivially_copyable<T>::value, "GPU storage must be trivially copyable");
public:
    explicit DynamicDeviceBuffer(std::shared_ptr<DeviceRuntime> runtime, size_t initial_capacity = 0)
        : runtime_(std::move(runtime)), storage_(runtime_, bytes(initial_capacity)) {}
    DynamicDeviceBuffer(const DynamicDeviceBuffer&) = delete;
    DynamicDeviceBuffer& operator=(const DynamicDeviceBuffer&) = delete;
    DynamicDeviceBuffer(DynamicDeviceBuffer&&) noexcept = default;
    DynamicDeviceBuffer& operator=(DynamicDeviceBuffer&&) noexcept = default;
    void ensureCapacity(size_t requested) {
        if (requested <= capacity()) return;
        RuntimeAllocation replacement(runtime_, bytes(nextCapacity(requested)));
        storage_ = std::move(replacement);
    }
    size_t nextCapacity(size_t requested) const { return deviceBufferCapacity(requested); }
    void zero() { runtime_->fillAsync(data(), 0, storage_.bytes()); }
    /** @brief Snapshot ordinary CPU input in pinned storage; wait before either owner can die. */
    void copyFromHost(const std::vector<T>& source, size_t offset = 0) {
        checkRange(offset, source.size());
        if (source.empty()) return;
        RuntimeAllocation staging(runtime_, bytes(source.size()), MemoryKind::Pinned);
        std::memcpy(staging.data(), source.data(), staging.bytes());
        runtime_->copyAsync(data() + offset, staging.data(), staging.bytes());
        runtime_->wait();
    }
    void copyToHost(std::vector<T>& host, size_t count) { copyRangeToHost(host, 0, count); }
    /** @brief Copy after preceding stream work; no unnecessary wait before enqueue. */
    void copyRangeToHost(std::vector<T>& host, size_t offset, size_t count) {
        checkRange(offset, count);
        host.resize(count);
        if (!count) return;
        RuntimeAllocation staging(runtime_, bytes(count), MemoryKind::Pinned);
        runtime_->copyAsync(staging.data(), data() + offset, staging.bytes());
        runtime_->wait();
        std::memcpy(host.data(), staging.data(), staging.bytes());
    }
    T* data() const { return static_cast<T*>(storage_.data()); }
    size_t capacity() const { return storage_.bytes() / sizeof(T); }
private:
    std::shared_ptr<DeviceRuntime> runtime_;
    RuntimeAllocation storage_;
    static size_t bytes(size_t count) {
        const auto result = checkedSizeMultiply(count, sizeof(T));
        if (result > std::numeric_limits<size_t>::max()) throw std::overflow_error("GPU buffer byte size overflow");
        return static_cast<size_t>(result);
    }
    void checkRange(size_t offset, size_t count) const {
        if (offset > capacity() || count > capacity() - offset) throw std::out_of_range("GPU buffer range exceeds capacity");
    }
};
}
