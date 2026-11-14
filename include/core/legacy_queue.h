#pragma once
/** @file @brief Removable SYCL launch view over the single PCUDA-owned executor. */
#include "core/pcuda_runtime.h"
#include <memory>

namespace db {
/** @brief Only the implementation knows SYCL; the queue never owns another device/stream.
 * @details Generated legacy wrappers use USM and local accessors only, which the
 * installed runtime submits instantly on a dedicated executor. Their wrapper
 * checks this property; wait also drains any unexpectedly deferred submission.
 */
class LegacyQueue {
public:
    explicit LegacyQueue(std::shared_ptr<DeviceRuntime> runtime);
    ~LegacyQueue() noexcept;
    LegacyQueue(const LegacyQueue&) = delete;
    LegacyQueue& operator=(const LegacyQueue&) = delete;
    void* handle() const;
    void wait() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
