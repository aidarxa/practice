#pragma once
/** @file @brief Prebuilt GPU byte materialization for a typed CPU Text result window. */
#include "core/pcuda_scan.h"
#include "core/text.h"
#include <string>
namespace db {
/** @brief Reusable, single-writer fetch workspace; no CPU evaluation of SQL operators. */
class TextFetch {
public:
    explicit TextFetch(std::shared_ptr<DeviceRuntime> runtime);
    /** @brief Gather only a checked row window; return owned strings, with NULL validity kept separately. */
    std::vector<std::string> read(const TextView* values, const uint64_t* validity,
                                  size_t source_rows, size_t offset, size_t count);
private:
    std::shared_ptr<DeviceRuntime> runtime_;
    ExclusiveScan scan_;
    DynamicDeviceBuffer<unsigned long long> lengths_, offsets_;
    DynamicDeviceBuffer<uint8_t> bytes_;
    DynamicDeviceBuffer<int> error_;
    void sizes(const TextView*, const uint64_t*, size_t offset, size_t count);
    void gather(const TextView*, size_t offset, size_t count);
};
}
