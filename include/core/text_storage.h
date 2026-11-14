#pragma once
/** @file @brief Compact Text allocation owners; lifetime is independent of borrowed device views. */
#include "core/device_buffer.h"
#include "storage/text_column.h"
namespace db {
struct LoadedTextColumn {
    std::unique_ptr<DynamicDeviceBuffer<uint8_t>> bytes;
    std::unique_ptr<DynamicDeviceBuffer<int32_t>> offsets32;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> offsets64;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> validity;
    uint64_t byte_count = 0;
};
/** @brief Upload a normalized column and its optional bitmap, checking offset narrowing. */
LoadedTextColumn uploadTextColumn(const std::shared_ptr<DeviceRuntime>& runtime, const HostTextColumn& column,
                                 bool wide_offsets, const std::vector<uint64_t>& validity = {});
}
