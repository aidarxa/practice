#include "core/text_storage.h"
#include <algorithm>
#include <climits>

namespace db {
LoadedTextColumn uploadTextColumn(const std::shared_ptr<DeviceRuntime>& runtime, const HostTextColumn& source,
                                 bool wide, const std::vector<uint64_t>& validity) {
    if (source.offsets.empty() || source.offsets.front() || source.offsets.back() != source.bytes.size())
        throw std::invalid_argument("Invalid normalized Text offsets/bytes");
    if (!std::is_sorted(source.offsets.begin(), source.offsets.end())) throw std::invalid_argument("Non-monotonic Text offsets");
    const auto rows = source.offsets.size() - 1;
    if (!validity.empty() && validity.size() != rows / 64 + (rows % 64 != 0)) throw std::invalid_argument("Invalid Text bitmap extent");
    LoadedTextColumn column;
    column.byte_count = source.bytes.size();
    column.bytes = std::make_unique<DynamicDeviceBuffer<uint8_t>>(runtime, std::max<size_t>(source.bytes.size(), 1));
    column.bytes->copyFromHost(source.bytes);
    if (wide) {
        column.offsets64 = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, source.offsets.size());
        column.offsets64->copyFromHost(source.offsets);
    } else {
        if (column.byte_count > INT32_MAX) throw std::overflow_error("Text offsets do not fit Int32");
        std::vector<int32_t> offsets(source.offsets.begin(), source.offsets.end());
        column.offsets32 = std::make_unique<DynamicDeviceBuffer<int32_t>>(runtime, offsets.size());
        column.offsets32->copyFromHost(offsets);
    }
    if (!validity.empty()) {
        column.validity = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, validity.size());
        column.validity->copyFromHost(validity);
    }
    return column;
}
}
