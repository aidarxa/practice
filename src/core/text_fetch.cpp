#include "core/text_fetch.h"
#include <pcuda.hpp>
#include <climits>
#include <algorithm>

namespace db {
namespace {
KernelLaunchConfigV1 launchFor(size_t count) { return {{count/256+(count%256!=0),1,1},{256,1,1},0}; }
}
TextFetch::TextFetch(std::shared_ptr<DeviceRuntime> runtime)
    : runtime_(std::move(runtime)), scan_(runtime_), lengths_(runtime_,0), offsets_(runtime_,0), bytes_(runtime_,0), error_(runtime_,1) {}
void TextFetch::sizes(const TextView* values, const uint64_t* validity, size_t offset, size_t count) {
    const auto launch = launchFor(count); runtime_->validateLaunch(launch);
    auto* lengths = lengths_.data(); auto* error = error_.data();
    error_.zero();
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(launch.grid[0])), dim3(256), 0, runtime_->stream(), [=] {
        const uint64_t i = static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        if (i >= count) return;
        const auto row = offset+i;
        const bool valid = !validity || ((validity[row/64]>>(row%64))&1U);
        const auto value = values[row];
        if (valid && (value.bytes > INT32_MAX || (value.bytes && !value.data))) {
            atomicExch(error, 1); lengths[i] = 0; return;
        }
        lengths[i] = valid ? value.bytes : 0;
    }), "Text output lengths");
}
void TextFetch::gather(const TextView* values, size_t offset, size_t count) {
    const auto launch = launchFor(count); runtime_->validateLaunch(launch);
    auto* bytes = bytes_.data(); const auto* offsets = offsets_.data(); const auto* lengths = lengths_.data();
    checkPcudacall(pcudaParallelFor(dim3(static_cast<unsigned>(launch.grid[0])), dim3(256), 0, runtime_->stream(), [=] {
        const uint64_t i = static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        if (i >= count) return;
        const auto value = values[offset+i];
        for (uint64_t j=0;j<lengths[i];++j) bytes[offsets[i]+j] = value.data[j];
    }), "Text output bytes gather");
}
std::vector<std::string> TextFetch::read(const TextView* values, const uint64_t* validity,
                                        size_t rows, size_t offset, size_t count) {
    if (offset > rows || count > rows-offset || (count && !values)) throw std::out_of_range("Invalid Text result window");
    if (!count) return {};
    lengths_.ensureCapacity(count); offsets_.ensureCapacity(count);
    sizes(values, validity, offset, count);
    scan_.enqueue(lengths_.data(), offsets_.data(), count);
    const auto bytes = scan_.finish();
    if (bytes > SIZE_MAX) throw std::overflow_error("Text fetch byte extent exceeds size_t");
    std::vector<int> error; error_.copyToHost(error, 1);
    if (error.front()) throw std::runtime_error("Invalid Text view in device result");
    bytes_.ensureCapacity(std::max<size_t>(bytes,1));
    gather(values, offset, count);
    std::vector<uint8_t> host;
    std::vector<unsigned long long> starts;
    bytes_.copyToHost(host, bytes); offsets_.copyToHost(starts, count);
    std::vector<std::string> result(count);
    for (size_t i=0;i<count;++i) {
        const auto end = i+1<count ? starts[i+1] : bytes;
        if (end > starts[i]) result[i].assign(reinterpret_cast<const char*>(host.data()+starts[i]), end-starts[i]);
    }
    return result;
}
}
