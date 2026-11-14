/** @file @brief Hardware checks of actual PCUDA services, transfer ownership and error propagation. */
#include "core/device_buffer.h"
#include "core/pcuda_services.h"
#include <hipSYCL/runtime/application.hpp>
#include <hipSYCL/runtime/async_errors.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void compact(const std::shared_ptr<db::DeviceRuntime>& runtime, size_t rows) {
    constexpr size_t width = 3;
    std::vector<unsigned long long> input(rows * width), expected;
    for (size_t row = 0; row < rows; ++row) {
        if (row % 3 == 0) continue;
        input[row * width] = 9007199254740993ULL + row;
        input[row * width + 1] = row;
        expected.push_back(input[row * width]);
    }
    db::DynamicDeviceBuffer<unsigned long long> source(runtime, input.size());
    db::DynamicDeviceBuffer<unsigned long long> target(runtime, input.size());
    db::DynamicDeviceBuffer<unsigned long long> count(runtime, 1);
    source.copyFromHost(input);
    for (int repeat = 0; repeat < 3; ++repeat) {
        db::compactSparseRows(*runtime, source.data(), target.data(), count.data(), rows, width);
        std::vector<unsigned long long> host_count, host;
        count.copyToHost(host_count, 1); // no wait between enqueue and D2H
        require(host_count[0] == expected.size(), "sparse count");
        target.copyToHost(host, expected.size() * width);
        std::vector<unsigned long long> keys;
        for (size_t row = 0; row < expected.size(); ++row) {
            keys.push_back(host[row * width]);
            const auto original = host[row * width + 1];
            require(original < rows && original % 3 != 0, "sparse original row");
            require(host[row * width] == input[original * width] && host[row * width + 2] == 0, "whole tuple copy");
        }
        std::sort(keys.begin(), keys.end());
        require(keys == expected, "sparse exact Int64-compatible bits and multiplicity");
    }
}
void asynchronousError(const std::shared_ptr<db::DeviceRuntime>& runtime) {
    hipsycl::rt::register_error(__acpp_here(), hipsycl::rt::error_info{"stage3 injected non-PCUDA async error"});
    bool rejected = false;
    try { runtime->wait(); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find("stage3 injected") != std::string::npos; }
    require(rejected, "compatibility wait must not swallow non-PCUDA async error");
    runtime->wait(); // known error consumed; successful completion allows safe teardown
}
}
int main() try {
    auto runtime = std::make_shared<db::DeviceRuntime>();
    std::cout << runtime->properties().name << ", FP64=" << runtime->properties().pcudaHasFp64
              << ", atomic64=" << runtime->properties().pcudaHasAtomic64 << '\n';
    for (size_t rows : {0,1,2,127,128,129,4097}) compact(runtime, rows);
    asynchronousError(runtime);
    require(runtime->residentDeviceBytes() == 0 && runtime->residentPinnedBytes() == 0, "all hardware buffers released");
    std::cout << "GPU runtime: 21 sparse fixtures, exact values, repeats and injected async error passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
