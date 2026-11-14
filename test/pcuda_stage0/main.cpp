#include "probe.hpp"
#include "runtime.hpp"

#include <iomanip>
#include <memory>

namespace {
using namespace stage0;
using ProbeModule = stage0::Module<ProbeLaunch>;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4 || !*argv[3])
            throw std::runtime_error("usage: probe seed.so transform.so DEVICE_NAME_SUBSTRING");
        const auto prop = selectDevice(argv[3]);
        constexpr unsigned block = 128;
        constexpr std::size_t capacity = 100003;
        if (prop.maxThreadsPerBlock < static_cast<int>(block) ||
            prop.maxThreadsDim[0] < static_cast<int>(block) ||
            prop.maxGridSize[0] < static_cast<int>(capacity / block + 1))
            throw std::runtime_error("device launch limits insufficient for probe");
        Stream stream;
        Buffer<std::uint32_t> input(stream.value, capacity, true);
        Buffer<std::uint32_t> output(stream.value, capacity, true);
        Buffer<std::uint32_t> device(stream.value, capacity, false);
        for (std::size_t i = 0; i < capacity; ++i) input.data[i] = static_cast<std::uint32_t>(i);

        std::cout << std::setprecision(9) << "{\"cycles\":[";
        for (unsigned cycle = 0; cycle < 3; ++cycle) {
            const auto loadStart = Clock::now();
            // Reverse load order while preserving logical dataflow.
            auto first = std::make_unique<ProbeModule>(argv[cycle % 2 ? 2 : 1], stream.value, "launch_probe");
            auto second = std::make_unique<ProbeModule>(argv[cycle % 2 ? 1 : 2], stream.value, "launch_probe");
            ProbeModule& seed = cycle % 2 ? *second : *first;
            ProbeModule& transform = cycle % 2 ? *first : *second;
            const double loadMs = elapsed(loadStart);
            if (seed.launch(nullptr, device.data, 1, block, 0) != pcudaErrorInvalidValue ||
                transform.launch(stream.value, device.data, 1, 0, 0) != pcudaErrorInvalidValue)
                throw std::runtime_error("invalid launch was not rejected");

            if (cycle) std::cout << ',';
            std::cout << "{\"load_ms\":" << loadMs << ",\"executions\":[";
            unsigned trial = 0;
            // Large first invocation, warm repetitions, then boundary fixtures.
            for (std::size_t count : {capacity, capacity, capacity, std::size_t{0},
                                     std::size_t{1}, std::size_t{127}, std::size_t{128}, std::size_t{129}}) {
                for (std::uint32_t value : {0u, 17u, std::numeric_limits<std::uint32_t>::max() - 3u}) {
                    const auto bytes = count * sizeof(std::uint32_t); // count <= checked capacity
                    const auto start = Clock::now();
                    if (count) {
                        check(pcudaMemsetAsync(device.data, 0, bytes, stream.value), "memset");
                        check(pcudaMemcpyAsync(device.data, input.data, bytes,
                                               pcudaMemcpyHostToDevice, stream.value), "copy input");
                    }
                    check(seed.launch(stream.value, device.data, count, block, value), "seed launch");
                    check(transform.launch(stream.value, device.data, count, block, 7), "transform launch");
                    if (count)
                        check(pcudaMemcpyAsync(output.data, device.data, bytes,
                                               pcudaMemcpyDeviceToHost, stream.value), "copy output");
                    // No wait between kernels, or before the asynchronous result copy.
                    check(pcudaStreamSynchronize(stream.value), "await result");
                    const double executionMs = elapsed(start);
                    for (std::size_t i = 0; i < count; ++i) {
                        const std::uint32_t expected = (input.data[i] + value) * std::uint32_t{3} + 7u;
                        if (output.data[i] != expected)
                            throw std::runtime_error("result mismatch at " + std::to_string(i));
                    }
                    if (trial++) std::cout << ',';
                    std::cout << "{\"count\":" << count << ",\"value\":" << value
                              << ",\"execute_ms\":" << executionMs << '}';
                }
            }
            // Exercise destructor synchronization with work still in flight.
            const auto bytes = capacity * sizeof(std::uint32_t);
            check(pcudaMemcpyAsync(device.data, input.data, bytes,
                                   pcudaMemcpyHostToDevice, stream.value), "copy before unload");
            check(seed.launch(stream.value, device.data, capacity, block, 29), "seed before unload");
            check(transform.launch(stream.value, device.data, capacity, block, 7), "transform before unload");
            check(pcudaMemcpyAsync(output.data, device.data, bytes,
                                   pcudaMemcpyDeviceToHost, stream.value), "copy before unload");
            // Both module destructors wait before dlclose, even in reverse load order.
            second.reset();
            first.reset();
            for (std::size_t i = 0; i < capacity; ++i) {
                const std::uint32_t expected = (input.data[i] + 29u) * std::uint32_t{3} + 7u;
                if (output.data[i] != expected)
                    throw std::runtime_error("pending-work unload mismatch");
            }
            std::cout << "],\"pending_work_unload_verified\":true}";
        }
        std::cout << "],\"verified\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
