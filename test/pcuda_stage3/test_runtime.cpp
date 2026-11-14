/** @file
 * @brief Test actual runtime ownership against a deferred, failure-injecting API.
 * @details No assert/NDEBUG dependency and no GPU SDK implementation is linked.
 * Expectations describe observable memory, ordering and allocation accounting.
 */
#include "core/device_buffer.h"
#include "core/memory_guard.h"
#include "fake_pcuda.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F function, const std::string& text) {
    try { function(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(text) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected rejection: " + text);
}
db::CrystalConfig policy() {
    db::CrystalConfig config;
    config.memory_guard_reserve_bytes = 0;
    config.memory_guard_reserve_fraction = 0;
    return config;
}
void selection() {
    auto runtime = std::make_shared<db::DeviceRuntime>(policy());
    require(runtime->address().backend == 2, "enumerate empty/CPU backends before GPU");
    require(runtime->stream() != nullptr, "explicit nondefault stream");
    require(runtime->properties().pcudaHasFp64, "properties retained");
    db::CrystalConfig config = policy();
    config.device_name = "missing";
    rejects([&] { db::DeviceRuntime absent(config); }, "No GPU");
    fake_pcuda::duplicateGpu(true);
    config.device_name = "test gpu";
    rejects([&] { db::DeviceRuntime ambiguous(config); }, "Ambiguous");
    fake_pcuda::duplicateGpu(false);
    config.device_name = "TEST GPU";
    db::DeviceRuntime named(config);
    require(named.address().device == 0, "case-insensitive device selection");
    fake_pcuda::emptyFirstPlatform(true);
    {
        db::DeviceRuntime later(config);
        require(later.address().platform == 1, "empty platform 0 must not hide later GPU platforms");
    }
    fake_pcuda::emptyFirstPlatform(false);
}
void memoryPolicy() {
    auto config = policy();
    config.memory_guard_reserve_fraction = 0.25;
    require(db::deviceMemoryBudget(1024, config) == 768, "fractional reserve");
    config.memory_guard_reserve_bytes = 800;
    require(db::deviceMemoryBudget(1024, config) == 224, "absolute reserve");
    db::checkDeviceMemoryRequest(1024, 200, 24, config);
    rejects([&] { db::checkDeviceMemoryRequest(1024, 200, 25, config); }, "Insufficient GPU memory");
    config.memory_guard_reserve_bytes = 1024;
    require(db::deviceMemoryBudget(1024, config) == 0, "reserve cannot become a half-memory fallback");
    config.memory_guard_enabled = false;
    db::checkDeviceMemoryRequest(1024, 1024, 1024, config);
    rejects([&] { db::checkDeviceMemoryRequest(1024, SIZE_MAX, 1, config); }, "overflow");
    config.memory_guard_reserve_fraction = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { db::deviceMemoryBudget(1024, config); }, "invalid memory guard");
    config.memory_guard_reserve_fraction = -0.1;
    rejects([&] { db::deviceMemoryBudget(1024, config); }, "invalid memory guard");
}
void buffers() {
    auto runtime = std::make_shared<db::DeviceRuntime>(policy());
    db::DynamicDeviceBuffer<uint64_t> buffer(runtime, 4);
    buffer.copyFromHost(std::vector<uint64_t>{11,22,33,44});
    std::vector<uint64_t> host;
    buffer.copyRangeToHost(host, 1, 2);
    require(host == std::vector<uint64_t>({22,33}), "copy range values");
    auto* original = buffer.data();
    buffer.ensureCapacity(3);
    buffer.ensureCapacity(4);
    require(buffer.data() == original && buffer.capacity() == 4, "high-water reuse");
    buffer.ensureCapacity(5);
    require(buffer.capacity() == 7, "small growth policy");
    require(runtime->residentDeviceBytes() == 7 * sizeof(uint64_t), "resident capacity not requested size");
    buffer.zero();
    buffer.copyToHost(host, 7);
    require(host == std::vector<uint64_t>(7, 0), "ordered zero and download");
    rejects([&] { buffer.copyRangeToHost(host, 7, 1); }, "range");
    rejects([&] { buffer.copyFromHost(std::vector<uint64_t>(8)); }, "range");
    rejects([&] { buffer.ensureCapacity(SIZE_MAX); }, "overflow");
    require(buffer.capacity() == 7, "overflow leaves old storage alive");
    require(buffer.nextCapacity((1ULL << 20) + 1) == (1ULL << 20) + 1, "large buffers have exact capacity");
    db::DynamicDeviceBuffer<uint64_t> moved(std::move(buffer));
    require(!buffer.data() && moved.capacity() == 7, "move ownership");
    db::DynamicDeviceBuffer<uint64_t> target(runtime, 1);
    target = std::move(moved);
    require(!moved.data() && target.capacity() == 7, "move assignment releases target");
    require(runtime->residentDeviceBytes() == 56 && runtime->residentPinnedBytes() == 0, "no stale accounting");
}
template<class T> void typedStorage() {
    auto runtime = std::make_shared<db::DeviceRuntime>(policy());
    db::DynamicDeviceBuffer<T> buffer(runtime, 3);
    buffer.copyFromHost(std::vector<T>{T(1),T(2),T(3)});
    std::vector<T> host;
    buffer.copyToHost(host, 3);
    require(host == std::vector<T>({T(1),T(2),T(3)}), "typed round trip");
    buffer.ensureCapacity(5);
    buffer.zero();
    buffer.copyToHost(host, 5);
    require(host == std::vector<T>(5, T(0)), "typed growth and zero");
}
void failures() {
    auto runtime = std::make_shared<db::DeviceRuntime>(policy());
    db::DynamicDeviceBuffer<int> buffer(runtime, 4);
    fake_pcuda::nullAllocation();
    rejects([&] { buffer.ensureCapacity(8); }, "returned null");
    fake_pcuda::failAllocation();
    rejects([&] { buffer.ensureCapacity(8); }, "allocate");
    require(buffer.capacity() == 4 && runtime->residentDeviceBytes() == 16, "failed growth is transactional");
    fake_pcuda::failCopy();
    rejects([&] { buffer.copyFromHost(std::vector<int>{1,2}); }, "copy");
    require(runtime->residentPinnedBytes() == 0, "upload failure releases staging");
    fake_pcuda::failWait();
    rejects([&] { buffer.copyFromHost(std::vector<int>{5,6}); }, "injected wait");
    std::vector<int> host;
    buffer.copyToHost(host, 2);
    require(host == std::vector<int>({5,6}), "error cleanup drains pending host transfer");
    auto config = policy();
    config.memory_guard_reserve_bytes = runtime->properties().totalGlobalMem - 16;
    runtime->setMemoryPolicy(config);
    rejects([&] { db::RuntimeAllocation pinned(runtime, 1, db::MemoryKind::Pinned); }, "Insufficient GPU memory");
    runtime->setMemoryPolicy(policy());
    runtime->copyAsync(nullptr, nullptr, 0);
    runtime->fillAsync(nullptr, 0, 0);
    rejects([&] { runtime->copyAsync(nullptr, buffer.data(), 1); }, "null");
    rejects([&] { runtime->fillAsync(nullptr, 0, 1); }, "null");
}
void capabilities() {
    fake_pcuda::capabilities(false, false);
    auto runtime = std::make_shared<db::DeviceRuntime>(policy());
    rejects([&] { runtime->requireCapabilities(true, false, 0); }, "FP64");
    rejects([&] { runtime->requireCapabilities(false, true, 0); }, "atomic64");
    rejects([&] { runtime->requireCapabilities(false, false, SIZE_MAX); }, "shared");
    runtime->requireCapabilities(false, false, 0);
    KernelLaunchConfigV1 launch{{1,1,1},{48,1,1},0};
    runtime->validateLaunch(launch);
    launch.block[0] = 1025;
    rejects([&] { runtime->validateLaunch(launch); }, "block");
    launch.block[0] = 0;
    rejects([&] { runtime->validateLaunch(launch); }, "block");
    launch.block[0] = 1024; launch.block[1] = 2;
    rejects([&] { runtime->validateLaunch(launch); }, "block");
    launch.block[0] = 1; launch.block[1] = 1; launch.grid[0] = UINT64_MAX;
    rejects([&] { runtime->validateLaunch(launch); }, "grid");
    fake_pcuda::capabilities(true, true);
}
void lifetime() {
    std::weak_ptr<db::DeviceRuntime> owner;
    {
        auto runtime = std::make_shared<db::DeviceRuntime>(policy());
        owner = runtime;
        db::DynamicDeviceBuffer<int> buffer(runtime, 1);
        runtime.reset();
        require(!owner.expired(), "buffer keeps stream alive");
        buffer.zero(); // destruction must finish this operation before free
    }
    require(owner.expired() && fake_pcuda::allocations() == 0, "all owners released");
    const auto& events = fake_pcuda::events();
    auto free = std::find(events.rbegin(), events.rend(), "free-device");
    require(free != events.rend() && *(free + 1) == "wait", "wait precedes free");
    require(events.back() == "destroy-stream", "stream destroyed last");
    const auto child = fork();
    if (child == 0) {
        auto runtime = std::make_shared<db::DeviceRuntime>(policy());
        db::DynamicDeviceBuffer<int> buffer(runtime, 1);
        fake_pcuda::failWait(true);
        return; // must abort instead of freeing a potentially busy allocation
    }
    require(child > 0, "fork");
    int status = 0;
    require(waitpid(child, &status, 0) == child, "waitpid");
    require(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "unsafe teardown fails closed");
}
}
int main() try {
    fake_pcuda::reset();
    selection(); memoryPolicy(); buffers(); failures(); capabilities(); lifetime();
    typedStorage<int>(); typedStorage<uint8_t>(); typedStorage<int64_t>();
    typedStorage<double>(); typedStorage<unsigned long long>();
    require(fake_pcuda::allocations() == 0, "no leaked allocations");
    std::cout << "PCUDA runtime ownership, ordering, guards and failure tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
