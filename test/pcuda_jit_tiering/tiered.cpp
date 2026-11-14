#include "tier.hpp"
#include "jit_control.hpp"
#include "../pcuda_stage0/runtime.hpp"
#include <algorithm>
#include <cerrno>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;
namespace {
using namespace stage0;
constexpr long long base = 9007199254740993LL;
void build(const char* script) {
    pid_t pid;
    char bash[] = "bash";
    char* args[] = {bash, const_cast<char*>(script), nullptr};
    const int error = posix_spawnp(&pid, bash, nullptr, nullptr, args, environ);
    if (error) throw std::runtime_error("spawn build: " + std::to_string(error));
    int status;
    pid_t waited;
    do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
        throw std::runtime_error("background/build command failed");
}
void ready() {
    if (const char* text = std::getenv("PERF_READY_FD")) {
        const int fd = std::stoi(text);
        ssize_t written;
        do { written = write(fd, "R", 1); } while (written < 0 && errno == EINTR);
        if (written != 1) throw std::runtime_error("CPU-ready notification failed");
        close(fd);
    }
}
struct Join {
    std::thread thread;
    ~Join() { if (thread.joinable()) thread.join(); }
};
struct DriverLifetime {
    void* handle = nullptr;
    DriverLifetime() {
        if (const char* path = std::getenv("JIT_POC_DRIVER_PIN")) {
            // Diagnostic only: do not unmap vendor runtime code while its
            // internal helper threads/finalizers may still be using it.
            handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
            if (!handle) throw std::runtime_error(dlerror());
        }
    }
    ~DriverLifetime() { if (handle && dlclose(handle)) std::abort(); }
};
struct Run {
    const char* tier;
    long long threshold, offset;
    double execute, fetch, total;
    unsigned long long before, after;
    std::vector<long long> result;
};
}

int main(int argc, char** argv) {
    const auto mainStart = Clock::now();
    try {
        DriverLifetime driverLifetime;
        if (argc != 11) throw std::runtime_error("usage: tiered DEVICE LOW.sh HIGH.sh DIR IR TOOL CLANG MODE ROWS RUNS");
        const std::filesystem::path directory(argv[4]);
        const int lowIR = std::stoi(argv[5]), lowTool = std::stoi(argv[6]), lowClang = std::stoi(argv[7]);
        const std::string mode(argv[8]);
        const auto rows = std::stoull(argv[9]);
        const int runs = std::stoi(argv[10]);
        if (rows < 1 || rows > 64 * 1024 * 1024 || runs < 2 || runs > 50 ||
            (mode != "overlap" && mode != "idle" && mode != "failure"))
            throw std::runtime_error("invalid experiment arguments");
        const auto prop = selectDevice(argv[1]);
        if (prop.maxThreadsPerBlock < 128 || prop.maxThreadsDim[0] < 128 ||
            (rows + 127) / 128 > static_cast<unsigned>(prop.maxGridSize[0]))
            throw std::runtime_error("unsupported launch geometry");
        const auto bytes = checkedBytes(rows, sizeof(long long));
        if (bytes > prop.totalGlobalMem / 8) throw std::runtime_error("memory budget");
        int backend, platform, device;
        check(pcudaGetBackend(&backend), "get backend");
        check(pcudaGetPlatform(&platform), "get platform");
        check(pcudaGetDevice(&device), "get device");
        Stream stream;
        const auto startup = elapsed(mainStart);
        const auto dataStart = Clock::now();
        Buffer<long long> values(stream.value, rows), hostValues(stream.value, rows, true);
        Buffer<unsigned char> valid(stream.value, rows), hostValid(stream.value, rows, true);
        for (std::size_t i = 0; i < rows; ++i) {
            hostValues.data[i] = base + static_cast<long long>(i % 1009) - 504;
            hostValid.data[i] = i % 11 != 0;
        }
        check(pcudaMemcpyAsync(values.data, hostValues.data, bytes, pcudaMemcpyHostToDevice, stream.value), "upload values");
        check(pcudaMemcpyAsync(valid.data, hostValid.data, rows, pcudaMemcpyHostToDevice, stream.value), "upload validity");
        check(pcudaStreamSynchronize(stream.value), "data ready");
        const auto dataMs = elapsed(dataStart);
        const auto summaryStart = Clock::now();
        const auto prepStart = Clock::now();
        Buffer<long long> output(stream.value, rows), hostOutput(stream.value, rows, true);
        std::vector<Run> records;
        records.reserve(2 * runs + 2);
        const auto prepMs = elapsed(prepStart);
        jit_poc_set_policy(lowIR, lowTool, lowClang);
        const auto buildStart = Clock::now();
        build(argv[2]);
        const auto buildMs = elapsed(buildStart);
        const auto loadStart = Clock::now();
        Module<TierLaunch> low((directory / "low.so").c_str(), stream.value, "tier_launch");
        const auto loadMs = elapsed(loadStart);
        auto execute = [&](TierLaunch launch, const char* tier, int index) {
            const long long threshold = index % 4 == 3 ? base + 1000 : base + (index % 3) - 1;
            const long long offset = index * 13 - 47;
            const auto before = jit_poc_compile_count();
            const auto start = Clock::now();
            check(launch(stream.value, {values.data, valid.data, output.data, rows, threshold, offset}), "filter/project");
            check(pcudaStreamSynchronize(stream.value), "execute wait");
            const auto executeMs = elapsed(start);
            const auto fetchStart = Clock::now();
            check(pcudaMemcpyAsync(hostOutput.data, output.data, bytes, pcudaMemcpyDeviceToHost, stream.value), "fetch");
            check(pcudaStreamSynchronize(stream.value), "CPU ready");
            const auto fetchMs = elapsed(fetchStart), totalMs = elapsed(start);
            Run result{tier, threshold, offset, executeMs, fetchMs, totalMs, before, jit_poc_compile_count(), {}};
            return result;
        };
        auto first = execute(low.launch, "low-first", 0);
        const auto summaryMs = elapsed(summaryStart), mainToFirstMs = elapsed(mainStart);
        ready(); // No background work starts before this CPU-ready marker.
        first.result.assign(hostOutput.data, hostOutput.data + rows);
        records.push_back(std::move(first));
        if (jit_poc_compile_count() != 1) throw std::runtime_error("interposition/cache counter failed");
        std::unique_ptr<Module<TierLaunch>> high;
        std::exception_ptr backgroundError;
        double backgroundBuild = 0, backgroundLoad = 0, backgroundPrepare = 0, backgroundTotal = 0;
        jit_poc_arm_notification();
        const auto backgroundStart = Clock::now();
        Join worker;
        worker.thread = std::thread([&] {
            try {
                const auto start = Clock::now();
                build(argv[3]);
                backgroundBuild = elapsed(start);
                check(pcudaSetDeviceExt(backend, platform, device), "worker select device");
                Stream preparationStream;
                const auto loading = Clock::now();
                auto module = std::make_unique<Module<TierLaunch>>((directory / "high.so").c_str(), stream.value, "tier_launch");
                backgroundLoad = elapsed(loading);
                jit_poc_set_policy(3, 3, 0);
                const auto preparing = Clock::now();
                check(module->launch(preparationStream.value, {values.data, valid.data, output.data, 0, base, 0}), "prepare optimized kernel");
                check(pcudaStreamSynchronize(preparationStream.value), "optimized kernel ready");
                backgroundPrepare = elapsed(preparing);
                high = std::move(module); // Published only after join, never replacing active code.
            } catch (...) { backgroundError = std::current_exception(); }
            backgroundTotal = elapsed(backgroundStart);
        });
        if (mode == "overlap") {
            if (!jit_poc_wait_notification(30000)) throw std::runtime_error("background optimizer not entered");
            auto run = execute(low.launch, "low-during-background", 1);
            run.result.assign(hostOutput.data, hostOutput.data + rows);
            records.push_back(std::move(run));
        }
        worker.thread.join();
        if (backgroundError && mode != "failure") std::rethrow_exception(backgroundError);
        if (!backgroundError && mode == "failure") throw std::runtime_error("failure injection did not fail");
        const auto countAtPromotion = jit_poc_compile_count();
        for (int index = 1; index <= runs; ++index) {
            auto lowRun = execute(low.launch, "low-warm", index);
            lowRun.result.assign(hostOutput.data, hostOutput.data + rows);
            records.push_back(std::move(lowRun));
            if (high) {
                auto highRun = execute(high->launch, "high-warm", index);
                highRun.result.assign(hostOutput.data, hostOutput.data + rows);
                records.push_back(std::move(highRun));
            }
        }
        if (jit_poc_compile_count() != countAtPromotion) throw std::runtime_error("changed values caused device recompile");
        std::ofstream binary(directory / "results.bin", std::ios::binary);
        for (const auto& run : records) binary.write(reinterpret_cast<const char*>(run.result.data()), bytes);
        if (!binary) throw std::runtime_error("write correctness outputs");
        std::cout << std::setprecision(12) << "{\"scope\":\"synthetic filter/project; no SQL parser\",\"rows\":" << rows
            << ",\"summary_ms\":" << summaryMs << ",\"main_to_first_result_ms\":" << mainToFirstMs
            << ",\"startup_ms\":" << startup << ",\"data_load_ms\":" << dataMs << ",\"preparation_ms\":" << prepMs
            << ",\"build_elapsed_ms\":" << buildMs << ",\"load_ms\":" << loadMs
            << ",\"background_build_ms\":" << backgroundBuild << ",\"background_load_ms\":" << backgroundLoad
            << ",\"background_prepare_ms\":" << backgroundPrepare << ",\"background_total_ms\":" << backgroundTotal
            << ",\"background_failed\":" << (backgroundError ? "true" : "false")
            << ",\"promoted\":" << (high ? "true" : "false") << ",\"compile_count_at_promotion\":" << countAtPromotion
            << ",\"compile_count_final\":" << jit_poc_compile_count() << ",\"runs\":[";
        bool comma = false;
        for (const auto& run : records) {
            if (comma) std::cout << ',';
            comma = true;
            std::cout << "{\"tier\":\"" << run.tier << "\",\"threshold\":" << run.threshold << ",\"offset\":" << run.offset
                << ",\"execute_ms\":" << run.execute << ",\"fetch_ms\":" << run.fetch << ",\"total_ms\":" << run.total
                << ",\"compilations_before\":" << run.before << ",\"compilations_after\":" << run.after << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
