#include "typed_params.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
using namespace stage0;
using namespace param_probe;

void execute(const std::vector<std::string>& command) {
    std::vector<char*> argv;
    for (const auto& arg : command) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error("fork failed");
    if (child == 0) {
        execv(argv[0], argv.data());
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) throw std::runtime_error("compiler job failed");
}

// Independent explicit formulas for each promotion order; does not call GPU helpers.
Scalar expected(Scalar input, Parameters p, unsigned mask) {
    Scalar out{};
    out.tag = mask ? floating : integer;
    out.valid = input.valid;
    for (const auto& slot : p.slots) out.valid = out.valid && slot.valid;
    if (!out.valid) return out;
    const long long x = input.value.i;
    out.tag = (mask & 7) ? floating : integer;
    switch (mask & 7) {
        case 0: out.value.i = (x + p.slots[0].value.i) * p.slots[1].value.i + p.slots[2].value.i; break;
        case 1: out.value.d = (static_cast<double>(x) + p.slots[0].value.d) * p.slots[1].value.i + p.slots[2].value.i; break;
        case 2: out.value.d = static_cast<double>(x + p.slots[0].value.i) * p.slots[1].value.d + p.slots[2].value.i; break;
        case 3: out.value.d = (static_cast<double>(x) + p.slots[0].value.d) * p.slots[1].value.d + p.slots[2].value.i; break;
        case 4: out.value.d = static_cast<double>((x + p.slots[0].value.i) * p.slots[1].value.i) + p.slots[2].value.d; break;
        case 5: out.value.d = (static_cast<double>(x) + p.slots[0].value.d) * p.slots[1].value.i + p.slots[2].value.d; break;
        case 6: out.value.d = static_cast<double>(x + p.slots[0].value.i) * p.slots[1].value.d + p.slots[2].value.d; break;
        case 7: out.value.d = (static_cast<double>(x) + p.slots[0].value.d) * p.slots[1].value.d + p.slots[2].value.d; break;
        default: throw std::runtime_error("unknown type mask");
    }
    if constexpr (parameterCount == 4) {
        if (out.tag == floating || (mask & 8)) {
            const double left = out.tag == floating ? out.value.d : static_cast<double>(out.value.i);
            out.value.d = left + ((mask & 8) ? p.slots[3].value.d : static_cast<double>(p.slots[3].value.i));
            out.tag = floating;
        } else out.value.i += p.slots[3].value.i;
    }
    return out;
}

Parameters values(unsigned mask) {
    Parameters p{};
    const long long integers[] = {(1LL << 53) + 1, 3, -7, 5};
    const double doubles[] = {0.5, 0.5, 0.25, 0.125};
    for (unsigned i = 0; i < parameterCount; ++i) {
        p.slots[i].tag = (mask >> i) & 1;
        p.slots[i].valid = 1;
        if (p.slots[i].tag == floating) p.slots[i].value.d = doubles[i];
        else p.slots[i].value.i = integers[i];
    }
    return p;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: types STRATEGY OUTPUT ACPP SOURCE FLAGS DEVICE");
        const std::string strategy(argv[1]);
        if (strategy != "lazy" && strategy != "tags" && strategy != "eager" && strategy != "bounded")
            throw std::runtime_error("unknown strategy");
        if (strategy == "bounded" && parameterCount != 4) throw std::runtime_error("bounded control requires four parameters");
        const std::filesystem::path output(argv[2]), source(argv[4]);
        const auto prop = selectDevice(argv[6]);
        constexpr std::size_t count = 262147;
        if (!prop.pcudaHasFp64 || prop.maxThreadsPerBlock < 128 || prop.maxThreadsDim[0] < 128 ||
            prop.maxGridSize[0] < static_cast<int>(count / 128 + 1))
            throw std::runtime_error("type experiment needs FP64 and block 128");
        // Largest integer intermediate < 3 * (2^53 + 126) + 1000 < INT64_MAX.
        static_assert(((1ULL << 53) + 126) < (static_cast<unsigned long long>(std::numeric_limits<long long>::max()) - 1000) / 3,
                      "fixture arithmetic bound");
        std::vector<std::string> flags;
        std::ifstream file(argv[5]);
        for (std::string flag; std::getline(file, flag);) if (!flag.empty()) flags.push_back(flag);
        if (!file.eof() || flags.empty()) throw std::runtime_error("invalid flags file");
        Stream stream;
        Buffer<Scalar> input(stream.value, count), result(stream.value, count);
        Buffer<Scalar> hostInput(stream.value, count, true), hostResult(stream.value, count, true);
        for (std::size_t i = 0; i < count; ++i) {
            hostInput.data[i] = {};
            hostInput.data[i].value.i = static_cast<long long>(i % 251) - 125;
            hostInput.data[i].valid = i % 13 != 0;
        }
        check(pcudaMemcpyAsync(input.data, hostInput.data, checkedBytes(count, sizeof(Scalar)),
                              pcudaMemcpyHostToDevice, stream.value), "upload parameter input");
        check(pcudaStreamSynchronize(stream.value), "await parameter input");
        std::map<unsigned, std::unique_ptr<Module<Launch>>> modules;
        unsigned compilers = 0, linkers = 0;
        auto ensure = [&](unsigned mask) -> Launch {
            auto typed = modules.find(mask);
            if (typed != modules.end()) return typed->second->launch;
            const unsigned key = strategy == "tags" || (strategy == "bounded" && modules.size() >= 8)
                               ? tagIdentity : mask;
            auto found = modules.find(key);
            if (found != modules.end()) return found->second->launch;
            const auto name = "variant-" + std::to_string(key);
            const auto object = output / (name + ".o"), library = output / (name + ".so");
            if (std::filesystem::exists(library) || std::filesystem::exists(object))
                throw std::runtime_error("refusing to reuse persistent experimental module");
            std::vector<std::string> command = {"/usr/bin/bash", (source / "native_command.sh").string(),
                (output / (name + "-compile")).string(), argv[3]};
            command.insert(command.end(), flags.begin(), flags.end());
            command.push_back(key == tagIdentity ? "-DSTAGE0_RUNTIME_TAGS=1" : "-DSTAGE0_TYPE_MASK=" + std::to_string(mask));
            command.insert(command.end(), {"-ftime-trace=" + (output / (name + "-compile-trace.json")).string(),
                "-MD", "-MF", (output / (name + ".d")).string(), "-c", (source / "typed_params.cpp").string(), "-o", object.string()});
            execute(command);
            ++compilers;
            execute({"/usr/bin/bash", (source / "native_command.sh").string(), (output / (name + "-link")).string(), argv[3],
                "--acpp-targets=generic", "--acpp-pcuda", "-fPIC", "-shared", "-fuse-ld=lld", "-Wl,-O0",
                "-Wl,--time-trace=" + (output / (name + "-link-trace.json")).string(), object.string(), "-o", library.string()});
            ++linkers;
            auto module = std::make_unique<Module<Launch>>(library.c_str(), stream.value, "launch_parameters");
            const auto launch = module->launch;
            modules.emplace(key, std::move(module));
            return launch;
        };
        const auto coldStart = Clock::now();
        if (strategy == "eager") for (unsigned mask = 0; mask < tagIdentity; ++mask) ensure(mask);
        std::cout << std::setprecision(9) << "{\"strategy\":\"" << strategy << "\",\"cases\":[";
        unsigned caseIndex = 0;
        const unsigned masks[] = {0, 1, 3, 7, 2, 6, 4, 5, 8, 9, 11, 15, 10, 14, 12, 13};
        for (unsigned index = 0; index < tagIdentity; ++index) {
            const unsigned mask = masks[index];
            // Three value/validity fixtures for each of eight independent type combinations.
            for (unsigned fixture = 0; fixture < 3; ++fixture) {
                auto p = values(mask);
                if (fixture == 1) p.slots[0].valid = 0;
                if (fixture == 2) for (auto& slot : p.slots) slot.valid = 0;
                const auto caseStart = Clock::now();
                const auto beforeCompile = compilers;
                const auto launch = ensure(mask);
                const auto prepareMs = elapsed(caseStart);
                if (caseIndex++) std::cout << ',';
                std::cout << "{\"mask\":" << mask << ",\"fixture\":" << fixture
                          << ",\"new_compiles\":" << compilers - beforeCompile << ",\"prepare_ms\":" << prepareMs << ",\"runs\":[";
                for (unsigned run = 0; run < 9; ++run) {
                    const auto start = Clock::now();
                    check(launch(stream.value, input.data, result.data, count, p), "parameter launch");
                    check(pcudaStreamSynchronize(stream.value), "await parameter execution");
                    const auto executionMs = elapsed(start);
                    check(pcudaMemcpyAsync(hostResult.data, result.data, checkedBytes(count, sizeof(Scalar)),
                                           pcudaMemcpyDeviceToHost, stream.value), "parameter fetch");
                    check(pcudaStreamSynchronize(stream.value), "await parameter fetch");
                    if (run) std::cout << ',';
                    std::cout << "{\"execute_ms\":" << executionMs;
                    if (caseIndex == 1 && run == 0) std::cout << ",\"cold_first_ms\":" << elapsed(coldStart);
                    std::cout << '}';
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto want = expected(hostInput.data[i], p, mask), got = hostResult.data[i];
                        if (got.tag != want.tag || got.valid != want.valid ||
                            (want.valid && (want.tag == integer ? got.value.i != want.value.i : got.value.d != want.value.d)))
                            throw std::runtime_error("parameter mixed/NULL result mismatch");
                    }
                }
                std::cout << "]}";
            }
        }
        // Independent equal-valued parameters must not collapse into one slot.
        auto p = values(0);
        for (unsigned change = 0; change < 3; ++change) {
            p.slots[0].value.i = change == 2 ? 15 : 10;
            p.slots[2].value.i = change == 1 ? 15 : 10;
            const auto before = compilers;
            const auto launch = ensure(0);
            check(launch(stream.value, input.data, result.data, count, p), "independent parameter launch");
            check(pcudaMemcpyAsync(hostResult.data, result.data, checkedBytes(count, sizeof(Scalar)),
                                   pcudaMemcpyDeviceToHost, stream.value), "independent parameter fetch");
            check(pcudaStreamSynchronize(stream.value), "await independent parameter result");
            if (compilers != before) throw std::runtime_error("value-only change recompiled");
            for (std::size_t i = 0; i < count; ++i) {
                const auto want = expected(hostInput.data[i], p, 0);
                if (hostResult.data[i].valid != want.valid || (want.valid && hostResult.data[i].value.i != want.value.i))
                    throw std::runtime_error("independent equal-valued slots collapsed");
            }
        }
        const unsigned expectedCompiles = strategy == "tags" ? 1 : strategy == "bounded" ? 9 : tagIdentity;
        if (compilers != expectedCompiles || linkers != expectedCompiles || modules.size() != expectedCompiles)
            throw std::runtime_error("unexpected module/compiler/linker counters");
        std::cout << "],\"compiler_calls\":" << compilers << ",\"linker_calls\":" << linkers
                  << ",\"module_loads\":" << modules.size() << ",\"independent_slots_verified\":true,\"verified\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
