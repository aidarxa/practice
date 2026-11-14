// Research-only ELF interposition for AdaptiveCpp 5989e1cf / LLVM 17 on Linux.
// No installed library is modified. Never deploy this as an engine component.
// Compiler-method adaptations: Copyright The AdaptiveCpp Contributors.
// SPDX-License-Identifier: BSD-2-Clause (see LICENSE in this directory).
#include "jit_control.hpp"
#include <hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp>
#include <hipSYCL/compiler/llvm-to-backend/Utils.hpp>
#include <hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.hpp>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#define __HIP_PLATFORM_AMD__
#include <hip/hiprtc.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {
using Clock = std::chrono::steady_clock;
const auto origin = Clock::now();
double stamp() {
    return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}
int setting(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed < 0 || parsed > 3) std::abort();
    return static_cast<int>(parsed);
}
struct Policy { int ir, tool, clang; };
thread_local Policy policy{setting("JIT_POC_IR", 3), setting("JIT_POC_TOOL", 3),
                           setting("JIT_POC_CLANG", 0)};
thread_local unsigned long long compilation = 0;
std::atomic<unsigned long long> count{0}, tools{0};
std::mutex mutex;
std::condition_variable notification;
bool armed = false, notified = false;
llvm::json::Array events;

void record(const char* kind, double start, llvm::json::Object detail = {}) {
    static const int registered = std::atexit([] {
        if (const char* directory = std::getenv("JIT_POC_DIRECTORY"))
            if (jit_poc_dump((std::string(directory) + "/jit-events.json").c_str())) std::abort();
    });
    (void)registered;
    detail["kind"] = kind;
    detail["start_ms"] = start;
    detail["end_ms"] = stamp();
    detail["compilation"] = static_cast<int64_t>(compilation);
    detail["ir_level"] = policy.ir;
    detail["tool_level"] = policy.tool;
    detail["clang_route"] = policy.clang != 0;
    std::lock_guard<std::mutex> lock(mutex);
    events.emplace_back(std::move(detail));
}
}

extern "C" void jit_poc_set_policy(int ir, int tool, int clang) {
    if (ir < 0 || ir > 3 || tool < 0 || tool > 3 || clang < 0 || clang > 1)
        std::abort();
    policy = {ir, tool, clang};
}
extern "C" unsigned long long jit_poc_compile_count() { return count.load(); }
extern "C" void jit_poc_arm_notification() {
    std::lock_guard<std::mutex> lock(mutex);
    armed = true;
    notified = false;
}
extern "C" int jit_poc_wait_notification(unsigned timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return notification.wait_for(lock, std::chrono::milliseconds(timeout), [] { return notified; });
}
extern "C" int jit_poc_dump(const char* path) {
    std::string text;
    {
        std::lock_guard<std::mutex> lock(mutex);
        llvm::raw_string_ostream stream(text);
        llvm::json::Object document;
        document["compilations"] = static_cast<int64_t>(count.load());
        document["external_tools"] = static_cast<int64_t>(tools.load());
        document["events"] = llvm::json::Array(events);
        stream << llvm::json::Value(std::move(document));
    }
    std::ofstream file(path);
    file << text << '\n';
    return file ? 0 : 1;
}

extern "C" hiprtcResult hiprtcLinkComplete(hiprtcLinkState state, void** binary, size_t* size) {
    using Function = hiprtcResult (*)(hiprtcLinkState, void**, size_t*);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hiprtcLinkComplete"));
    if (!original) std::abort();
    const auto start = stamp();
    const auto result = original(state, binary, size);
    llvm::json::Object detail;
    detail["status"] = static_cast<int>(result);
    record("hiprtc_complete", start, std::move(detail));
    return result;
}

namespace hipsycl::compiler {
// Same required lowering pipeline as upstream; only the optional LLVM default
// optimization pipeline is selected here. O0 requires its dedicated builder.
bool LLVMToBackendTranslator::optimizeFlavoredIR(llvm::Module& module, PassHandler& passes) {
    const auto start = stamp();
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (armed) {
            armed = false;
            notified = true;
            notification.notify_all();
        }
    }
    module.getContext().setDiagnosticHandlerCallBack(
        [](const llvm::DiagnosticInfo& info, void*) {
            if (info.getSeverity() == llvm::DS_Error) {
                llvm::DiagnosticPrinterRawOStream printer(llvm::errs());
                info.print(printer);
                llvm::errs() << '\n';
            }
        });
    const llvm::OptimizationLevel levels[] = {llvm::OptimizationLevel::O0, llvm::OptimizationLevel::O1,
        llvm::OptimizationLevel::O2, llvm::OptimizationLevel::O3};
    auto pipeline = policy.ir == 0
        ? passes.PassBuilder->buildO0DefaultPipeline(levels[0])
        : passes.PassBuilder->buildPerModuleDefaultPipeline(levels[policy.ir]);
    pipeline.run(module, *passes.ModuleAnalysisManager);
    llvm::json::Object detail;
    detail["identifier"] = getCompilationIdentifier();
    record("optimizer", start, std::move(detail));
    return true;
}

// Upstream fullTransformation, with a direct timer rather than cold-warm subtraction.
bool LLVMToBackendTranslator::fullTransformation(const std::string& input, std::string& output) {
    compilation = ++count;
    const auto start = stamp();
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
    auto error = loadModuleFromString(input, context, module);
    bool success = !error;
    if (error) {
        llvm::handleAllErrors(std::move(error), [&](llvm::ErrorInfoBase& info) {
            registerError(info.message());
        });
    }
    if (success) {
        success = prepareIR(*module);
        if (success) {
            // The installed AMD library devirtualizes this call internally.
            // Qualify it explicitly so the experiment really selects its route.
            if (module->getTargetTriple() == "amdgcn-amd-amdhsa")
                success = static_cast<LLVMToAmdgpuTranslator*>(this)->LLVMToAmdgpuTranslator::translateToBackendFormat(*module, output);
            else
                success = translatePreparedIR(*module, output);
        }
        if (!success) setFailedIR(*module);
    }
    llvm::json::Object detail;
    detail["success"] = success;
    detail["output_bytes"] = static_cast<int64_t>(output.size());
    record("full_transformation", start, std::move(detail));
    return success;
}

// Installed AMD path is hipRTC (including external opt), not necessarily clang.
// Explicitly distinguish the stock path from an experimental clang lowering path.
bool LLVMToAmdgpuTranslator::translateToBackendFormat(llvm::Module& module, std::string& output) {
    if (getWavefrontSize() != 32 && getWavefrontSize() != 64) {
        registerError("invalid AMD wavefront size");
        return false;
    }
    if (!policy.clang) {
        std::string bitcode;
        llvm::raw_string_ostream stream(bitcode);
        llvm::WriteBitcodeToFile(module, stream);
        stream.flush();
        return hiprtcJitLink(bitcode, output);
    }
    return clangJitLink(module, output);
}

int executeAndWait(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args,
                  std::optional<llvm::ArrayRef<llvm::StringRef>> environment,
                  llvm::ArrayRef<std::optional<llvm::StringRef>> redirects) {
    const auto start = stamp();
    const auto id = ++tools;
    std::vector<std::string> command;
    for (const auto& arg : args) command.push_back(arg.str());
    const bool clang = program.contains("clang");
    const bool opt = !clang && program.endswith("/opt");
    if (clang || opt) {
        for (auto& arg : command) {
            if (arg == "-O3") arg = "-O" + std::to_string(policy.tool);
        }
    }
    const char* directory = std::getenv("JIT_POC_DIRECTORY");
    const char* timer = std::getenv("JIT_POC_NATIVE_TIMER");
    const bool traced = setting("JIT_POC_TRACES", 0) != 0;
    std::string prefix;
    if (directory) prefix = std::string(directory) + "/s2-tool-" + std::to_string(id);
    if (traced && !prefix.empty()) {
        for (const auto& arg : command) {
            if (llvm::StringRef(arg).endswith(".bc")) {
                std::ifstream input(arg, std::ios::binary);
                if (input) {
                    std::ofstream saved(prefix + "-input.bc", std::ios::binary);
                    saved << input.rdbuf();
                    break;
                }
            }
        }
        if (clang) command.push_back("-ftime-trace=" + prefix + "-clang.json");
    }
    llvm::json::Array actual;
    for (const auto& arg : command) actual.emplace_back(arg);
    std::string executable = program.str();
    if (timer && !prefix.empty()) {
        command.insert(command.begin(), {"bash", timer, prefix});
        executable = "/usr/bin/bash";
    }
    std::vector<llvm::StringRef> refs;
    for (const auto& arg : command) refs.emplace_back(arg);
    // Sanitize a CHILD environment, never mutate process-global environ while
    // other threads are submitting kernels. Also avoid loading LLVM into Bash.
    std::vector<llvm::StringRef> childEnvironment;
    if (environment) {
        for (const auto& entry : *environment)
            if (!entry.startswith("LD_PRELOAD=") && !entry.startswith("JIT_POC_")) childEnvironment.push_back(entry);
    } else {
        for (char** entry = environ; *entry; ++entry) {
            llvm::StringRef text(*entry);
            if (!text.startswith("LD_PRELOAD=") && !text.startswith("JIT_POC_")) childEnvironment.push_back(text);
        }
    }
    const int status = llvm::sys::ExecuteAndWait(executable, refs, llvm::ArrayRef<llvm::StringRef>(childEnvironment), redirects);
    llvm::json::Object detail;
    detail["argv"] = std::move(actual);
    detail["status"] = status;
    detail["prefix"] = prefix;
    detail["tool"] = clang ? "clang" : opt ? "opt" : "other";
    record("external_tool", start, std::move(detail));
    return status;
}
}
