#pragma once
/** @file @brief Process-only module preparation, ownership and instrumentation (ordinary C++17). */
#include "core/config.h"
#include "core/module_codegen.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>

namespace db {
/** @brief Cumulative events, not timing heuristics; compatibility reuse must leave build/load counts unchanged. */
struct ModuleCounters {
    uint64_t plan_codegen = 0, module_codegen = 0, compiler = 0, linker = 0;
    uint64_t hits = 0, misses = 0, loads = 0, symbols = 0;
};
/** @brief Instrumentation cannot silently wrap even in a very long-lived session. */
inline void recordModuleEvent(uint64_t& counter, uint64_t count = 1) { counter = checkedSizeAdd(counter,count); }
/** @brief Native command sums and direct preparation elapsed are distinct under parallel compilation. */
struct ModuleBuildWave {
    size_t first=0,count=0;
    uint64_t begin_ns=0,end_ns=0;
};
/** @brief Experimental waves are absolute observations, never additional component-sum terms. */
struct ModuleBuildTiming {
    double compile_wall_ms = 0, compile_user_ms = 0, compile_system_ms = 0;
    double link_wall_ms = 0, link_user_ms = 0, link_system_ms = 0;
    double build_elapsed_ms = 0, load_ms = 0;
    std::vector<ModuleBuildWave> waves;
};
/** @brief A private mkdtemp directory, never a persistent SQL-module cache. */
class ModuleArtifacts {
public:
    ModuleArtifacts(const std::filesystem::path& parent, bool keep);
    ~ModuleArtifacts() noexcept;
    ModuleArtifacts(const ModuleArtifacts&) = delete;
    ModuleArtifacts& operator=(const ModuleArtifacts&) = delete;
    const std::filesystem::path& path() const { return path_; }
    /** @brief Preserve a failed build's diagnostic evidence; never reuse it as a cache. */
    void preserve() { keep_ = true; }
private:
    std::filesystem::path path_;
    bool keep_;
};
struct CompiledModule {
    std::filesystem::path path;
    std::shared_ptr<ModuleArtifacts> artifacts;
};
/** @brief Compiler strategy does not know SQL, query buffers or the execution stream. */
class ICompiler {
public:
    virtual ~ICompiler() = default;
    virtual const std::string& fingerprint() const = 0;
    virtual std::vector<CompiledModule> compile(const std::vector<KernelModuleSource>& modules,
                                               ModuleCounters&, ModuleBuildTiming&) = 0;
};
/** @brief Stock acpp compile/link chains with a validated build-time PCH and a bounded worker count. */
class AdaptiveCppCompiler final : public ICompiler {
public:
    AdaptiveCppCompiler();
    explicit AdaptiveCppCompiler(const CrystalConfig& config);
    ~AdaptiveCppCompiler() override;
    AdaptiveCppCompiler(const AdaptiveCppCompiler&) = delete;
    AdaptiveCppCompiler& operator=(const AdaptiveCppCompiler&) = delete;
    const std::string& fingerprint() const override;
    std::vector<CompiledModule> compile(const std::vector<KernelModuleSource>& modules,
                                       ModuleCounters&, ModuleBuildTiming&) override;
    /** @brief Inspect reproducibility logs, without using this directory as a future-process cache. */
    const std::filesystem::path& artifactPath() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
/** @brief One loaded .so and resolved ABI entries. Completion owners precede unload and artifact removal. */
class LoadedKernelModule {
public:
    LoadedKernelModule(CompiledModule binary, const std::string& identity, ModuleCounters& counters,
                       const std::vector<KernelArgumentSchemaV2>& schema = {});
    ~LoadedKernelModule() noexcept;
    LoadedKernelModule(const LoadedKernelModule&) = delete;
    LoadedKernelModule& operator=(const LoadedKernelModule&) = delete;
    const KernelModuleDescriptorV2* descriptor() const { return descriptor_; }
    DbKernelLaunchV2 launch() const { return launch_; }
    /** @brief Register one resource domain before submission; drain must fail closed and not throw. */
    void retainCompletion(const void* identity, std::function<void()> drain);
private:
    CompiledModule binary_;
    void* handle_ = nullptr;
    const KernelModuleDescriptorV2* descriptor_ = nullptr;
    DbKernelLaunchV2 launch_ = nullptr;
    std::unordered_map<const void*, std::function<void()>> completions_;
};
using LoadedKernels = std::vector<std::shared_ptr<LoadedKernelModule>>;
/** @brief Borrowed, main-thread-only readiness lookup; never retain it beyond its execution callback. */
using KernelLookup = std::function<std::shared_ptr<LoadedKernelModule>(size_t)>;
/** @brief Full-key cache of loaded kernels, with at most eight typed schemas and one tagged fallback per layout. */
class ProcessModuleCache {
public:
    ProcessModuleCache() = default;
    ProcessModuleCache(const ProcessModuleCache&) = delete;
    ProcessModuleCache& operator=(const ProcessModuleCache&) = delete;
    /**
     * @brief Select ready modules and emit every missing source before native build/load.
     * @param sources_ready Optional synchronous observation boundary, called exactly once after
     * successful source selection, including all-hit plans, and before ICompiler::compile.
     * @details The callback is not retained. It neither publishes cache reservations nor changes
     * the existing transactional build/ABI validation policy; a failed preparation remains retryable.
     */
    LoadedKernels prepare(const PlannedQuery& query, ICompiler& compiler, ModuleCounters&, ModuleBuildTiming&,
                           const std::function<void()>& sources_ready = {});
    /**
     * @brief Opt-in experiment: execute with per-module readiness, then atomically publish all modules.
     * @details All sources/reservations are selected first. Only the worker touches ICompiler; only
     * this caller loads, validates and uses modules. Callback errors and late build/ABI failures
     * join the worker and drop staged modules without consuming typed slots. The execution callback
     * must drain submitted GPU work on unwind; LoadedKernelModule completion owners also drain
     * before unload. No partly ready module enters the process cache.
     */
    LoadedKernels prepareOverlapped(const PlannedQuery&,ICompiler&,ModuleCounters&,ModuleBuildTiming&,size_t jobs,
                                   const std::function<void(const KernelLookup&)>& execute,
                                   const std::function<void()>& sources_ready = {});
    size_t size() const { return modules_.size(); }
private:
    struct Preparation;
    std::unordered_map<std::string, std::shared_ptr<LoadedKernelModule>> modules_;
    std::unordered_map<std::string, size_t> typed_;
    std::unordered_map<std::string, std::string> identities_;
};
/** @brief Source compatibility for callers of the former path-only cache; storage is now loaded modules. */
using FileBasedQueryCache = ProcessModuleCache;
} // namespace db
