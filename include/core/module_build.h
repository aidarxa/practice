#pragma once
/** @file @brief Experimental one-writer compiler waves; no GPU or cache state is shared with the worker. */
#include "core/module_cache.h"

namespace db {
/**
 * @brief Own a finite sequence of at most four-chain waves and immutable readiness futures.
 * @details The caller remains the sole cache/loader/GPU owner. The worker alone uses ICompiler
 * until finish/destruction joins it. Diagnostics merge once, including on failure; no worker
 * outlives its compiler, caller counters, artifacts or timing sink. This is not parallel S2.
 */
class ModuleBuildSequence {
public:
    ModuleBuildSequence(ICompiler&,std::vector<KernelModuleSource>,size_t jobs,ModuleCounters&,ModuleBuildTiming&);
    ~ModuleBuildSequence() noexcept;
    ModuleBuildSequence(const ModuleBuildSequence&)=delete;
    ModuleBuildSequence& operator=(const ModuleBuildSequence&)=delete;
    /** @brief Wait only for this binary; a failed wave fails every not-yet-ready future. */
    CompiledModule module(size_t index);
    /** @brief Join, merge actual work once and rethrow any compiler error. Idempotent. */
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
