#pragma once
/** @file @brief One actual kernel plus its fixed ABI wrapper, without host query execution. */
#include "core/physical_planner.h"

namespace db {
/** @brief Immutable source recipe. Parameter slot/buffer bindings remain exclusively in the plan. */
struct KernelModuleSource {
    std::string structure;
    std::string key;
    std::string identity;
    std::string source;
    std::vector<KernelArgumentSchemaV2> schema;
};
/** @brief Canonicalize local ABI names so unchanged kernels share modules across distinct plans. */
KernelModuleSource generatePcudacode(const KernelContract& contract, const KernelBody& body,
                                   const std::string& toolchain, bool tagged = false, bool emit_source = true);
/** @brief Inspect separately compilable sources; separators are comments, not a query bundle. */
std::string describePcudamodules(const PlannedQuery& query);
} // namespace db
