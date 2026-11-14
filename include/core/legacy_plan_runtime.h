#pragma once
/** @file
 * @brief Historical ABI-negative fixture adapter, not linked into the engine or manual SSB.
 * @details Retains the old backend identity so tests can reject it before launch.
 * This is not a second supported production backend.
 */
#include "core/execution.h"

namespace db {
/** @brief Borrowed entry points/descriptors, kept alive by the loaded bundle's RAII owner. */
struct LegacyKernelEntry { const KernelModuleDescriptorV1* descriptor; DbKernelLaunchV1 launch; };
/** @brief Guard actual resident high-water capacities before preparing result storage. */
void prepareLegacyPlanResult(ExecutionContext& context, size_t elements);
/** @brief Interpret with PCUDA resources and a temporary same-executor legacy launch view. */
void executeLegacyPlan(const ExecutionPlan& plan, const ParameterValues& values,
                       const std::vector<LegacyKernelEntry>& kernels, ExecutionContext& context);
} // namespace db
