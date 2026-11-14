#pragma once
/** @file @brief Native PCUDA adapter for the backend-independent execution plan. */
#include "core/execution.h"
namespace db {
/** @brief Guard actual resident high-water capacities before preparing result storage. */
/** @brief Retain high-water workspace for one value-free plan, never the union of historical queries. */
void preparePcudaplanResult(ExecutionContext& context, size_t elements, const std::string& plan_identity);
/** @brief Execute all dependent operations on the resource domain's one explicit PCUDA stream. */
void executePcudaplan(const ExecutionPlan&, const ParameterValues&, const LoadedKernels&, ExecutionContext&);
/** @brief Experimental path: validate logical contracts eagerly and each loaded module before its launch. */
void executePcudaplan(const ExecutionPlan&,const ParameterValues&,const KernelLookup&,ExecutionContext&);
}
