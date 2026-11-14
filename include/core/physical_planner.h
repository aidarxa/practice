#pragma once
/** @file
 * @brief Physical lowering helpers, separate from kernel-body generation and execution.
 */
#include "core/execution_plan.h"
#include <map>
#include <set>

namespace db {
/** @brief Static PCUDA shared-memory declaration; query storage is always runtime-owned. */
struct KernelLocalArray { std::string name; uint32_t type; uint64_t elements; };
/** @brief Only device code and local-memory declarations; no host execution algorithm. */
struct KernelBody {
    enum class Kind { Range, NdRange } kind;
    std::string name;
    std::string item;
    std::string code;
    std::vector<KernelLocalArray> locals;
    std::vector<std::string> scalar_captures;
    /** @brief Launch shape and optional range extent are not SQL literal specialization. */
    uint32_t threads = 1;
    SizeExpr range_items;
};
/** @brief Kernel contracts/operations and bodies have independent consumers. */
struct PlannedQuery { ExecutionPlan plan; std::vector<KernelBody> bodies; };

/**
 * @brief Lower physical operations without executing or emitting host C++.
 * @details Buffer declarations are the sole symbol authority. The temporary
     * lowering captures registered buffer identifiers used by a kernel body;
 * this is not parsing host source or inferring arguments in the loader. Dynamic
 * scalar captures are explicit. Table row counts are registered common scalars.
 */
class PhysicalPlanBuilder {
public:
    ExecutionPlan plan;
    PlanBlock post;
    /** @brief Declare one buffer identity; repeated identical declarations are idempotent. */
    bool declareBuffer(PlanBuffer buffer);
    /** @brief Declare and ensure scratch; initialization is a separate visible step. */
    void scratch(PlanBlock& block, const std::string& name, uint32_t type, SizeExpr elements, bool clear = false,
                 const std::string& legacy_cpp_type = {});
    /** @brief Add one real kernel and its launch, preserving fused device code. */
    void kernel(PlanBlock& block, KernelBody body, SizeExpr grid, uint32_t threads = 1);
    /** @brief Add a legacy range kernel with an explicit item count. */
    void range(PlanBlock& block, const std::string& name, const std::string& code, SizeExpr items,
               std::vector<std::string> captures = {});
    /** @brief Add a block kernel with explicit group count, block size and local memory. */
    void ndRange(PlanBlock& block, const std::string& name, const std::string& code, SizeExpr groups,
                 uint32_t threads, std::vector<std::string> captures = {}, std::vector<KernelLocalArray> locals = {});
    /** @brief Publish typed output buffer bindings before a materialization kernel. */
    void columns(PlanBlock& block, const std::vector<uint32_t>& types, SizeExpr rows);
    /** @brief Add a common scalar, such as an input table row count. */
    void commonScalar(const std::string& name, SizeExpr value);
    /** @brief Finish exact slot bindings and concatenate explicit ordered blocks, not source text. */
    PlannedQuery finish(size_t tuple_size, SizeExpr result_storage_elements,
                        const ParameterSchema& parameters = {}) const;
private:
    std::vector<KernelBody> bodies_;
    std::map<std::string, SizeExpr> common_scalars_;
};

} // namespace db
