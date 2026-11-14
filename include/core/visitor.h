#pragma once

#include "core/expressions.h"
#include "core/operators.h"
#include "core/memory.h"   // Catalog, TableMetadata, ColumnStatistics
#include "core/physical_planner.h"
#include "core/expression_codegen.h"

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace db {

/** @brief Fused device-register scope, not an execution or compilation unit. */
struct Pipeline {
    std::string       kernel_name;
    std::stringstream kernel_body;
};

// ============================================================================
// JITContext — shared mutable state threaded through both visitors.
// ============================================================================
struct JITContext {
    /** @brief Types only: ordinary parameter values never enter GPU lowering. */
    ParameterSchema parameter_schema;
    /** @brief Explicit host operations and separately stored kernel bodies. */
    PhysicalPlanBuilder physical;
    // Temporary device-body scopes; physical.plan.steps owns execution order.
    std::vector<Pipeline> pipelines;
    Pipeline*             current_pipeline = nullptr;

    void startNewPipeline(const std::string& name) {
        pipelines.emplace_back();
        current_pipeline = &pipelines.back();
        current_pipeline->kernel_name = name;
    }

    // ---- Column tracking (probe kernel) ----
    // column_name → register name in the probe kernel.
    std::unordered_map<std::string, std::string> col_to_reg;
    std::unordered_map<std::string, std::string> col_to_valid_reg;
    /** @brief Written register declarations survive removal of transient column bindings. */
    std::set<std::string> pipeline_validity_registers;

    // Catalog-derived nullability cache used by codegen to completely avoid
    // validity loads/checks for non-nullable columns in hot kernels.
    std::set<std::string> nullable_columns;

    // table_name → row-id vector register. Used by projection/materialization
    // after PK/FK joins: the hash table carries only build row_id, and
    // late columns are gathered by row_id at the end of the pipeline.
    std::unordered_map<std::string, std::string> table_rowid_regs;

    // Which columns have already been BlockLoad-ed in the probe kernel.
    std::set<std::string> loaded_in_probe;

    // Counter for unique temporary OR-mask names ("mask_1", "mask_2", …)
    int mask_counter = 0;

    std::string current_items_col;

    // Columns that must be fetched from ctx (EXTERNAL_INPUT):
    std::set<std::string> external_columns; // e.g. "d_lo_orderdate"
    std::set<std::string> external_null_columns; // nullable bitmap symbols, e.g. "n_lo_orderdate"

    // Result buffer size expression (set by AggregateNode/ProjectionNode visitor).
    // For AVG/MIN this is physical storage; explicit plan finalization compacts it.
    std::string result_size_expr;   // checked legacy size atom, e.g. "21000"
    int         tuple_size = 1;
    std::string visible_result_size_expr;

    // Exact projection materialization: the first scan pass counts surviving
    // rows per tile; explicit GPU prefix-scan computes offsets and result size;
    // the second scan pass writes rows into the exact result buffer.
    bool projection_exact_materialization = false;
    std::string projection_row_count_expr;
    int projection_tuple_size = 0;

    std::set<std::string> emitted_auxiliary_kernels;

    // ORDER BY/LIMIT is applied after the child pipeline has materialized the
    // visible result.  It therefore requires dense columnar output from an
    // aggregate child and disables sparse aggregate output.
    bool requires_dense_result = false;

    // Set when AggregateNode emits a fused small-result materialization that
    // already applies HAVING and/or ORDER BY/LIMIT. SortLimitNode must not
    // append its generic post-sort path in that case.
    bool sort_limit_post_execution_fused = false;

    // ---------- helpers ----------
    std::string getNewMask() {
        return "mask_" + std::to_string(++mask_counter);
    }

    std::string getNewBuildKernelName(const std::string& prefix) {
        return "build_hashtable_" + prefix;
    }
};

// ============================================================================
// JITExprVisitor — micro-level: emits BlockPred/BlockLoad calls into
// the currently active kernel stream for a single predicate subtree.
//
// AND:  both children write into target_mask_ (accumulating flags).
// OR:   creates a fresh zero-initialised temporary mask, visits both
//       children into it, then ANDs the temp mask into target_mask_.
// Leaf comparison:  emits BlockLoad (if column not yet loaded), then
//   the appropriate BlockPred{A|Or}{Op} call.
// ============================================================================
class JITExprVisitor : public ExprVisitor {
public:
    // `stream`        – kernel stream to write into.
    // `ctx`           – shared JIT context.
    // `target_mask`   – name of the flags/mask array to write into.
    // `is_or_context` – if true, leaf comparisons use BlockPredOr{Op}
    //                   instead of the default AND-accumulation primitives.
    // `first_pred`    – pointer to a bool shared across the whole
    //                   (AND-chain) subtree: first write uses BlockPred{Op},
    //                   subsequent writes use BlockPredA{Op}.
    //                   In OR-context this pointer is separate per OR-mask.
    JITExprVisitor(JITContext& ctx,
                   std::stringstream& stream,
                   const std::string& target_mask,
                   bool is_or_context = false,
                   bool* first_pred   = nullptr);

    void visit(const ColumnRefExpr&   node) override;
    void visit(const LiteralIntExpr&  node) override;
    void visit(const LiteralFloatExpr& node) override;
    void visit(const LiteralNullExpr& node) override;
    /** @brief A leaf emits no statements; numeric value/validity are captured as independent ABI slots. */
    void visit(const ParameterRefExpr& node) override;
    void visit(const BinaryExpr&      node) override;
    void visit(const CaseWhenExpr&    node) override;
    void visit(const StarExpr&        node) override;

    // Translates AST expressions into inline C++ code
    std::string translateInlineExpr(const ExprNode* expr, bool is_probe);

private:
    JITContext&        ctx_;
    std::stringstream& stream_;
    std::string        target_mask_;
    bool               is_or_context_;
    bool*              first_pred_;
    bool               own_first_pred_; // true when we allocated first_pred_ ourselves
    bool               local_first_pred_storage_ = true;

    // Emit a BlockLoad for `col` into `reg` if not already loaded.
    void ensureLoaded(const std::string& col, const std::string& reg);

    // Return the predicate function name suffix based on ExprType.
    static const char* predSuffix(ExprType t);

    // Visit a binary comparison node (OP_EQ, OP_LT, etc.)
    void visitComparison(const BinaryExpr& node);
};

// ============================================================================
// JITOperatorVisitor — macro-level: walks the Operator Tree using the
// Data-Centric Push Model and emits fused PCUDA kernel bodies into JITContext.
//
// Entry point: visit(AggregateNode) → produce(root)
//
// Runtime flow (canonical Data-Centric path, no legacy scalar fallback):
//   produce(Agg) → produce(HashJoin) → produce(Filter) → produce(Scan)
//   Scan opens a for-loop, then:
//   consumeVector(Filter) → consumeVector(HashJoin) → consumeVector(Agg)
//   If a join expands 1-to-N (MHT), control switches to consumeItem(...)
//   for the remainder of the parent chain in that scope.
//
// Pipeline Breakers:
//   HashBuildNode generates one or more separate build kernels (pipelines)
//   before triggering the probe pipeline.
// ============================================================================
class JITOperatorVisitor : public OperatorVisitor {
public:
    enum class ExecutionMode : uint8_t {
        DataCentric = 0
    };

    explicit JITOperatorVisitor(JITContext& ctx, const Catalog& catalog);

    // ---- Legacy OperatorVisitor interface (entry point via accept()) ----
    // AggregateNode::accept() calls visit(AggregateNode) which kicks off produce().
    // Other visit() methods are stubs — traversal is driven by produce/consume.
    void visit(const TableScanNode& node) override;
    void visit(const FilterNode&    node) override;
    void visit(const HashJoinNode&  node) override;
    void visit(const AggregateNode& node) override;
    void visit(const ProjectionNode& node) override;
    void visit(const SortLimitNode& node) override;

    // ---- Push-model dispatchers ----
    void produce(const OperatorNode* node, JITContext& ctx);

    // Vector-mode consumer: called OUTSIDE any scalar loop.
    // All Crystal block primitives (BlockLoad, BlockPred, BlockProbeAndPHT) are valid here.
    void consumeVector(const OperatorNode* node, JITContext& ctx,
                       const OperatorNode* sender,
                       const std::vector<std::string>& active_vars);

    // Item-mode consumer: called INSIDE a scalar expansion loop (MHT j-loop).
    // Only scalar / atomic operations are valid here.
    void consumeItem(const OperatorNode* node, JITContext& ctx,
                     const OperatorNode* sender,
                     const std::vector<std::string>& active_vars);

    // Canonical dispatcher:
    //  - called only by produce* entry points.
    //  - routes into consumeVector/consumeItem according to consume_mode_.
    //  - legacy scalar consume path is intentionally removed.
    void consume(const OperatorNode* node, JITContext& ctx,
                 const OperatorNode* sender,
                 const std::vector<std::string>& active_vars);

    std::string generateCode() const;
    /** @brief Return physical operations/contracts independently of C++ source emission. */
    PlannedQuery buildPhysicalPlan() const;

private:
    JITContext&    ctx_;
    const Catalog& catalog_;

    // Build-side pending info (populated while visiting left of HashJoin).
    // Helper for Vectorized Push Model
    void ensureLoaded(const std::string& col_name, JITContext& ctx) const;

    struct BuildInfo {
        std::string ht_name;       // "d_s_hash_table"
        /** @brief Separate UInt32 PHT presence map; zero keys remain ordinary data. */
        std::string occupancy_name;
        std::string ht_size_expr;  // "20000" or "2*20000"
        std::string key_mins;      // "0" or "1"
        uint8_t     variant;       // 1 = keys only (PHT_1), 2 = key-value pairs (PHT_2)
        bool        use_mht;       // true = Multi-value HT (two-pass), false = Perfect HT
        bool        payload_is_row_id = false; // PHT_2 payload is build-side row_id, not a value column
        std::vector<std::string> fk_cols; // FK column(s) in the probe/fact table
        std::string val_col;       // payload column (variant 2 / MHT only)
        std::string row_id_reg;    // vector register name for build-side row_id payload
        std::string dim_table;     // build-side table name
        std::string dim_prefix;    // "s", "c", "p", "d"
        std::string size_macro;    // "S_LEN", "C_LEN", etc.
        std::string pk_col;        // PK column in the dim table
    };
    std::unordered_map<const OperatorNode*, BuildInfo> build_infos_;
    std::vector<std::unique_ptr<ExprNode>> expanded_projection_exprs_;
    std::set<std::string> agg_cols_;    // Columns required for aggregation (high priority for joins)
    std::set<std::string> filter_cols_; // Columns required for filtering (low priority for joins)
    void collectAllColumnsFromTree(const OperatorNode* node);

    // ---- Produce handlers (top-down) ----
    void produceTableScan (const TableScanNode*  node, JITContext& ctx);
    void produceFilter    (const FilterNode*     node, JITContext& ctx);
    void produceHashJoin  (const HashJoinNode*   node, JITContext& ctx);
    void produceAggregate (const AggregateNode*  node, JITContext& ctx);
    void produceProjection(const ProjectionNode* node, JITContext& ctx);
    void produceSortLimit (const SortLimitNode*  node, JITContext& ctx);

    // ---- Vector-mode consume handlers (block level, outside scalar loop) ----
    void consumeFilterVector   (const FilterNode*    node, JITContext& ctx,
                                const OperatorNode* sender,
                                const std::vector<std::string>& active_vars);
    void consumeHashJoinVector (const HashJoinNode*  node, JITContext& ctx,
                                const OperatorNode* sender,
                                const std::vector<std::string>& active_vars);
    void consumeAggregateVector(const AggregateNode* node, JITContext& ctx,
                                const OperatorNode* sender,
                                const std::vector<std::string>& active_vars);
    void consumeProjectionVector(const ProjectionNode* node, JITContext& ctx,
                                 const OperatorNode* sender,
                                 const std::vector<std::string>& active_vars);
    void consumeSortLimitVector(const SortLimitNode* node, JITContext& ctx,
                                const OperatorNode* sender,
                                const std::vector<std::string>& active_vars);

    // ---- Item-mode consume handlers (scalar level, inside scalar loop) ----
    void consumeFilterItem   (const FilterNode*    node, JITContext& ctx,
                              const OperatorNode* sender,
                              const std::vector<std::string>& active_vars);
    void consumeHashJoinItem (const HashJoinNode*  node, JITContext& ctx,
                              const OperatorNode* sender,
                              const std::vector<std::string>& active_vars);
    void consumeAggregateItem(const AggregateNode* node, JITContext& ctx,
                              const OperatorNode* sender,
                              const std::vector<std::string>& active_vars);
    void consumeProjectionItem(const ProjectionNode* node, JITContext& ctx,
                               const OperatorNode* sender,
                               const std::vector<std::string>& active_vars);
    void consumeSortLimitItem(const SortLimitNode* node, JITContext& ctx,
                              const OperatorNode* sender,
                              const std::vector<std::string>& active_vars);

    enum class ConsumeMode : uint8_t {
        Vector = 0,
        Item
    };

    ExecutionMode execution_mode_ = ExecutionMode::DataCentric;
    ConsumeMode   consume_mode_   = ConsumeMode::Vector;

    enum class ProjectionPass : uint8_t {
        None = 0,
        Count,
        Write,
        DirectDense
    };
    ProjectionPass projection_pass_ = ProjectionPass::None;

    // ---- Build kernel emitters ----
    // PHT: one fused build kernel with a separately planned scratch allocation.
    void emitPHTBuildKernel(const BuildInfo& bi,
                            const FilterNode* build_filter,
                            JITContext& ctx);

    // MHT: Count + five distinct prefix-sum launches + Write, with explicit storage/copies.
    void emitMHTBuildKernels(const BuildInfo& bi,
                             const FilterNode* build_filter,
                             JITContext& ctx);

    // ---- Helpers ----
    static std::string tablePrefix(const std::string& table_name);
    static std::string sizeMacro  (const std::string& table_name);

    // Derive hash-table sizing from catalog stats + join condition.
    BuildInfo computeBuildInfo(const std::string& dim_table,
                               const FilterNode*  filter,
                               const HashJoinNode* join_node) const;

    // Emit filter predicates (block-level BlockPred* calls) for a build kernel.
    // Uses `items` register, writes into `flags`.
    // Returns the column name that is currently loaded into `items` after filtering.
    std::string emitBuildFilter(const FilterNode* filter,
                                const std::string& dim_table,
                                std::stringstream& code,
                                JITContext& ctx) const;
};

} // namespace db
