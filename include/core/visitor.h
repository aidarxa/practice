#pragma once

#include "core/expressions.h"
#include "core/operators.h"
#include "core/memory.h"   // Catalog, TableMetadata, ColumnStatistics

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace db {

// ============================================================================
// JITContext — shared mutable state threaded through both visitors.
// ============================================================================
struct JITContext {
    std::stringstream includes_and_globals;
    std::stringstream build_kernels;  // All build-phase kernel bodies
    std::stringstream probe_kernel;   // Main select kernel body

    // column_name → register name in the *probe/select* kernel.
    // Set by JITOperatorVisitor when it emits a BlockLoad in the probe kernel.
    std::unordered_map<std::string, std::string> col_to_reg;

    // Which columns have already been BlockLoad-ed in the *current* kernel.
    // Reset at kernel boundaries.
    std::set<std::string> loaded_in_probe;

    // Counter for unique temporary OR-mask names ("mask_1", "mask_2", …)
    int mask_counter = 0;

    std::string current_items_col;
    
    // Counter used to generate unique kernel class names.
    int build_kernel_count = 0;

    // Ordered list of kernel class names (for forward declarations).
    std::vector<std::string> kernel_class_names;

    // Hash table info accumulated during build phase generation
    // (name → code to allocate/zero in the outer function).
    struct HashTableInfo {
        std::string name;        // "d_s_hash_table"
        std::string type;        // "int"
        std::string size_expr;   // "S_LEN" or "2*S_LEN"
    };
    std::vector<HashTableInfo> hash_tables;

    // Columns that must be fetched from ctx (EXTERNAL_INPUT):
    std::set<std::string> external_columns; // e.g. "d_lo_orderdate"

    // Result buffer size expression (set by AggregateNode visitor)
    std::string result_size_expr;   // e.g. "7*1000"
    int         tuple_size = 1;

    // ---------- helpers ----------
    std::string getNewMask() {
        return "mask_" + std::to_string(++mask_counter);
    }

    std::string getNewBuildKernelName(const std::string& prefix) {
        ++build_kernel_count;
        std::string name = "build_hashtable_" + prefix;
        kernel_class_names.push_back(name);
        return name;
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
    void visit(const BinaryExpr&      node) override;

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
// JITOperatorVisitor — macro-level: walks the Operator Tree and emits
// full SYCL kernel bodies into JITContext.
//
// HashJoinNode:  left  (build-side / dimension) → separate build kernel
//                right (probe-side / fact)       → continues in probe kernel
// ============================================================================
class JITOperatorVisitor : public OperatorVisitor {
public:
    explicit JITOperatorVisitor(JITContext& ctx, const Catalog& catalog);

    void visit(const TableScanNode& node) override;
    void visit(const FilterNode&    node) override;
    void visit(const HashJoinNode&  node) override;
    void visit(const AggregateNode& node) override;

    // Assembles the complete execute_query() source file from JITContext.
    std::string generateCode() const;

private:
    JITContext&    ctx_;
    const Catalog& catalog_;

    // Points to whichever stream is currently being written into.
    std::stringstream* active_stream_;

    // Whether we are currently generating inside a build kernel.
    bool in_build_kernel_ = false;

    // Track which columns have been loaded in the CURRENT build kernel.
    std::set<std::string> loaded_in_build_;

    // first_pred flag shared within the probe kernel's filter chain.
    bool probe_first_pred_ = true;

    // Build-side pending info (populated while visiting left of HashJoin).
    struct BuildInfo {
        std::string ht_name;       // "d_s_hash_table"
        std::string ht_size_expr;  // "20000" or "2*20000"
        std::string key_mins;      // "0" or "1"
        uint8_t     variant;       // 1 = keys only, 2 = key-value pairs
        std::vector<std::string> fk_cols; // <--- ИЗМЕНЕНО с std::string fk_col;
        std::string val_col;       // value column (variant 2 only)
        std::string dim_prefix;    // "s", "c", "p", "d"
        std::string size_macro;    // "S_LEN", "C_LEN", etc.
    };
    std::vector<BuildInfo> build_infos_;

    // Helpers
    static std::string tablePrefix(const std::string& table_name);
    static std::string sizeMacro  (const std::string& table_name);

    // Generate a complete build kernel for one dimension.
    // Writes into ctx_.build_kernels.
    void emitBuildKernel(const std::string& dim_table,
                         const FilterNode*  filter,
                         const TableScanNode& scan);

    // Derive hash-table sizing from catalog stats.
    BuildInfo computeBuildInfo(const std::string& dim_table,
                               const FilterNode*  filter) const;
};

} // namespace db
