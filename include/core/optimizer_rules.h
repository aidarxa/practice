#pragma once

#include "operators.h"
#include "expressions.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace db {

// ============================================================================
// IOptimizationRule — base interface for every optimization rule.
// apply() takes the tree root by reference so the rule can:
//   • restructure the tree in place, or
//   • completely replace the root (e.g. remove a FilterNode).
// ============================================================================
class IOptimizationRule {
public:
    virtual ~IOptimizationRule() = default;
    virtual void apply(std::unique_ptr<OperatorNode>& root_node) = 0;
};

// ============================================================================
// Helpers (used both by the rule and by the JIT visitor)
// ============================================================================

/// Returns the set of all table_name strings referenced by ColumnRefExpr leaves.
/// Entries with empty table_name are ignored (unqualified columns).
std::set<std::string> extractTableNames(const ExprNode* expr);

/// Returns the set of all table_name strings from every TableScanNode
/// in the subtree rooted at `node`.
std::set<std::string> collectTableNames(const OperatorNode* node);

/// Recursively flatten a tree of OP_AND nodes into a flat list.
/// Ownership of the leaves is transferred into `out`.
void flattenAndExpr(std::unique_ptr<ExprNode>& root,
                    std::vector<std::unique_ptr<ExprNode>>& out);

/// Build an AND chain from a non-empty list (left-associative).
/// Ownership of list elements is consumed.
std::unique_ptr<ExprNode> buildAndChain(
    std::vector<std::unique_ptr<ExprNode>>& preds);

// ============================================================================
// PredicatePushdownRule
//
// Post-order traversal.  At each FilterNode whose sole child is a
// HashJoinNode, the rule:
//   1. Flattens the predicate into individual conditions (splitting AND).
//   2. Routes each condition to the left (build) or right (probe) branch
//      of the HashJoin based on which tables the condition references.
//   3. Conditions that cannot be pushed (span both branches) stay at the
//      current FilterNode.  If all conditions are pushed, the FilterNode
//      is removed.
// ============================================================================
class PredicatePushdownRule : public IOptimizationRule {
public:
    void apply(std::unique_ptr<OperatorNode>& root_node) override;

private:
    // Recursive post-order worker.
    void applyRecursive(std::unique_ptr<OperatorNode>& node);

    // Attempt a single pushdown at `node` (which must be a FilterNode).
    void tryPushdown(std::unique_ptr<OperatorNode>& node);
};

// ============================================================================
// Optimizer — registers rules and applies them sequentially.
// ============================================================================
class Optimizer {
    std::vector<std::unique_ptr<IOptimizationRule>> rules_;
public:
    /// Constructor registers all built-in rules (PredicatePushdownRule).
    Optimizer();

    /// Optimise the plan in place and return the (potentially new) root.
    std::unique_ptr<OperatorNode> optimize(std::unique_ptr<OperatorNode> plan);
};

} // namespace db
