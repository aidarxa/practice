#include "core/optimizer_rules.h"

#include <algorithm>
#include <cassert>

namespace db {

// ============================================================================
// Helper: extractTableNames
// ============================================================================
static void extractTableNamesImpl(const ExprNode* expr,
                                   std::set<std::string>& out) {
    if (!expr) return;
    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            if (!col->table_name.empty())
                out.insert(col->table_name);
            break;
        }
        case ExprType::OP_AND:
        case ExprType::OP_OR:
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE:
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            extractTableNamesImpl(bin->left.get(), out);
            extractTableNamesImpl(bin->right.get(), out);
            break;
        }
        default:
            break;
    }
}

std::set<std::string> extractTableNames(const ExprNode* expr) {
    std::set<std::string> result;
    extractTableNamesImpl(expr, result);
    return result;
}

// ============================================================================
// Helper: collectTableNames
// ============================================================================
static void collectTableNamesImpl(const OperatorNode* node,
                                   std::set<std::string>& out) {
    if (!node) return;
    if (node->getType() == OperatorType::TABLE_SCAN) {
        const auto* scan = static_cast<const TableScanNode*>(node);
        out.insert(scan->table_name);
    }
    for (const auto& child : node->getChildren())
        collectTableNamesImpl(child.get(), out);
}

std::set<std::string> collectTableNames(const OperatorNode* node) {
    std::set<std::string> result;
    collectTableNamesImpl(node, result);
    return result;
}

// ============================================================================
// Helper: flattenAndExpr
// ============================================================================
void flattenAndExpr(std::unique_ptr<ExprNode>& root,
                    std::vector<std::unique_ptr<ExprNode>>& out) {
    if (!root) return;
    if (root->getType() == ExprType::OP_AND) {
        auto* bin = static_cast<BinaryExpr*>(root.get());
        // Flatten children first (pre-order is fine here — we just split AND chains)
        flattenAndExpr(bin->left, out);
        flattenAndExpr(bin->right, out);
        // root itself is consumed — its left/right have been moved into `out`
        root.reset();
    } else {
        out.push_back(std::move(root));
    }
}

// ============================================================================
// Helper: buildAndChain
// ============================================================================
std::unique_ptr<ExprNode> buildAndChain(
        std::vector<std::unique_ptr<ExprNode>>& preds) {
    assert(!preds.empty());
    if (preds.size() == 1) return std::move(preds[0]);
    // Left-associative: fold left
    auto result = std::move(preds[0]);
    for (std::size_t i = 1; i < preds.size(); ++i) {
        result = std::make_unique<BinaryExpr>(
            ExprType::OP_AND,
            std::move(result),
            std::move(preds[i]));
    }
    return result;
}

// ============================================================================
// Helper: check all elements of `subset` exist in `superset`
// ============================================================================
static bool allIn(const std::set<std::string>& subset,
                  const std::set<std::string>& superset) {
    for (const auto& s : subset) {
        if (superset.find(s) == superset.end()) return false;
    }
    return true;
}

// ============================================================================
// PredicatePushdownRule::apply
// ============================================================================
void PredicatePushdownRule::apply(std::unique_ptr<OperatorNode>& root) {
    applyRecursive(root);
}

void PredicatePushdownRule::applyRecursive(std::unique_ptr<OperatorNode>& node) {
    if (!node) return;

    // Post-order: process children first so inner joins are optimised
    // before we try to push predicates at the current level.
    for (auto& child : node->getChildrenMutable())
        applyRecursive(child);

    tryPushdown(node);
}

void PredicatePushdownRule::tryPushdown(std::unique_ptr<OperatorNode>& node) {
    // Only applies when: FilterNode → HashJoinNode
    if (!node || node->getType() != OperatorType::FILTER) return;

    auto* filter = static_cast<FilterNode*>(node.get());
    if (filter->getChildren().empty() || !filter->predicate) return;

    auto& child_ref = filter->getChildrenMutable()[0];
    if (!child_ref || child_ref->getType() != OperatorType::HASH_JOIN) return;

    auto* join = static_cast<HashJoinNode*>(child_ref.get());
    auto& join_children = join->getChildrenMutable();
    if (join_children.size() < 2) return;

    auto& left_child  = join_children[0];
    auto& right_child = join_children[1];

    const auto left_tables  = collectTableNames(left_child.get());
    const auto right_tables = collectTableNames(right_child.get());

    // ---- Flatten AND predicate into individual conditions ----
    std::vector<std::unique_ptr<ExprNode>> conditions;
    flattenAndExpr(filter->predicate, conditions);

    // ---- Route each condition ----
    std::vector<std::unique_ptr<ExprNode>> left_preds;
    std::vector<std::unique_ptr<ExprNode>> right_preds;
    std::vector<std::unique_ptr<ExprNode>> remaining_preds;

    for (auto& cond : conditions) {
        if (!cond) continue;
        const auto tables = extractTableNames(cond.get());

        if (!tables.empty() && allIn(tables, left_tables)) {
            left_preds.push_back(std::move(cond));
        } else if (!tables.empty() && allIn(tables, right_tables)) {
            right_preds.push_back(std::move(cond));
        } else {
            remaining_preds.push_back(std::move(cond));
        }
    }

    // If nothing could be pushed, restore and return
    if (left_preds.empty() && right_preds.empty()) {
        // Rebuild predicate from remaining (all conditions)
        filter->predicate = buildAndChain(remaining_preds);
        return;
    }

    // ---- Insert FilterNodes for pushed predicates ----
    if (!left_preds.empty()) {
        auto lf = std::make_unique<FilterNode>(buildAndChain(left_preds));
        lf->addChild(std::move(left_child));
        left_child = std::move(lf);
        // Re-apply pushdown to the newly created FilterNode
        applyRecursive(left_child);
    }

    if (!right_preds.empty()) {
        auto rf = std::make_unique<FilterNode>(buildAndChain(right_preds));
        rf->addChild(std::move(right_child));
        right_child = std::move(rf);
        // Re-apply pushdown to the newly created FilterNode
        applyRecursive(right_child);
    }

    // ---- Remove or update the current FilterNode ----
    if (remaining_preds.empty()) {
        // All predicates pushed → remove the FilterNode entirely
        node = std::move(child_ref);
    } else {
        // Some predicates remain here
        filter->predicate = buildAndChain(remaining_preds);
    }
}

// ============================================================================
// Optimizer
// ============================================================================
Optimizer::Optimizer() {
    rules_.push_back(std::make_unique<PredicatePushdownRule>());
}

std::unique_ptr<OperatorNode> Optimizer::optimize(
        std::unique_ptr<OperatorNode> plan) {
    for (auto& rule : rules_)
        rule->apply(plan);
    return plan;
}

} // namespace db
