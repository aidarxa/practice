#include "core/visitor.h"
#include "core/optimizer_rules.h" // extractTableNames, collectTableNames

#include <algorithm>
#include <cassert>
#include <cctype>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

// ============================================================================
// crystal/utils.h helpers (getTableName, lookup, etc.) are used for
// mapping column names to tables and generating device-pointer names.
// We include it here for table prefix / size macro logic.
// ============================================================================
#include "crystal/utils.h"

namespace db {

// ============================================================================
// Static helpers
// ============================================================================

static std::string tablePrefix(const std::string& table_name) {
    if (table_name == "LINEORDER") return "lo";
    if (table_name == "SUPPLIER")  return "s";
    if (table_name == "CUSTOMER")  return "c";
    if (table_name == "PART")      return "p";
    if (table_name == "DDATE")     return "d";
    return "";
}

static std::string sizeMacroFor(const std::string& table_name) {
    if (table_name == "LINEORDER") return "LO_LEN";
    if (table_name == "SUPPLIER")  return "S_LEN";
    if (table_name == "CUSTOMER")  return "C_LEN";
    if (table_name == "PART")      return "P_LEN";
    if (table_name == "DDATE")     return "D_LEN";
    return "0";
}

enum class FilterPredicateSupport {
    FastPathSupported,
    NeedsUniversalPath,
    Unsupported
};

static const char* exprTypeName(ExprType t) {
    switch (t) {
        case ExprType::COLUMN_REF:    return "COLUMN_REF";
        case ExprType::LITERAL_INT:   return "LITERAL_INT";
        case ExprType::LITERAL_FLOAT: return "LITERAL_FLOAT";
        case ExprType::OP_AND:        return "OP_AND";
        case ExprType::OP_OR:         return "OP_OR";
        case ExprType::OP_NOT:        return "OP_NOT";
        case ExprType::OP_EQ:         return "OP_EQ";
        case ExprType::OP_NEQ:        return "OP_NEQ";
        case ExprType::OP_LT:         return "OP_LT";
        case ExprType::OP_LTE:        return "OP_LTE";
        case ExprType::OP_GT:         return "OP_GT";
        case ExprType::OP_GTE:        return "OP_GTE";
        case ExprType::OP_ADD:        return "OP_ADD";
        case ExprType::OP_SUB:        return "OP_SUB";
        case ExprType::OP_MUL:        return "OP_MUL";
        case ExprType::OP_DIV:        return "OP_DIV";
        case ExprType::OP_IS_NULL:    return "OP_IS_NULL";
        case ExprType::OP_IS_NOT_NULL:return "OP_IS_NOT_NULL";
        case ExprType::STAR:          return "STAR";
        default:                      return "UNKNOWN_EXPR_TYPE";
    }
}


static LogicalType resultLogicalTypeForExpr(const ExprNode* expr) {
    if (!expr) return LogicalType::UInt64;
    switch (expr->getType()) {
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE:
        case ExprType::OP_AND:
        case ExprType::OP_OR:
        case ExprType::OP_NOT:
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL:
            return LogicalType::UInt64;
        case ExprType::LITERAL_FLOAT:
            return LogicalType::Float64;
        default:
            return LogicalType::Int64;
    }
}

static LogicalType resultLogicalTypeForAggregateOutput(const AggregateNode* node, int visible_col) {
    if (!node) return LogicalType::UInt64;
    const int group_cols = static_cast<int>(node->group_by_exprs.size());
    if (visible_col < group_cols) {
        return resultLogicalTypeForExpr(node->group_by_exprs[static_cast<std::size_t>(visible_col)].get());
    }
    const int agg_idx = visible_col - group_cols;
    if (agg_idx >= 0 && agg_idx < static_cast<int>(node->aggregates.size())) {
        const auto& agg = node->aggregates[static_cast<std::size_t>(agg_idx)];
        if (agg.isAvg()) return LogicalType::Float64;
        if (agg.isCount()) return LogicalType::UInt64;
        return LogicalType::UInt64;
    }
    return LogicalType::UInt64;
}


static bool catalogColumnNullable(const Catalog& catalog, const std::string& col_name);

static bool resultColumnNullableForExpr(const ExprNode* expr, const Catalog& catalog) {
    if (!expr) return true;
    switch (expr->getType()) {
        case ExprType::STAR:
        case ExprType::LITERAL_INT:
        case ExprType::LITERAL_FLOAT:
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL:
            return false;
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            return catalogColumnNullable(catalog, col->column_name);
        }
        case ExprType::OP_NOT: {
            const auto* b = static_cast<const BinaryExpr*>(expr);
            return resultColumnNullableForExpr(b->left.get(), catalog);
        }
        case ExprType::OP_AND:
        case ExprType::OP_OR:
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV:
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* b = static_cast<const BinaryExpr*>(expr);
            return resultColumnNullableForExpr(b->left.get(), catalog) || resultColumnNullableForExpr(b->right.get(), catalog);
        }
        default:
            return true;
    }
}

static bool resultColumnNullableForAggregateOutput(const AggregateNode* node, int visible_col, const Catalog& catalog) {
    if (!node) return true;
    const int group_cols = static_cast<int>(node->group_by_exprs.size());
    if (visible_col < group_cols) {
        return resultColumnNullableForExpr(node->group_by_exprs[static_cast<std::size_t>(visible_col)].get(), catalog);
    }
    const int agg_idx = visible_col - group_cols;
    if (agg_idx >= 0 && agg_idx < static_cast<int>(node->aggregates.size())) {
        const auto& agg = node->aggregates[static_cast<std::size_t>(agg_idx)];
        if (agg.isCount()) return false;
        return resultColumnNullableForExpr(agg.agg_expr.get(), catalog);
    }
    return true;
}

static std::string resultColumnPointerType(LogicalType type) {
    switch (type) {
        case LogicalType::Float64: return "double*";
        case LogicalType::Int64:   return "std::int64_t*";
        case LogicalType::UInt64:  return "std::uint64_t*";
    }
    return "std::uint64_t*";
}

static std::string resultColumnGetter(LogicalType type) {
    switch (type) {
        case LogicalType::Float64: return "getResultColumnFloat64Pointer";
        case LogicalType::Int64:   return "getResultColumnInt64Pointer";
        case LogicalType::UInt64:  return "getResultColumnUInt64Pointer";
    }
    return "getResultColumnUInt64Pointer";
}

static std::string castToResultColumnType(const std::string& value, LogicalType type) {
    switch (type) {
        case LogicalType::Float64: return "static_cast<double>(" + value + ")";
        case LogicalType::Int64:   return "static_cast<std::int64_t>(" + value + ")";
        case LogicalType::UInt64:  return "static_cast<std::uint64_t>(" + value + ")";
    }
    return "static_cast<std::uint64_t>(" + value + ")";
}

static bool isComparisonOp(ExprType t) {
    return t >= ExprType::OP_EQ && t <= ExprType::OP_GTE;
}

static bool isBinaryExprType(ExprType t) {
    return t == ExprType::OP_AND || t == ExprType::OP_OR || t == ExprType::OP_NOT ||
           t == ExprType::OP_EQ  || t == ExprType::OP_NEQ ||
           t == ExprType::OP_LT  || t == ExprType::OP_LTE ||
           t == ExprType::OP_GT  || t == ExprType::OP_GTE ||
           t == ExprType::OP_ADD || t == ExprType::OP_SUB ||
           t == ExprType::OP_MUL || t == ExprType::OP_DIV ||
           t == ExprType::OP_IS_NULL || t == ExprType::OP_IS_NOT_NULL;
}

static FilterPredicateSupport validateFilterFastPathPredicate(
        const ExprNode* expr,
        std::string& error_message) {
    if (!expr) return FilterPredicateSupport::FastPathSupported;

    if (expr->getType() == ExprType::OP_AND) {
        const auto* bin = static_cast<const BinaryExpr*>(expr);
        FilterPredicateSupport left = validateFilterFastPathPredicate(bin->left.get(), error_message);
        if (left == FilterPredicateSupport::Unsupported) return left;
        FilterPredicateSupport right = validateFilterFastPathPredicate(bin->right.get(), error_message);
        if (right == FilterPredicateSupport::Unsupported) return right;
        if (left == FilterPredicateSupport::NeedsUniversalPath ||
            right == FilterPredicateSupport::NeedsUniversalPath) {
            return FilterPredicateSupport::NeedsUniversalPath;
        }
        return FilterPredicateSupport::FastPathSupported;
    }

    if (expr->getType() == ExprType::OP_OR || expr->getType() == ExprType::OP_NOT) {
        return FilterPredicateSupport::NeedsUniversalPath;
    }

    if (expr->getType() == ExprType::OP_IS_NULL ||
        expr->getType() == ExprType::OP_IS_NOT_NULL) {
        return FilterPredicateSupport::NeedsUniversalPath;
    }

    if (!isComparisonOp(expr->getType())) {
        error_message = std::string("Unsupported filter predicate node: ") + exprTypeName(expr->getType());
        return FilterPredicateSupport::Unsupported;
    }

    const auto* cmp = static_cast<const BinaryExpr*>(expr);
    if (!cmp->left || !cmp->right) {
        error_message = std::string("Malformed comparison predicate (null child), op=") + exprTypeName(cmp->op_type);
        return FilterPredicateSupport::Unsupported;
    }

    const ExprType left_type = cmp->left->getType();
    const ExprType right_type = cmp->right->getType();

    const bool col_vs_int = left_type == ExprType::COLUMN_REF && right_type == ExprType::LITERAL_INT;
    const bool col_vs_col = left_type == ExprType::COLUMN_REF && right_type == ExprType::COLUMN_REF;
    if (col_vs_int) {
        return FilterPredicateSupport::FastPathSupported;
    }
    if (col_vs_col) {
        return FilterPredicateSupport::NeedsUniversalPath;
    }

    const bool has_math = left_type == ExprType::OP_ADD || left_type == ExprType::OP_SUB ||
                          left_type == ExprType::OP_MUL || left_type == ExprType::OP_DIV ||
                          right_type == ExprType::OP_ADD || right_type == ExprType::OP_SUB ||
                          right_type == ExprType::OP_MUL || right_type == ExprType::OP_DIV;
    if (has_math) {
        return FilterPredicateSupport::NeedsUniversalPath;
    }

    const bool has_literal_float = left_type == ExprType::LITERAL_FLOAT || right_type == ExprType::LITERAL_FLOAT;
    if (has_literal_float) {
        error_message = std::string("Unsupported float literal in fast-path predicate, op=") +
                        exprTypeName(cmp->op_type) + ", left=" + exprTypeName(left_type) +
                        ", right=" + exprTypeName(right_type);
        return FilterPredicateSupport::Unsupported;
    }

    error_message = std::string("Unsupported comparison shape in fast-path predicate, op=") +
                    exprTypeName(cmp->op_type) + ", left=" + exprTypeName(left_type) +
                    ", right=" + exprTypeName(right_type);
    return FilterPredicateSupport::Unsupported;
}



static std::string combineAndTerms(const std::vector<std::string>& terms) {
    std::string out;
    for (std::string term : terms) {
        term.erase(term.begin(), std::find_if(term.begin(), term.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        term.erase(std::find_if(term.rbegin(), term.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), term.end());
        if (term.empty() || term == "1" || term == "true" || term == "(1)" || term == "(true)") continue;
        if (!out.empty()) out += " && ";
        out += term;
    }
    return out.empty() ? "1" : out;
}

static std::string makeValidComparisonCondition(const std::string& left_valid,
                                                const std::string& right_valid,
                                                const std::string& comparison_call) {
    return "(" + combineAndTerms({left_valid, right_valid, comparison_call}) + ")";
}

static bool isCatalogUniqueBuildKey(const Catalog& catalog,
                                    const std::string& table_name,
                                    const std::string& col_name) {
    try {
        const auto& meta = catalog.getTableMetadata(table_name);
        if (meta.isColumnPrimaryKey(col_name) || meta.isColumnUnique(col_name)) {
            return true;
        }
        if (meta.hasColumnStats(col_name)) {
            const auto& stats = meta.getColumnStats(col_name);
            if (stats.cardinality_ > 0 && stats.cardinality_ == meta.getSize()) {
                return true;
            }
        }
    } catch (...) {
        return false;
    }
    return false;
}


static void collectExpressionColumns(const ExprNode* e, std::vector<std::string>& cols);

static bool catalogColumnNullable(const Catalog& catalog, const std::string& col_name) {
    try {
        const std::string table_name = getTableName(col_name);
        const auto& meta = catalog.getTableMetadata(table_name);
        return meta.isColumnNullable(col_name);
    } catch (...) {
        return false;
    }
}

static void markColumnNullability(JITContext& ctx, const Catalog& catalog, const std::string& col_name) {
    if (col_name.empty()) return;
    if (catalogColumnNullable(catalog, col_name)) ctx.nullable_columns.insert(col_name);
}

static void markExpressionNullability(JITContext& ctx, const Catalog& catalog, const ExprNode* expr) {
    std::vector<std::string> cols;
    collectExpressionColumns(expr, cols);
    for (const auto& col : cols) markColumnNullability(ctx, catalog, col);
}

static void markColumnSetNullability(JITContext& ctx, const Catalog& catalog, const std::set<std::string>& cols) {
    for (const auto& col : cols) markColumnNullability(ctx, catalog, col);
}

static std::string validRegFor(const std::string& reg_name) {
    return reg_name + "_valid";
}

static std::string nullBitmapSymbolFor(const std::string& col_name) {
    return "n_" + col_name;
}

static void emitLoadOrGatherIntoReg(const std::string& col_name,
                                    const std::string& reg_name,
                                    JITContext& ctx,
                                    std::stringstream& stream) {
    auto mapped = ctx.col_to_reg.find(col_name);
    if (mapped != ctx.col_to_reg.end() && mapped->second == reg_name) return;

    const std::string table_name = getTableName(col_name);
    auto rid = ctx.table_rowid_regs.find(table_name);
    const bool nullable = ctx.nullable_columns.count(col_name) != 0;
    ctx.external_columns.insert("d_" + col_name);
    ctx.col_to_reg[col_name] = reg_name;
    if (nullable) {
        ctx.external_null_columns.insert(nullBitmapSymbolFor(col_name));
        ctx.col_to_valid_reg[col_name] = validRegFor(reg_name);
    } else {
        ctx.col_to_valid_reg.erase(col_name);
    }

    if (rid != ctx.table_rowid_regs.end() && !rid->second.empty()) {
        stream << "            BlockGather<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
               << "d_" << col_name << ", tid, " << rid->second
               << ", flags, " << reg_name << ", num_tile_items);\n";
        if (nullable) {
            stream << "            BlockGatherValidity<BLOCK_THREADS, ITEMS_PER_THREAD>("
                   << nullBitmapSymbolFor(col_name) << ", tid, " << rid->second
                   << ", flags, " << validRegFor(reg_name) << ", num_tile_items);\n";
        }
    } else {
        stream << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
               << "d_" << col_name << " + tile_offset, tid, tile_offset, "
               << reg_name << ", num_tile_items);\n";
        if (nullable) {
            stream << "            BlockLoadValidity<BLOCK_THREADS, ITEMS_PER_THREAD>("
                   << nullBitmapSymbolFor(col_name) << ", tid, tile_offset, "
                   << validRegFor(reg_name) << ", num_tile_items);\n";
        }
    }
}

static void collectExpressionColumns(const ExprNode* e, std::vector<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        const auto* col = static_cast<const ColumnRefExpr*>(e);
        if (std::find(cols.begin(), cols.end(), col->column_name) == cols.end()) cols.push_back(col->column_name);
        return;
    }
    if (isBinaryExprType(e->getType())) {
        const auto* b = static_cast<const BinaryExpr*>(e);
        if (b->left) collectExpressionColumns(b->left.get(), cols);
        if (b->right) collectExpressionColumns(b->right.get(), cols);
    }
}

static std::string expressionValueNoEmit(const ExprNode* e, const JITContext& ctx);
static std::string expressionValidExpr(const ExprNode* e, const JITContext& ctx);

static bool isLiteralTrue(const std::string& v) {
    return v == "1" || v == "true" || v == "(1)" || v == "(true)";
}

static bool isLiteralFalse(const std::string& v) {
    return v == "0" || v == "false" || v == "(0)" || v == "(false)";
}

static std::string booleanAndValue(const std::string& a, const std::string& b) {
    if (isLiteralFalse(a) || isLiteralFalse(b)) return "0";
    if (isLiteralTrue(a)) return b;
    if (isLiteralTrue(b)) return a;
    return "(" + a + " && " + b + ")";
}

static std::string booleanOrValue(const std::string& a, const std::string& b) {
    if (isLiteralTrue(a) || isLiteralTrue(b)) return "1";
    if (isLiteralFalse(a)) return b;
    if (isLiteralFalse(b)) return a;
    return "(" + a + " || " + b + ")";
}

static std::string booleanNotValue(const std::string& a) {
    if (isLiteralTrue(a)) return "0";
    if (isLiteralFalse(a)) return "1";
    return "(!(" + a + "))";
}

static std::string expressionValueNoEmit(const ExprNode* e, const JITContext& ctx) {
    if (!e) return "0";
    switch (e->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(e);
            auto it = ctx.col_to_reg.find(col->column_name);
            const std::string reg = (it == ctx.col_to_reg.end() || it->second.empty()) ? col->column_name : it->second;
            return reg + "[i]";
        }
        case ExprType::LITERAL_INT:
            return std::to_string(static_cast<const LiteralIntExpr*>(e)->value);
        case ExprType::LITERAL_FLOAT:
            return std::to_string(static_cast<const LiteralFloatExpr*>(e)->value);
        case ExprType::STAR:
            return "1";
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string l = expressionValueNoEmit(b->left.get(), ctx);
            const std::string r = expressionValueNoEmit(b->right.get(), ctx);
            switch (e->getType()) {
                case ExprType::OP_ADD: return "db::safe_add(" + l + ", " + r + ")";
                case ExprType::OP_SUB: return "db::safe_sub(" + l + ", " + r + ")";
                case ExprType::OP_MUL: return "db::safe_mul(" + l + ", " + r + ")";
                case ExprType::OP_DIV: return "db::safe_div(" + l + ", " + r + ")";
                default: break;
            }
            return "0";
        }
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string l = expressionValueNoEmit(b->left.get(), ctx);
            const std::string r = expressionValueNoEmit(b->right.get(), ctx);
            switch (e->getType()) {
                case ExprType::OP_EQ:  return "db::safe_eq(" + l + ", " + r + ")";
                case ExprType::OP_NEQ: return "db::safe_neq(" + l + ", " + r + ")";
                case ExprType::OP_LT:  return "db::safe_lt(" + l + ", " + r + ")";
                case ExprType::OP_LTE: return "db::safe_lte(" + l + ", " + r + ")";
                case ExprType::OP_GT:  return "db::safe_gt(" + l + ", " + r + ")";
                case ExprType::OP_GTE: return "db::safe_gte(" + l + ", " + r + ")";
                default: break;
            }
            return "0";
        }
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string v = expressionValidExpr(b->left.get(), ctx);
            if (isLiteralTrue(v)) return e->getType() == ExprType::OP_IS_NULL ? "0" : "1";
            if (isLiteralFalse(v)) return e->getType() == ExprType::OP_IS_NULL ? "1" : "0";
            return e->getType() == ExprType::OP_IS_NULL ? "(!(" + v + "))" : "(" + v + ")";
        }
        case ExprType::OP_NOT: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanNotValue(expressionValueNoEmit(b->left.get(), ctx));
        }
        case ExprType::OP_AND: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanAndValue(expressionValueNoEmit(b->left.get(), ctx), expressionValueNoEmit(b->right.get(), ctx));
        }
        case ExprType::OP_OR: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanOrValue(expressionValueNoEmit(b->left.get(), ctx), expressionValueNoEmit(b->right.get(), ctx));
        }
    }
    return "0";
}

static std::string expressionValidExpr(const ExprNode* e, const JITContext& ctx) {
    if (!e) return "1";
    switch (e->getType()) {
        case ExprType::STAR:
        case ExprType::LITERAL_INT:
        case ExprType::LITERAL_FLOAT:
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL:
            return "1";
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(e);
            auto it = ctx.col_to_valid_reg.find(col->column_name);
            if (it == ctx.col_to_valid_reg.end() || it->second.empty()) return "1";
            return it->second + "[i]";
        }
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV:
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return combineAndTerms({expressionValidExpr(b->left.get(), ctx), expressionValidExpr(b->right.get(), ctx)});
        }
        case ExprType::OP_NOT: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return expressionValidExpr(b->left.get(), ctx);
        }
        case ExprType::OP_AND: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string lv = expressionValidExpr(b->left.get(), ctx);
            const std::string rv = expressionValidExpr(b->right.get(), ctx);
            const std::string lval = expressionValueNoEmit(b->left.get(), ctx);
            const std::string rval = expressionValueNoEmit(b->right.get(), ctx);
            // SQL 3VL: AND is known if either side is known FALSE, or both sides are known.
            const std::string left_false_known = combineAndTerms({lv, booleanNotValue(lval)});
            const std::string right_false_known = combineAndTerms({rv, booleanNotValue(rval)});
            const std::string both_known = combineAndTerms({lv, rv});
            return booleanOrValue(booleanOrValue(left_false_known, right_false_known), both_known);
        }
        case ExprType::OP_OR: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string lv = expressionValidExpr(b->left.get(), ctx);
            const std::string rv = expressionValidExpr(b->right.get(), ctx);
            const std::string lval = expressionValueNoEmit(b->left.get(), ctx);
            const std::string rval = expressionValueNoEmit(b->right.get(), ctx);
            // SQL 3VL: OR is known if either side is known TRUE, or both sides are known.
            const std::string left_true_known = combineAndTerms({lv, lval});
            const std::string right_true_known = combineAndTerms({rv, rval});
            const std::string both_known = combineAndTerms({lv, rv});
            return booleanOrValue(booleanOrValue(left_true_known, right_true_known), both_known);
        }
    }
    return "1";
}


// Item-mode expression helpers are defined near consumeProjectionItem; forward
// declarations are needed for scalar filter/MHT consumers above that point.
static std::string itemExprValue(const ExprNode* e, JITContext& ctx);
static std::string itemExprValid(const ExprNode* e, JITContext& ctx);
static std::pair<std::string, uint64_t> generatePerfectHashItem(
        const std::vector<std::unique_ptr<ExprNode>>& group_by,
        const Catalog& catalog,
        JITContext& ctx);

// ============================================================================
// JITExprVisitor — constructor
// ============================================================================

JITExprVisitor::JITExprVisitor(JITContext& ctx,
                               std::stringstream& stream,
                               const std::string& target_mask,
                               bool is_or_context,
                               bool* first_pred)
    : ctx_(ctx), stream_(stream), target_mask_(target_mask),
      is_or_context_(is_or_context), first_pred_(first_pred),
      own_first_pred_(false)
{
    if (!first_pred_) {
        own_first_pred_ = true;
        first_pred_ = &local_first_pred_storage_;
    }
}

// ============================================================================
// JITExprVisitor — leaf visitors (ColumnRef, LiteralInt, LiteralFloat)
// These are only called directly by BinaryExpr::accept when the node is
// a leaf — they don't emit code on their own.
// ============================================================================

void JITExprVisitor::visit(const ColumnRefExpr& /*node*/) {
    // No standalone code generation — handled inside visitComparison
}

void JITExprVisitor::visit(const LiteralIntExpr& /*node*/) {
    // No standalone code generation — value used in visitComparison
}

void JITExprVisitor::visit(const LiteralFloatExpr& /*node*/) {
    // No standalone code generation
}

void JITExprVisitor::visit(const StarExpr& /*node*/) {
    // Star is handled by aggregate/projection-specific code.
}

// ============================================================================
// JITExprVisitor — predSuffix
// ============================================================================

const char* JITExprVisitor::predSuffix(ExprType t) {
    switch (t) {
        case ExprType::OP_EQ:  return "Eq";
        case ExprType::OP_NEQ: return "NEq";
        case ExprType::OP_LT:  return "LT";
        case ExprType::OP_LTE: return "LTE";
        case ExprType::OP_GT:  return "GT";
        case ExprType::OP_GTE: return "GTE";
        default: return "Eq";
    }
}

// ============================================================================
// JITExprVisitor — ensureLoaded
// ============================================================================
/*
void JITExprVisitor::ensureLoaded(const std::string& col, const std::string& reg) {
    // Check if column already loaded in probe context
    if (ctx_.loaded_in_probe.count(col)) return;
    ctx_.loaded_in_probe.insert(col);

    // Register external column for ctx buffer extraction
    ctx_.external_columns.insert("d_" + col);

    stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
            << "d_" << col << " + tile_offset, tid, tile_offset, "
            << reg << ", num_tile_items);\n";
}
*/
// ============================================================================
// JITExprVisitor — translateInlineExpr
// ============================================================================

std::string JITExprVisitor::translateInlineExpr(const ExprNode* expr, bool is_probe) {
    if (!expr) return "";

    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            std::string col_name = col->column_name;

            if (is_probe) {
                auto mapped = ctx_.col_to_reg.find(col_name);
                if (mapped != ctx_.col_to_reg.end() && !mapped->second.empty()) {
                    return mapped->second + "[i]";
                }
                emitLoadOrGatherIntoReg(col_name, col_name, ctx_, stream_);
                return ctx_.col_to_reg[col_name] + "[i]";
            }
            return col_name + "[i]";
        }
        case ExprType::LITERAL_INT:
            return std::to_string(static_cast<const LiteralIntExpr*>(expr)->value);
        case ExprType::LITERAL_FLOAT:
            return std::to_string(static_cast<const LiteralFloatExpr*>(expr)->value);
        case ExprType::STAR:
            return "1";
            
        case ExprType::OP_ADD:
            return "db::safe_add(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_SUB:
            return "db::safe_sub(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_MUL:
            return "db::safe_mul(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_DIV:
            return "db::safe_div(" + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->left.get(), is_probe) + ", " 
                                   + translateInlineExpr(static_cast<const BinaryExpr*>(expr)->right.get(), is_probe) + ")";
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            const std::string l = translateInlineExpr(bin->left.get(), is_probe);
            const std::string r = translateInlineExpr(bin->right.get(), is_probe);
            switch (expr->getType()) {
                case ExprType::OP_EQ:  return "db::safe_eq(" + l + ", " + r + ")";
                case ExprType::OP_NEQ: return "db::safe_neq(" + l + ", " + r + ")";
                case ExprType::OP_LT:  return "db::safe_lt(" + l + ", " + r + ")";
                case ExprType::OP_LTE: return "db::safe_lte(" + l + ", " + r + ")";
                case ExprType::OP_GT:  return "db::safe_gt(" + l + ", " + r + ")";
                case ExprType::OP_GTE: return "db::safe_gte(" + l + ", " + r + ")";
                default: break;
            }
            return "0";
        }
        case ExprType::OP_AND: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            return booleanAndValue(translateInlineExpr(bin->left.get(), is_probe),
                                   translateInlineExpr(bin->right.get(), is_probe));
        }
        case ExprType::OP_OR: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            return booleanOrValue(translateInlineExpr(bin->left.get(), is_probe),
                                  translateInlineExpr(bin->right.get(), is_probe));
        }
        case ExprType::OP_NOT: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            return booleanNotValue(translateInlineExpr(bin->left.get(), is_probe));
        }
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            (void)translateInlineExpr(bin->left.get(), is_probe); // ensure operand columns are loaded
            const std::string valid = expressionValidExpr(bin->left.get(), ctx_);
            if (isLiteralTrue(valid)) {
                return (expr->getType() == ExprType::OP_IS_NULL) ? "0" : "1";
            }
            if (isLiteralFalse(valid)) {
                return (expr->getType() == ExprType::OP_IS_NULL) ? "1" : "0";
            }
            if (expr->getType() == ExprType::OP_IS_NULL) {
                return "(!(" + valid + "))";
            }
            return "(" + valid + ")";
        }
        default: 
            return "";
    }
}

// ============================================================================
// JITExprVisitor — visitComparison
// ============================================================================

void JITExprVisitor::visitComparison(const BinaryExpr& node) {
    // ===== ВЕТКА: IS NULL / IS NOT NULL =====
    if (node.op_type == ExprType::OP_IS_NULL || node.op_type == ExprType::OP_IS_NOT_NULL) {
        const std::string condition = translateInlineExpr(&node, true);

        // Constant-fold IS NULL / IS NOT NULL over non-nullable expressions.
        // This avoids generating no-op predicates such as `flags[i] = flags[i] && (1)`
        // and correctly handles impossible predicates such as `non_nullable_col IS NULL`.
        const bool condition_true = (condition == "1" || condition == "true" || condition == "(1)" || condition == "(true)");
        const bool condition_false = (condition == "0" || condition == "false" || condition == "(0)" || condition == "(false)");
        if (condition_true && !is_or_context_) {
            return;
        }
        if (condition_true && is_or_context_) {
            stream_ << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << target_mask_ << ");\n";
            *first_pred_ = false;
            return;
        }
        if (condition_false && is_or_context_) {
            return;
        }
        if (condition_false && !is_or_context_) {
            stream_ << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << target_mask_ << ");\n";
            *first_pred_ = false;
            return;
        }

        stream_ << "            #pragma unroll\n"
                << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n"
                << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        if (is_or_context_) {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] || " << condition << ";\n";
            }
        } else {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] && " << condition << ";\n";
            }
        }
        stream_ << "                }\n            }\n";
        return;
    }

    // ===== ВЕТКА: COLUMN OP COLUMN =====
    if (node.left && node.left->getType() == ExprType::COLUMN_REF &&
        node.right && node.right->getType() == ExprType::COLUMN_REF) {

        const auto* left_col  = static_cast<const ColumnRefExpr*>(node.left.get());
        const auto* right_col = static_cast<const ColumnRefExpr*>(node.right.get());
        const std::string col1_name = left_col->column_name;
        const std::string col2_name = right_col->column_name;

        auto load_col_if_needed = [&](const std::string& cname, const std::string& reg) {
            if (ctx_.col_to_reg.count(cname) && !ctx_.col_to_reg[cname].empty()) return;
            emitLoadOrGatherIntoReg(cname, reg, ctx_, stream_);
        };

        load_col_if_needed(col1_name, col1_name);
        load_col_if_needed(col2_name, col2_name);

        const std::string reg1 = ctx_.col_to_reg.count(col1_name) ? ctx_.col_to_reg[col1_name] : col1_name;
        const std::string reg2 = ctx_.col_to_reg.count(col2_name) ? ctx_.col_to_reg[col2_name] : col2_name;
        const std::string valid1 = ctx_.col_to_valid_reg.count(col1_name) ? ctx_.col_to_valid_reg[col1_name] + "[i]" : "1";
        const std::string valid2 = ctx_.col_to_valid_reg.count(col2_name) ? ctx_.col_to_valid_reg[col2_name] + "[i]" : "1";

        std::string func;
        switch(node.op_type) {
            case ExprType::OP_LT:  func = "db::safe_lt"; break;
            case ExprType::OP_GT:  func = "db::safe_gt"; break;
            case ExprType::OP_LTE: func = "db::safe_lte"; break;
            case ExprType::OP_GTE: func = "db::safe_gte"; break;
            case ExprType::OP_EQ:  func = "db::safe_eq"; break;
            case ExprType::OP_NEQ: func = "db::safe_neq"; break;
            default: return;
        }
        const std::string condition = makeValidComparisonCondition(valid1, valid2, func + "(" + reg1 + "[i], " + reg2 + "[i])");

        stream_ << "            #pragma unroll\n"
                << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n"
                << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        if (is_or_context_) {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] || " << condition << ";\n";
            }
        } else {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] && " << condition << ";\n";
            }
        }
        stream_ << "                }\n            }\n";
        return;
    }

    // ===== ВЕТКА: COLUMN OP LITERAL =====
    bool is_build = false; // push-model: always probe context

    // ВЕТКА: Probe Phase Inline Math
    if (!is_build) {
        std::string left_expr = translateInlineExpr(node.left.get(), true);
        std::string right_expr = translateInlineExpr(node.right.get(), true);
        
        std::string func;
        switch(node.op_type) {
            case ExprType::OP_LT:  func = "db::safe_lt"; break;
            case ExprType::OP_GT:  func = "db::safe_gt"; break;
            case ExprType::OP_LTE: func = "db::safe_lte"; break;
            case ExprType::OP_GTE: func = "db::safe_gte"; break;
            case ExprType::OP_EQ:  func = "db::safe_eq"; break;
            case ExprType::OP_NEQ: func = "db::safe_neq"; break;
            default: return;
        }

        std::string condition = makeValidComparisonCondition(
            expressionValidExpr(node.left.get(), ctx_),
            expressionValidExpr(node.right.get(), ctx_),
            func + "(" + left_expr + ", " + right_expr + ")");

        stream_ << "            #pragma unroll\n"
                << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n"
                << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        
        if (is_or_context_) {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] || " << condition << ";\n";
            }
        } else {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] && " << condition << ";\n";
            }
        }
        
        stream_ << "                }\n            }\n";
        return;
    }

    // ВЕТКА: Build Phase (COLUMN OP LITERAL ONLY)
    const ExprNode* col_node = nullptr;
    const ExprNode* lit_node = nullptr;

    if (node.left && node.left->getType() == ExprType::COLUMN_REF && node.right && (node.right->getType() == ExprType::LITERAL_INT || node.right->getType() == ExprType::LITERAL_FLOAT)) {
        col_node = node.left.get(); lit_node = node.right.get();
    } else if (node.right && node.right->getType() == ExprType::COLUMN_REF && node.left && (node.left->getType() == ExprType::LITERAL_INT || node.left->getType() == ExprType::LITERAL_FLOAT)) {
        col_node = node.right.get(); lit_node = node.left.get();
    } else { return; }

    const auto* col = static_cast<const ColumnRefExpr*>(col_node);
    std::string col_name = col->column_name;

    std::string lit_value;
    if (lit_node->getType() == ExprType::LITERAL_INT) {
        lit_value = std::to_string(static_cast<const LiteralIntExpr*>(lit_node)->value);
    } else { lit_value = std::to_string(static_cast<const LiteralFloatExpr*>(lit_node)->value); }

    std::string reg_name = "items";

    // ОПТИМИЗАЦИЯ: Избегаем повторных загрузок в items
    if (ctx_.current_items_col != col_name) {
        ctx_.external_columns.insert("d_" + col_name);
        stream_ << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                << "d_" << col_name << " + tile_offset, tid, tile_offset, "
                << reg_name << ", num_tile_items);\n";
        ctx_.current_items_col = col_name; // Запоминаем!
    }

    std::string prefix;
    if (is_or_context_) {
        prefix = *first_pred_ ? "BlockPred" : "BlockPredOr";
        if (*first_pred_) *first_pred_ = false;
    } else {
        prefix = *first_pred_ ? "BlockPred" : "BlockPredA";
        if (*first_pred_) *first_pred_ = false;
    }

    stream_ << "            " << prefix << predSuffix(node.op_type)
            << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " << reg_name << ", "
            << target_mask_ << ", " << lit_value << ", num_tile_items);\n";
}
// ============================================================================
// JITExprVisitor — visit(BinaryExpr)
// ============================================================================

void JITExprVisitor::visit(const BinaryExpr& node) {
    if (node.op_type == ExprType::OP_AND) {
        if (!is_or_context_) {
            node.left->accept(*this);
            node.right->accept(*this);
        } else {
            // AND внутри OR: создаем новую маску и применяем ее
            std::string new_mask = ctx_.getNewMask();
            stream_ << "            int " << new_mask << "[ITEMS_PER_THREAD];\n";
            stream_ << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << new_mask << ");\n";
            
            bool first_pred = true;
            JITExprVisitor and_visitor(ctx_, stream_, new_mask, false, &first_pred);
            node.left->accept(and_visitor);
            node.right->accept(and_visitor);
            
            stream_ << "            BlockApplyMaskOr<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " 
                    << target_mask_ << ", " << new_mask << ");\n";
            if (first_pred_ && *first_pred_) *first_pred_ = false;
        }
    } else if (node.op_type == ExprType::OP_OR) {
        if (is_or_context_) {
            // OR внутри OR: Сплющиваем! Не создаем новую маску, пишем напрямую в target_mask_
            node.left->accept(*this);
            node.right->accept(*this);
        } else {
            // OR внутри AND: выделяем один накопитель для всей цепочки OR
            std::string new_mask = ctx_.getNewMask();
            stream_ << "            int " << new_mask << "[ITEMS_PER_THREAD];\n";
            stream_ << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << new_mask << ");\n";
            
            bool first_pred = true;
            JITExprVisitor or_visitor(ctx_, stream_, new_mask, true, &first_pred);
            node.left->accept(or_visitor);
            node.right->accept(or_visitor);
            
            stream_ << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " 
                    << target_mask_ << ", " << new_mask << ");\n";
            if (first_pred_ && *first_pred_) *first_pred_ = false;
        }
    } else if (node.op_type == ExprType::OP_NOT) {
        const std::string value = translateInlineExpr(&node, true);
        const std::string valid = expressionValidExpr(&node, ctx_);
        const std::string condition = combineAndTerms({valid, value});
        if (isLiteralTrue(condition) && !is_or_context_) return;
        if (isLiteralFalse(condition) && !is_or_context_) {
            stream_ << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << target_mask_ << ");\n";
            if (first_pred_) *first_pred_ = false;
            return;
        }
        stream_ << "            #pragma unroll\n"
                << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n"
                << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
        if (is_or_context_) {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] || " << condition << ";\n";
            }
        } else {
            if (*first_pred_) {
                stream_ << "                    " << target_mask_ << "[i] = " << condition << ";\n";
                *first_pred_ = false;
            } else {
                stream_ << "                    " << target_mask_ << "[i] = " << target_mask_ << "[i] && " << condition << ";\n";
            }
        }
        stream_ << "                }\n            }\n";
    } else {
        visitComparison(node);
    }
}

// ============================================================================
// ============================================================================
// JITOperatorVisitor — static helpers
// ============================================================================

std::string JITOperatorVisitor::tablePrefix(const std::string& table_name) {
    return db::tablePrefix(table_name);
}

std::string JITOperatorVisitor::sizeMacro(const std::string& table_name) {
    return sizeMacroFor(table_name);
}

// Helper to extract columns for a table from an expression tree
static void extractColumnsForTable(const ExprNode* e, const std::string& table_name, std::vector<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        const auto* col = static_cast<const ColumnRefExpr*>(e);
        if (getTableName(col->column_name) == table_name) {
            if (std::find(cols.begin(), cols.end(), col->column_name) == cols.end()) {
                cols.push_back(col->column_name);
            }
        }
    } 
    // Безопасный каст: только если это действительно бинарный оператор (тип >= OP_AND)
    else if (isBinaryExprType(e->getType())) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractColumnsForTable(bin->left.get(), table_name, cols);
        if (bin->right) extractColumnsForTable(bin->right.get(), table_name, cols);
    }
}

// Recursive helper to find all columns needed by the whole tree for a specific table
static void findAllColumnsForTable(const OperatorNode* node, const std::string& table_name, std::vector<std::string>& cols) {
    if (!node) return;
    
    if (node->getType() == OperatorType::FILTER) {
        const auto* filter = static_cast<const FilterNode*>(node);
        extractColumnsForTable(filter->predicate.get(), table_name, cols);
    } else if (node->getType() == OperatorType::HASH_JOIN) {
        const auto* join = static_cast<const HashJoinNode*>(node);
        extractColumnsForTable(join->join_condition.get(), table_name, cols);
    } else if (node->getType() == OperatorType::AGGREGATE) {
        const auto* agg = static_cast<const AggregateNode*>(node);
        for (const auto& g : agg->group_by_exprs) {
            extractColumnsForTable(g.get(), table_name, cols);
        }
        for (const auto& a : agg->aggregates) {
            extractColumnsForTable(a.agg_expr.get(), table_name, cols);
        }
    } else if (node->getType() == OperatorType::PROJECTION) {
        const auto* proj = static_cast<const ProjectionNode*>(node);
        for (const auto& e : proj->select_exprs) {
            extractColumnsForTable(e.get(), table_name, cols);
        }
    }
    
    // We don't recurse into HashJoin's left child (build side) if we are scanning fact table (probe side),
    // but the columns collected here are globally scoped by table_name anyway.
    for (const auto& child : node->getChildren()) {
        findAllColumnsForTable(child.get(), table_name, cols);
    }
}

// ============================================================================
// JITOperatorVisitor — constructor
// ============================================================================

JITOperatorVisitor::JITOperatorVisitor(JITContext& ctx, const Catalog& catalog)
    : ctx_(ctx), catalog_(catalog) {
    assert(execution_mode_ == ExecutionMode::DataCentric && "Only canonical Data-Centric mode is supported");
    assert(consume_mode_ == ConsumeMode::Vector && "Initial consume mode must be Vector");
}

// ============================================================================
// visit() — Legacy OperatorVisitor interface.
// The root is always AggregateNode; execution.cpp calls accept(visitor) on it.
// visit(AggregateNode) is the single entry point that kicks off produce().
// The other three stubs exist only to satisfy the pure-virtual contract.
// ============================================================================

static void repairOperatorParents(const OperatorNode* node) {
    std::function<void(const OperatorNode*, OperatorNode*)> repair = [&](const OperatorNode* current, OperatorNode* parent) {
        if (!current) return;
        const_cast<OperatorNode*>(current)->parent_ = parent;
        for (const auto& child : current->getChildren()) {
            repair(child.get(), const_cast<OperatorNode*>(current));
        }
    };
    repair(node, nullptr);
}

void JITOperatorVisitor::visit(const AggregateNode& node) {
    repairOperatorParents(&node);

    // 2. СБОР ВСЕХ КОЛОНОК: Обходим все дерево один раз для надежного обнаружения требований
    agg_cols_.clear();
    filter_cols_.clear();
    collectAllColumnsFromTree(&node);
    markColumnSetNullability(ctx_, catalog_, agg_cols_);
    markColumnSetNullability(ctx_, catalog_, filter_cols_);

    // 3. Старт генерации конвейера
    produce(&node, ctx_);
}

void JITOperatorVisitor::visit(const ProjectionNode& node) {
    repairOperatorParents(&node);
    agg_cols_.clear();
    filter_cols_.clear();
    collectAllColumnsFromTree(&node);
    markColumnSetNullability(ctx_, catalog_, agg_cols_);
    markColumnSetNullability(ctx_, catalog_, filter_cols_);
    produce(&node, ctx_);
}

void JITOperatorVisitor::visit(const TableScanNode& /*node*/) {}
void JITOperatorVisitor::visit(const FilterNode&    /*node*/) {}
void JITOperatorVisitor::visit(const HashJoinNode&  /*node*/) {}

// ============================================================================
// Main Push-Model Dispatchers
// ============================================================================

void JITOperatorVisitor::produce(const OperatorNode* node, JITContext& ctx) {
    // Дерево уже починено в visit(AggregateNode), поэтому здесь просто диспетчеризация
    if (node->getType() == OperatorType::TABLE_SCAN) produceTableScan(static_cast<const TableScanNode*>(node), ctx);
    else if (node->getType() == OperatorType::FILTER) produceFilter(static_cast<const FilterNode*>(node), ctx);
    else if (node->getType() == OperatorType::HASH_JOIN) produceHashJoin(static_cast<const HashJoinNode*>(node), ctx);
    else if (node->getType() == OperatorType::AGGREGATE) produceAggregate(static_cast<const AggregateNode*>(node), ctx);
    else if (node->getType() == OperatorType::PROJECTION) produceProjection(static_cast<const ProjectionNode*>(node), ctx);
}

void JITOperatorVisitor::consume(const OperatorNode* node, JITContext& ctx, const OperatorNode* sender, const std::vector<std::string>& active_vars) {
    if (!node) return;
    assert(execution_mode_ == ExecutionMode::DataCentric && "Legacy execution mode is not supported");
    if (consume_mode_ == ConsumeMode::Vector) {
        consumeVector(node, ctx, sender, active_vars);
    } else if (consume_mode_ == ConsumeMode::Item) {
        consumeItem(node, ctx, sender, active_vars);
    } else {
        assert(false && "Unknown consume mode");
    }
}

// ============================================================================
// Handlers
// ============================================================================

void JITOperatorVisitor::produceTableScan(const TableScanNode* node, JITContext& ctx) {
    const std::string size_macro = sizeMacro(node->table_name);

    // Collect all columns needed from this table (traversing upward through parents)
    const OperatorNode* root = node;
    while (root->parent_) root = root->parent_;

    std::vector<std::string> active_vars;
    findAllColumnsForTable(root, node->table_name, active_vars);

    const bool exact_projection = root && root->getType() == OperatorType::PROJECTION;

    auto emit_one_scan_pipeline = [&](const std::string& pipeline_name,
                                      ProjectionPass projection_pass) {
        ctx.startNewPipeline(pipeline_name);
        projection_pass_ = projection_pass;

        // A scan pipeline owns its vector-register scope.  Build kernels and
        // external column declarations remain accumulated in ctx, but per-scan
        // register bindings must not leak from Count pass to Write pass.
        ctx.col_to_reg.clear();
        ctx.col_to_valid_reg.clear();
        ctx.table_rowid_regs.clear();
        ctx.loaded_in_probe.clear();
        ctx.current_items_col.clear();

        // 1. Kernel launch header.
        std::stringstream header;
        header << "    q.submit([&](sycl::handler& h) {\n";
        header << "        int num_tiles = (" << size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
        header << "        h.parallel_for<class " << ctx.current_pipeline->kernel_name << ">"
               << "(sycl::nd_range<1>(num_tiles * BLOCK_THREADS, BLOCK_THREADS), [=](sycl::nd_item<1> it) {\n";
        header << "            int tid = it.get_local_linear_id();\n";
        header << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
        header << "            int num_tile_items = (it.get_group_linear_id() == it.get_group_range(0) - 1) ? "
               << size_macro << " - tile_offset : TILE_SIZE;\n";

        // 2. Capture pipeline body emitted by consumeVector.
        std::string existing_body = ctx.current_pipeline->kernel_body.str();
        ctx.current_pipeline->kernel_body.str("");

        if (node->parent_) {
            consume_mode_ = ConsumeMode::Vector;
            consume(node->parent_, ctx, node, active_vars);
        }
        std::string pipeline_logic = ctx.current_pipeline->kernel_body.str();

        // 3. Register declarations discovered during pipeline generation.
        std::stringstream declarations;
        declarations << "            int items[ITEMS_PER_THREAD];\n";
        declarations << "            int flags[ITEMS_PER_THREAD];\n";
        declarations << "            int items2[ITEMS_PER_THREAD];\n";

        std::set<std::string> emitted;
        for (const auto& pair : ctx.col_to_reg) {
            const std::string& reg = pair.second;
            if (reg != "items" && reg != "flags" && reg != "items2" && !emitted.count(reg)) {
                declarations << "            int " << reg << "[ITEMS_PER_THREAD];\n";
                emitted.insert(reg);
            }
        }
        for (const auto& pair : ctx.col_to_valid_reg) {
            const std::string& reg = pair.second;
            if (!reg.empty() && !emitted.count(reg)) {
                declarations << "            int " << reg << "[ITEMS_PER_THREAD];\n";
                emitted.insert(reg);
            }
        }
        declarations << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n\n";

        // 4. Stitch final kernel.
        ctx.current_pipeline->kernel_body.str("");
        ctx.current_pipeline->kernel_body << existing_body
                                          << header.str()
                                          << declarations.str()
                                          << pipeline_logic
                                          << "        });\n";
        ctx.current_pipeline->kernel_body << "    });\n\n";
    };

    if (exact_projection) {
        emit_one_scan_pipeline("Scan_" + node->table_name + "_Count", ProjectionPass::Count);
        emit_one_scan_pipeline("Scan_" + node->table_name + "_Write", ProjectionPass::Write);
    } else {
        emit_one_scan_pipeline("Scan_" + node->table_name, ProjectionPass::None);
    }

    projection_pass_ = ProjectionPass::None;
}

void JITOperatorVisitor::produceFilter(const FilterNode* node, JITContext& ctx) {
    if (!node->getChildren().empty()) {
        produce(node->getChildren()[0].get(), ctx);
    }
}


// ============================================================================
// loadIntoReg — helper for Vectorized Push Model to reuse registers
// ============================================================================
void JITOperatorVisitor::ensureLoaded(const std::string& col_name, JITContext& ctx) const {} // DEPRECATED

static void loadIntoReg(const std::string& col_name, const std::string& reg_name, JITContext& ctx) {
    auto mapped = ctx.col_to_reg.find(col_name);
    if (mapped != ctx.col_to_reg.end() && mapped->second == reg_name) {
        return;
    }
    auto& code = ctx.current_pipeline->kernel_body;
    emitLoadOrGatherIntoReg(col_name, reg_name, ctx, code);
}

// ============================================================================
// consumeVector / consumeItem — main dispatchers
// ============================================================================

void JITOperatorVisitor::consumeVector(const OperatorNode* node, JITContext& ctx,
                                       const OperatorNode* sender,
                                       const std::vector<std::string>& active_vars) {
    if (!node) return;
    assert(execution_mode_ == ExecutionMode::DataCentric && "Vector consume is only valid in Data-Centric mode");
    assert(consume_mode_ == ConsumeMode::Vector && "Invalid transition into vector consume");
    switch (node->getType()) {
        case OperatorType::FILTER:
            consumeFilterVector(static_cast<const FilterNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::HASH_JOIN:
            consumeHashJoinVector(static_cast<const HashJoinNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::AGGREGATE:
            consumeAggregateVector(static_cast<const AggregateNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::PROJECTION:
            consumeProjectionVector(static_cast<const ProjectionNode*>(node), ctx, sender, active_vars); break;
        default: break;
    }
}

void JITOperatorVisitor::consumeItem(const OperatorNode* node, JITContext& ctx,
                                     const OperatorNode* sender,
                                     const std::vector<std::string>& active_vars) {
    if (!node) return;
    assert(execution_mode_ == ExecutionMode::DataCentric && "Item consume is only valid in Data-Centric mode");
    assert(consume_mode_ == ConsumeMode::Item && "consumeItem requires item mode");
    switch (node->getType()) {
        case OperatorType::FILTER:
            consumeFilterItem(static_cast<const FilterNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::HASH_JOIN:
            consumeHashJoinItem(static_cast<const HashJoinNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::AGGREGATE:
            consumeAggregateItem(static_cast<const AggregateNode*>(node), ctx, sender, active_vars); break;
        case OperatorType::PROJECTION:
            consumeProjectionItem(static_cast<const ProjectionNode*>(node), ctx, sender, active_vars); break;
        default: break;
    }
}

// ============================================================================
// consumeFilterVector — BlockPred* via JITExprVisitor, then consumeVector
// ============================================================================
void JITOperatorVisitor::consumeFilterVector(const FilterNode* node, JITContext& ctx,
                                              const OperatorNode* /*sender*/,
                                              const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (node->predicate) {
        std::string validation_error;
        const FilterPredicateSupport support =
            validateFilterFastPathPredicate(node->predicate.get(), validation_error);
        if (support == FilterPredicateSupport::Unsupported) {
            throw std::runtime_error("Filter predicate is not translatable before JIT C++ compilation: " + validation_error);
        }

        if (support == FilterPredicateSupport::NeedsUniversalPath) {
            // `flags` already contains the incoming pipeline mask (for example,
            // successful hash-join probes).  A top-level filter must refine that
            // mask, not overwrite it.  Start with first_pred=false so the first
            // generated predicate uses BlockPredA* / `flags && condition`.
            bool fp = false;
            JITExprVisitor expr_vis(ctx, code, "flags", false, &fp);
            node->predicate->accept(expr_vis);
        } else {
        // Recursive lambda to handle AND chains and simple binary comparisons
        std::function<bool(const ExprNode*, bool&)> process_expr = [&](const ExprNode* expr, bool& first) -> bool {
            if (!expr) return true;
            if (expr->getType() == ExprType::OP_AND) {
                const auto* bin = static_cast<const BinaryExpr*>(expr);
                bool ok1 = process_expr(bin->left.get(), first);
                bool ok2 = process_expr(bin->right.get(), first);
                return ok1 && ok2;
            } else if (expr->getType() >= ExprType::OP_EQ && expr->getType() <= ExprType::OP_GTE) {
                const auto* bin = static_cast<const BinaryExpr*>(expr);
                if (bin->left->getType() == ExprType::COLUMN_REF && bin->right->getType() == ExprType::LITERAL_INT) {
                    const auto* col = static_cast<const ColumnRefExpr*>(bin->left.get());
                    const auto* lit = static_cast<const LiteralIntExpr*>(bin->right.get());
                    
                    loadIntoReg(col->column_name, "items", ctx);
                    
                    std::string pred_macro;
                    switch(expr->getType()) {
                        case ExprType::OP_EQ:  pred_macro = first ? "BlockPredEq"  : "BlockPredAEq"; break;
                        case ExprType::OP_NEQ: pred_macro = first ? "BlockPredNEq" : "BlockPredANEq"; break;
                        case ExprType::OP_LT:  pred_macro = first ? "BlockPredLT"  : "BlockPredALT"; break;
                        case ExprType::OP_LTE: pred_macro = first ? "BlockPredLTE" : "BlockPredALTE"; break;
                        case ExprType::OP_GT:  pred_macro = first ? "BlockPredGT"  : "BlockPredAGT"; break;
                        case ExprType::OP_GTE: pred_macro = first ? "BlockPredGTE" : "BlockPredAGTE"; break;
                        default: break;
                    }
                    
                    code << "            " << pred_macro << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                         << "tid, items, flags, " << lit->value << ", num_tile_items);\n";
                    first = false;
                    return true;
                }
            }
            return false; // Complex expression, cannot optimize
        };
        
        // Same rule for the literal fast path: preserve the incoming mask from
        // scans/joins and only AND additional filter predicates into it.
        bool first_flag = false;
        bool fully_optimized = process_expr(node->predicate.get(), first_flag);
        if (!fully_optimized) {
            throw std::runtime_error("Internal fast-path mismatch after successful validation in consumeFilterVector");
        }
        }
    }
    if (node->parent_) {
        consume_mode_ = ConsumeMode::Vector;
        consume(node->parent_, ctx, node, active_vars);
    }
}

// ============================================================================
// consumeFilterItem — scalar inline condition inside expansion loop
// ============================================================================
void JITOperatorVisitor::consumeFilterItem(const FilterNode* node, JITContext& ctx,
                                            const OperatorNode* /*sender*/,
                                            const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (node->predicate) {
        std::string cond_val = itemExprValue(node->predicate.get(), ctx);
        std::string cond_valid = itemExprValid(node->predicate.get(), ctx);
        std::string cond = combineAndTerms({cond_valid, cond_val});
        code << "                        if (" << cond << ") {\n";
        if (node->parent_) {
            consume_mode_ = ConsumeMode::Item;
            consume(node->parent_, ctx, node, active_vars);
        }
        code << "                        }\n";
    } else {
        if (node->parent_) {
            consume_mode_ = ConsumeMode::Item;
            consume(node->parent_, ctx, node, active_vars);
        }
    }
}

void JITOperatorVisitor::produceAggregate(const AggregateNode* node, JITContext& ctx) {
    if (!node->getChildren().empty()) {
        produce(node->getChildren()[0].get(), ctx);
    }
}

void JITOperatorVisitor::produceProjection(const ProjectionNode* node, JITContext& ctx) {
    if (!node->getChildren().empty()) {
        produce(node->getChildren()[0].get(), ctx);
    }
}

static std::string translateMathExprPush(
        const ExprNode* expr,
        JITContext& ctx,
        bool cast_to_ull = false) {
    if (!expr) return "0";

    switch (expr->getType()) {
        case ExprType::COLUMN_REF: {
            const auto* col = static_cast<const ColumnRefExpr*>(expr);
            std::string reg = ctx.col_to_reg.count(col->column_name) ? ctx.col_to_reg[col->column_name] : col->column_name;
            std::string res = reg + "[i]";
            if (cast_to_ull) return "(unsigned long long)" + res;
            return res;
        }
        case ExprType::LITERAL_INT: {
            const auto* lit = static_cast<const LiteralIntExpr*>(expr);
            return std::to_string(lit->value);
        }
        case ExprType::LITERAL_FLOAT: {
            const auto* lit = static_cast<const LiteralFloatExpr*>(expr);
            return std::to_string(lit->value);
        }
        case ExprType::STAR:
            return cast_to_ull ? "(unsigned long long)1" : "1";
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            std::string left = translateMathExprPush(bin->left.get(), ctx, cast_to_ull);
            std::string right = translateMathExprPush(bin->right.get(), ctx, false);
            std::string op;
            switch (expr->getType()) {
                case ExprType::OP_ADD: op = " + "; break;
                case ExprType::OP_SUB: op = " - "; break;
                case ExprType::OP_MUL: op = " * "; break;
                case ExprType::OP_DIV: op = " / "; break;
                default: break;
            }
            return "(" + left + op + right + ")";
        }
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL: {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            const std::string valid = expressionValidExpr(bin->left.get(), ctx);
            std::string res;
            if (valid == "1" || valid == "true" || valid == "(1)" || valid == "(true)") {
                res = (expr->getType() == ExprType::OP_IS_NULL) ? "0" : "1";
            } else {
                res = (expr->getType() == ExprType::OP_IS_NULL) ? "(!(" + valid + "))" : "(" + valid + ")";
            }
            if (cast_to_ull) return "(unsigned long long)" + res;
            return res;
        }
        default:
            return "0";
    }
}

static std::string resolveLoadedRegOrThrow(const std::string& col_name, const JITContext& ctx) {
    auto it = ctx.col_to_reg.find(col_name);
    if (it == ctx.col_to_reg.end() || it->second.empty()) {
        throw std::runtime_error(
            "Column '" + col_name + "' is not loaded into register before aggregate codegen");
    }
    return it->second;
}

// ============================================================================
// generatePerfectHash — builds a minimal perfect hash expression string and
// total table size for a GROUP BY column list.
// Uses Horner's scheme:  hash = (...((col0 - min0) * card1 + (col1 - min1)) * card2 ...) % total
// ============================================================================
static std::pair<std::string, uint64_t> generatePerfectHash(
        const std::vector<std::unique_ptr<ExprNode>>& group_by,
        const Catalog& catalog,
        const JITContext& ctx) {
    std::string hash_expr;
    uint64_t total_size = 1;

    for (std::size_t i = 0; i < group_by.size(); ++i) {
        if (group_by[i]->getType() != ExprType::COLUMN_REF) continue;
        const auto* col = static_cast<const ColumnRefExpr*>(group_by[i].get());
        const std::string& col_name = col->column_name;
        const std::string  table_name = getTableName(col_name);

        uint64_t min_val = 0;
        uint64_t card    = 1;
        bool nullable = false;
        try {
            const auto& meta = catalog.getTableMetadata(table_name);
            nullable = meta.isColumnNullable(col_name);
            if (meta.hasColumnStats(col_name)) {
                const auto& stats = meta.getColumnStats(col_name);
                min_val = (uint64_t)stats.min_value_;
                card    = stats.cardinality_;
            }
        } catch (...) {}

        const std::string reg_name = resolveLoadedRegOrThrow(col_name, ctx);
        const auto valid_it = ctx.col_to_valid_reg.find(col_name);
        const std::string valid_expr = (valid_it == ctx.col_to_valid_reg.end() || valid_it->second.empty()) ? "1" : (valid_it->second + "[i]");
        std::string term;
        if (nullable) {
            // SQL GROUP BY puts all NULL values of the same key into one group.
            // Reserve code 0 for NULL, shift non-NULL values by +1.
            term = "((" + valid_expr + ") ? (" + reg_name + "[i] - " + std::to_string(min_val) + " + 1) : 0)";
            card += 1;
        } else {
            term = "(" + reg_name + "[i] - " + std::to_string(min_val) + ")";
        }
        if (hash_expr.empty()) {
            hash_expr = term;
        } else {
            hash_expr = "(" + hash_expr + " * " + std::to_string(card) + " + " + term + ")";
        }
        total_size *= card;
    }

    if (!hash_expr.empty()) {
        hash_expr = "(" + hash_expr + ") % " + std::to_string(total_size);
    } else {
        hash_expr = "0";
        total_size = 1;
    }
    return {hash_expr, total_size};
}

// ============================================================================
// consumeAggregateVector — opens scalar reduction loop, reduce_over_group + atomic
// ============================================================================
static void extractAllColumns(const ExprNode* e, std::vector<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        const auto* col = static_cast<const ColumnRefExpr*>(e);
        if (std::find(cols.begin(), cols.end(), col->column_name) == cols.end()) {
            cols.push_back(col->column_name);
        }
    } 
    else if (isBinaryExprType(e->getType())) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractAllColumns(bin->left.get(), cols);
        if (bin->right) extractAllColumns(bin->right.get(), cols);
    }
}

static void extractAllColumns(const ExprNode* e, std::set<std::string>& cols) {
    if (!e) return;
    if (e->getType() == ExprType::COLUMN_REF) {
        cols.insert(static_cast<const ColumnRefExpr*>(e)->column_name);
    } 
    else if (isBinaryExprType(e->getType())) {
        const auto* bin = static_cast<const BinaryExpr*>(e);
        if (bin->left) extractAllColumns(bin->left.get(), cols);
        if (bin->right) extractAllColumns(bin->right.get(), cols);
    }
}

static bool aggregateNeedsHiddenCount(const AggregateNode* node) {
    return node && node->needsHiddenCountSlot();
}

static int hiddenCountSlotForAggregate(const AggregateNode* node, std::size_t agg_idx) {
    if (!node || agg_idx >= node->aggregates.size() || !node->aggregates[agg_idx].needsNonNullCount()) {
        return -1;
    }
    int slot = static_cast<int>(node->visibleTupleSize());
    for (std::size_t i = 0; i < agg_idx; ++i) {
        if (node->aggregates[i].needsNonNullCount()) ++slot;
    }
    return slot;
}

static bool isTriviallyTrueExpr(const std::string& expr) {
    return expr == "1" || expr == "true" || expr == "(1)" || expr == "(true)";
}

static bool isTriviallyFalseExpr(const std::string& expr) {
    return expr == "0" || expr == "false" || expr == "(0)" || expr == "(false)";
}

static bool aggregateOutputsAllNonNullable(const AggregateNode* node, const Catalog& catalog) {
    if (!node) return false;
    const int visible_ts = static_cast<int>(node->visibleTupleSize());
    for (int c = 0; c < visible_ts; ++c) {
        if (resultColumnNullableForAggregateOutput(node, c, catalog)) return false;
    }
    return true;
}

static bool expressionHasKnownNonZeroDomain(const ExprNode* expr, const Catalog& catalog) {
    if (!expr || expr->getType() != ExprType::COLUMN_REF) return false;
    const auto* col = static_cast<const ColumnRefExpr*>(expr);
    try {
        const auto& meta = catalog.getTableMetadata(getTableName(col->column_name));
        if (!meta.hasColumnStats(col->column_name)) return false;
        return meta.getColumnStats(col->column_name).min_value_ != 0;
    } catch (...) {
        return false;
    }
}

static bool aggregateFastSparseOutputEligible(const AggregateNode* node, const Catalog& catalog) {
    if (!node) return false;
    if (node->needsHiddenCountSlot()) return false; // AVG requires finalization.
    if (!aggregateOutputsAllNonNullable(node, catalog)) return false;
    for (const auto& agg : node->aggregates) {
        if (!(agg.isCount() || agg.isSum())) return false;
    }
    if (node->group_by_exprs.empty()) return true;
    for (const auto& g : node->group_by_exprs) {
        if (expressionHasKnownNonZeroDomain(g.get(), catalog)) return true;
    }
    for (const auto& agg : node->aggregates) {
        if (agg.isCount()) return true;
    }
    return false;
}

static std::string aggregateInputExpr(const AggregateDef& agg, JITContext& ctx, bool cast_to_ull) {
    if (agg.isCount()) {
        return cast_to_ull ? "(unsigned long long)1" : "1";
    }
    return translateMathExprPush(agg.agg_expr.get(), ctx, cast_to_ull);
}

static void collectAggregateColumns(const AggregateNode* node, std::vector<std::string>& cols) {
    for (const auto& agg : node->aggregates) {
        if (agg.agg_expr && agg.agg_expr->getType() != ExprType::STAR) {
            extractAllColumns(agg.agg_expr.get(), cols);
        }
    }
    for (const auto& g : node->group_by_exprs) {
        extractAllColumns(g.get(), cols);
    }
}

static void collectAggregateColumns(const AggregateNode* node, std::set<std::string>& cols) {
    for (const auto& agg : node->aggregates) {
        if (agg.agg_expr && agg.agg_expr->getType() != ExprType::STAR) {
            extractAllColumns(agg.agg_expr.get(), cols);
        }
    }
    for (const auto& g : node->group_by_exprs) {
        extractAllColumns(g.get(), cols);
    }
}

static void emitAggregateInitializationIfNeeded(const AggregateNode* node,
                                                JITContext& ctx,
                                                uint64_t group_count,
                                                int visible_ts,
                                                int storage_ts) {
    bool has_min = false;
    for (const auto& agg : node->aggregates) has_min = has_min || agg.isMin();
    if (!has_min) return;

    const std::string kernel_name = "init_aggregate_slots";
    if (!ctx.emitted_auxiliary_kernels.insert(kernel_name).second) return;
    if (std::find(ctx.kernel_class_names.begin(), ctx.kernel_class_names.end(), kernel_name) == ctx.kernel_class_names.end()) {
        ctx.kernel_class_names.push_back(kernel_name);
    }

    auto& out = ctx.current_pipeline->includes_and_globals;
    out << "    q.submit([&](sycl::handler& h) {\n";
    out << "        h.parallel_for<class " << kernel_name << ">(sycl::range<1>(" << group_count
        << "), [=](sycl::id<1> gid) {\n";
    out << "            unsigned long long base = (unsigned long long)gid[0] * " << storage_ts << ";\n";
    int slot = (int)node->group_by_exprs.size();
    for (const auto& agg : node->aggregates) {
        if (agg.isMin()) {
            out << "            d_result[base + " << slot << "] = ~0ULL;\n";
        }
        ++slot;
    }
    out << "        });\n";
    out << "    });\n\n";
}

static void emitAggregateFinalizationIfNeeded(const AggregateNode* node,
                                              JITContext& ctx,
                                              uint64_t group_count,
                                              int visible_ts,
                                              int storage_ts,
                                              int /*hidden_count_slot*/) {
    const bool needs_hidden = aggregateNeedsHiddenCount(node);
    if (!needs_hidden) return;

    const std::string kernel_name = "finalize_aggregate_slots";
    if (!ctx.emitted_auxiliary_kernels.insert(kernel_name).second) return;
    if (std::find(ctx.kernel_class_names.begin(), ctx.kernel_class_names.end(), kernel_name) == ctx.kernel_class_names.end()) {
        ctx.kernel_class_names.push_back(kernel_name);
    }

    const uint64_t visible_size = (uint64_t)group_count * (uint64_t)visible_ts;
    const uint64_t visible_words = (visible_size + 63ULL) / 64ULL;
    auto& out = ctx.post_execution_code;
    out << "    unsigned long long* d_visible_result = sycl::malloc_device<unsigned long long>(" << visible_size << ", q);\n";
    out << "    uint64_t* d_visible_validity = sycl::malloc_device<uint64_t>(" << (visible_words == 0 ? 1 : visible_words) << ", q);\n";
    out << "    q.memset(d_visible_result, 0, " << visible_size << " * sizeof(unsigned long long));\n";
    out << "    q.memset(d_visible_validity, 0, " << (visible_words == 0 ? 1 : visible_words) << " * sizeof(uint64_t));\n";
    out << "    q.submit([&](sycl::handler& h) {\n";
    out << "        h.parallel_for<class " << kernel_name << ">(sycl::range<1>(" << group_count
        << "), [=](sycl::id<1> gid) {\n";
    out << "            unsigned long long g = (unsigned long long)gid[0];\n";
    out << "            unsigned long long src = g * " << storage_ts << ";\n";
    out << "            unsigned long long dst = g * " << visible_ts << ";\n";

    int slot = 0;
    for (const auto& gexpr : node->group_by_exprs) {
        (void)gexpr;
        out << "            d_visible_result[dst + " << slot << "] = d_result[src + " << slot << "];\n";
        out << "            if (db::bitmap_valid_at(d_result_validity, src + " << slot << ")) db::atomic_set_valid_bit(d_visible_validity, dst + " << slot << ");\n";
        ++slot;
    }

    for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
        const auto& agg = node->aggregates[a];
        if (agg.isCount()) {
            out << "            d_visible_result[dst + " << slot << "] = d_result[src + " << slot << "];\n";
            if (node->group_by_exprs.empty()) {
                out << "            db::atomic_set_valid_bit(d_visible_validity, dst + " << slot << ");\n";
            } else {
                out << "            if (d_result[src + " << slot << "] != 0ULL) db::atomic_set_valid_bit(d_visible_validity, dst + " << slot << ");\n";
            }
        } else {
            const int cnt_slot = hiddenCountSlotForAggregate(node, a);
            if (cnt_slot >= 0) {
                out << "            unsigned long long cnt_" << slot << " = d_result[src + " << cnt_slot << "];\n";
                out << "            if (cnt_" << slot << " != 0ULL) {\n";
                if (agg.isAvg()) {
                    out << "                double avg_value_" << slot << " = static_cast<double>(d_result[src + " << slot << "]) / static_cast<double>(cnt_" << slot << ");\n";
                    out << "                d_visible_result[dst + " << slot << "] = db::bit_cast_double_to_ull(avg_value_" << slot << ");\n";
                } else if (agg.isMin()) {
                    out << "                d_visible_result[dst + " << slot << "] = (d_result[src + " << slot << "] == ~0ULL) ? 0ULL : d_result[src + " << slot << "];\n";
                } else {
                    out << "                d_visible_result[dst + " << slot << "] = d_result[src + " << slot << "];\n";
                }
                out << "                db::atomic_set_valid_bit(d_visible_validity, dst + " << slot << ");\n";
                out << "            }\n";
            } else {
                if (agg.isMin()) {
                    out << "            d_visible_result[dst + " << slot << "] = (d_result[src + " << slot << "] == ~0ULL) ? 0ULL : d_result[src + " << slot << "];\n";
                } else {
                    out << "            d_visible_result[dst + " << slot << "] = d_result[src + " << slot << "];\n";
                }
                out << "            if (db::bitmap_valid_at(d_result_validity, src + " << slot << ")) db::atomic_set_valid_bit(d_visible_validity, dst + " << slot << ");\n";
            }
        }
        ++slot;
    }
    out << "        });\n";
    out << "    });\n";
    out << "    q.memcpy(d_result, d_visible_result, " << visible_size << " * sizeof(unsigned long long));\n";
    out << "    q.memcpy(d_result_validity, d_visible_validity, " << (visible_words == 0 ? 1 : visible_words) << " * sizeof(uint64_t));\n";
    out << "    q.wait();\n";
    out << "    sycl::free(d_visible_result, q);\n";
    out << "    sycl::free(d_visible_validity, q);\n";
    out << "    ctx->expected_result_size_ = " << visible_size << ";\n";
    out << "    ctx->expected_result_validity_words_ = " << (visible_words == 0 ? 1 : visible_words) << ";\n\n";
}


static void emitAggregateDenseColumnarMaterialization(const AggregateNode* node,
                                                     JITContext& ctx,
                                                     const Catalog& catalog,
                                                     uint64_t group_count,
                                                     int visible_ts) {
    if (!node || visible_ts <= 0) return;
    const std::string count_kernel = "AggregateDenseCount";
    const std::string write_kernel = "AggregateDenseWrite";
    const std::string marker = "aggregate_dense_columnar_materialization";
    if (!ctx.emitted_auxiliary_kernels.insert(marker).second) return;
    for (const auto& k : {count_kernel, write_kernel}) {
        if (std::find(ctx.kernel_class_names.begin(), ctx.kernel_class_names.end(), k) == ctx.kernel_class_names.end()) {
            ctx.kernel_class_names.push_back(k);
        }
    }

    auto& out = ctx.post_execution_code;
    out << "    {\n";
    out << "        const unsigned long long aggregate_group_count = " << group_count << "ULL;\n";
    out << "        const unsigned long long aggregate_tuple_size = " << visible_ts << "ULL;\n";
    out << "        unsigned long long* d_aggregate_counts = sycl::malloc_device<unsigned long long>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count, q);\n";
    out << "        unsigned long long* d_aggregate_offsets = sycl::malloc_device<unsigned long long>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count, q);\n";
    out << "        q.memset(d_aggregate_counts, 0, (aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count) * sizeof(unsigned long long));\n";
    out << "        q.memset(d_aggregate_offsets, 0, (aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count) * sizeof(unsigned long long));\n";
    out << "        q.submit([&](sycl::handler& h) {\n";
    out << "            h.parallel_for<class " << count_kernel << ">(sycl::range<1>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count), [=](sycl::id<1> gid) {\n";
    out << "                unsigned long long g = static_cast<unsigned long long>(gid[0]);\n";
    out << "                if (g >= aggregate_group_count) return;\n";
    if (node->group_by_exprs.empty()) {
        out << "                d_aggregate_counts[g] = 1ULL;\n";
    } else {
        out << "                bool present = false;\n";
        out << "                unsigned long long base = g * aggregate_tuple_size;\n";
        out << "                for (unsigned long long c = 0; c < aggregate_tuple_size; ++c) {\n";
        out << "                    if (db::bitmap_valid_at(d_result_validity, base + c) || d_result[base + c] != 0ULL) { present = true; break; }\n";
        out << "                }\n";
        out << "                d_aggregate_counts[g] = present ? 1ULL : 0ULL;\n";
    }
    out << "            });\n";
    out << "        });\n";
    out << "        std::vector<unsigned long long> h_aggregate_counts(static_cast<std::size_t>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count), 0ULL);\n";
    out << "        std::vector<unsigned long long> h_aggregate_offsets(static_cast<std::size_t>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count), 0ULL);\n";
    out << "        if (aggregate_group_count != 0ULL) {\n";
    out << "            q.memcpy(h_aggregate_counts.data(), d_aggregate_counts, static_cast<std::size_t>(aggregate_group_count) * sizeof(unsigned long long)).wait();\n";
    out << "        }\n";
    out << "        unsigned long long aggregate_row_count = 0ULL;\n";
    out << "        for (unsigned long long g = 0; g < aggregate_group_count; ++g) {\n";
    out << "            h_aggregate_offsets[static_cast<std::size_t>(g)] = aggregate_row_count;\n";
    out << "            aggregate_row_count += h_aggregate_counts[static_cast<std::size_t>(g)];\n";
    out << "        }\n";
    out << "        if (aggregate_group_count != 0ULL) {\n";
    out << "            q.memcpy(d_aggregate_offsets, h_aggregate_offsets.data(), static_cast<std::size_t>(aggregate_group_count) * sizeof(unsigned long long)).wait();\n";
    out << "        }\n";
    out << "        if (aggregate_row_count != 0ULL && aggregate_row_count > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max() / aggregate_tuple_size)) {\n";
    out << "            throw std::overflow_error(\"Aggregate dense result size overflow\");\n";
    out << "        }\n";
    out << "        const std::size_t aggregate_result_values = static_cast<std::size_t>(aggregate_row_count * aggregate_tuple_size);\n";
    out << "        const std::size_t aggregate_result_bytes = aggregate_result_values * sizeof(unsigned long long);\n";
    out << "        const std::size_t aggregate_validity_bytes = static_cast<std::size_t>(aggregate_tuple_size) * (((static_cast<std::size_t>(aggregate_row_count) + 63ULL) / 64ULL) == 0 ? 1ULL : ((static_cast<std::size_t>(aggregate_row_count) + 63ULL) / 64ULL)) * sizeof(uint64_t);\n";
    out << "        const std::size_t aggregate_temp_bytes = static_cast<std::size_t>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count) * 2ULL * sizeof(unsigned long long);\n";
    out << "        const std::size_t aggregate_total_mem = static_cast<std::size_t>(q.get_device().get_info<sycl::info::device::global_mem_size>());\n";
    out << "        const std::size_t aggregate_reserve_fraction = static_cast<std::size_t>(static_cast<double>(aggregate_total_mem) * ctx->config_.memory_guard_reserve_fraction);\n";
    out << "        const std::size_t aggregate_reserve = aggregate_reserve_fraction > ctx->config_.memory_guard_reserve_bytes ? aggregate_reserve_fraction : ctx->config_.memory_guard_reserve_bytes;\n";
    out << "        const std::size_t aggregate_budget = aggregate_total_mem > aggregate_reserve ? aggregate_total_mem - aggregate_reserve : aggregate_total_mem / 2ULL;\n";
    out << "        const std::size_t aggregate_required = ctx->loaded_device_bytes_ + aggregate_temp_bytes + aggregate_result_bytes + aggregate_validity_bytes;\n";
    out << "        if (ctx->config_.memory_guard_enabled && aggregate_required > aggregate_budget) {\n";
    out << "            throw std::runtime_error(\"Insufficient GPU memory after aggregate count: exact dense aggregate result does not fit device memory\");\n";
    out << "        }\n";
    out << "        ctx->result_row_count_ = static_cast<std::size_t>(aggregate_row_count);\n";
    out << "        ctx->result_is_dense_ = true;\n";
    out << "        ctx->expected_result_size_ = aggregate_result_values == 0 ? aggregate_tuple_size : aggregate_result_values;\n";
    out << "        ctx->ensureColumnarResultCapacity(static_cast<std::size_t>(aggregate_tuple_size), static_cast<std::size_t>(aggregate_row_count));\n";
    for (int c = 0; c < visible_ts; ++c) {
        const LogicalType type = resultLogicalTypeForAggregateOutput(node, c);
        out << "        " << resultColumnPointerType(type) << " d_result_col_" << c << " = ctx->" << resultColumnGetter(type) << "(" << c << ");\n";
        out << "        uint64_t* d_result_validity_col_" << c << " = ctx->getResultColumnValidityPointer(" << c << ");\n";
    }
    out << "        q.submit([&](sycl::handler& h) {\n";
    out << "            h.parallel_for<class " << write_kernel << ">(sycl::range<1>(aggregate_group_count == 0ULL ? 1ULL : aggregate_group_count), [=](sycl::id<1> gid) {\n";
    out << "                unsigned long long g = static_cast<unsigned long long>(gid[0]);\n";
    out << "                if (g >= aggregate_group_count || d_aggregate_counts[g] == 0ULL) return;\n";
    out << "                unsigned long long out_row = d_aggregate_offsets[g];\n";
    out << "                unsigned long long src = g * aggregate_tuple_size;\n";
    for (int c = 0; c < visible_ts; ++c) {
        const LogicalType type = resultLogicalTypeForAggregateOutput(node, c);
        std::string value_expr = "d_result[src + " + std::to_string(c) + "]";
        if (type == LogicalType::Float64) {
            value_expr = "db::bit_cast_ull_to_double(" + value_expr + ")";
        } else {
            value_expr = castToResultColumnType(value_expr, type);
        }
        out << "                d_result_col_" << c << "[out_row] = " << value_expr << ";\n";
        if (resultColumnNullableForAggregateOutput(node, c, catalog)) {
            out << "                if (db::bitmap_valid_at(d_result_validity, src + " << c << ")) db::atomic_set_valid_bit(d_result_validity_col_" << c << ", out_row);\n";
        }
    }
    out << "            });\n";
    out << "        });\n";
    out << "        q.wait();\n";
    out << "        sycl::free(d_aggregate_counts, q);\n";
    out << "        sycl::free(d_aggregate_offsets, q);\n";
    out << "    }\n\n";
}

static void emitAtomicAggregateUpdate(std::stringstream& code,
                                      const std::string& ref_expr,
                                      const AggregateDef& agg,
                                      const std::string& value_expr) {
    if (agg.isMin()) {
        code << "                    db::atomic_min_ull(" << ref_expr << ", " << value_expr << ");\n";
    } else if (agg.isMax()) {
        code << "                    db::atomic_max_ull(" << ref_expr << ", " << value_expr << ");\n";
    } else {
        code << "                    db::atomic_add_ull(" << ref_expr << ", " << value_expr << ");\n";
    }
}

// ============================================================================
// consumeAggregateVector — opens scalar aggregation loop.
// ============================================================================
void JITOperatorVisitor::consumeAggregateVector(const AggregateNode* node, JITContext& ctx,
                                                 const OperatorNode* /*sender*/,
                                                 const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    const bool has_group_by = !node->group_by_exprs.empty();

    std::vector<std::string> agg_cols;
    collectAggregateColumns(node, agg_cols);

    int reg_idx = 0;
    for (const auto& col : agg_cols) {
        if (ctx.col_to_reg.count(col) && ctx.col_to_reg[col] == col) continue;
        std::string reg_name = (reg_idx == 0) ? "items" : ("items" + std::to_string(reg_idx + 1));
        loadIntoReg(col, reg_name, ctx);
        ++reg_idx;
    }

    for (const auto& g : node->group_by_exprs) {
        if (g->getType() == ExprType::COLUMN_REF) {
            const auto* col = static_cast<const ColumnRefExpr*>(g.get());
            (void)resolveLoadedRegOrThrow(col->column_name, ctx);
        }
    }

    const int visible_ts = (int)node->visibleTupleSize();
    const bool needs_hidden_count = node->needsHiddenCountSlot();
    const int storage_ts = (int)node->storageTupleSize();
    const int hidden_count_slot = visible_ts;
    const bool fast_sparse_output = aggregateFastSparseOutputEligible(node, catalog_);

    if (!has_group_by) {
        ctx.tuple_size = visible_ts;
        ctx.result_size_expr = std::to_string(storage_ts);
        ctx.visible_result_size_expr = std::to_string(visible_ts);
        emitAggregateInitializationIfNeeded(node, ctx, 1, visible_ts, storage_ts);

        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            const auto& agg = node->aggregates[a];
            if (agg.isMin()) {
                code << "            unsigned long long local_" << a << " = ~0ULL;\n";
            } else {
                code << "            unsigned long long local_" << a << " = 0ULL;\n";
            }
        }
        if (needs_hidden_count) {
            for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
                if (node->aggregates[a].needsNonNullCount()) {
                    code << "            unsigned long long local_count_hidden_" << a << " = 0ULL;\n";
                }
            }
        }
        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            if (!node->aggregates[a].isCount() && !node->aggregates[a].needsNonNullCount()) {
                code << "            unsigned long long local_seen_" << a << " = 0ULL;\n";
            }
        }

        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            const auto& agg = node->aggregates[a];
            std::string val = aggregateInputExpr(agg, ctx, true);
            std::string valid = agg.hasStarArgument() ? "1" : expressionValidExpr(agg.agg_expr.get(), ctx);
            if (agg.isCount()) {
                code << "                    if (" << valid << ") local_" << a << " += 1ULL;\n";
            } else if (agg.isMin()) {
                code << "                    if ((" << valid << ") && " << val << " < local_" << a << ") local_" << a << " = " << val << ";\n";
                if (agg.needsNonNullCount()) {
                    code << "                    if (" << valid << ") local_count_hidden_" << a << " += 1ULL;\n";
                } else {
                    code << "                    if (" << valid << ") local_seen_" << a << " += 1ULL;\n";
                }
            } else if (agg.isMax()) {
                code << "                    if ((" << valid << ") && " << val << " > local_" << a << ") local_" << a << " = " << val << ";\n";
                if (agg.needsNonNullCount()) {
                    code << "                    if (" << valid << ") local_count_hidden_" << a << " += 1ULL;\n";
                } else {
                    code << "                    if (" << valid << ") local_seen_" << a << " += 1ULL;\n";
                }
            } else {
                code << "                    if (" << valid << ") local_" << a << " += " << val << ";\n";
                if (agg.needsNonNullCount()) {
                    code << "                    if (" << valid << ") local_count_hidden_" << a << " += 1ULL;\n";
                } else if (!agg.isCount()) {
                    code << "                    if (" << valid << ") local_seen_" << a << " += 1ULL;\n";
                }
            }
        }
        code << "                }\n";
        code << "            }\n";

        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            const auto& agg = node->aggregates[a];
            const char* op = agg.isMin() ? "sycl::minimum<unsigned long long>{}" :
                             agg.isMax() ? "sycl::maximum<unsigned long long>{}" :
                                           "sycl::plus<unsigned long long>{}";
            code << "            unsigned long long agg_" << a
                 << " = sycl::reduce_over_group(it.get_group(), local_" << a << ", " << op << ");\n";
            code << "            if (tid == 0) {\n";
            if (agg.isMin()) {
                code << "                db::atomic_min_ull(d_result[" << a << "], agg_" << a << ");\n";
            } else if (agg.isMax()) {
                code << "                db::atomic_max_ull(d_result[" << a << "], agg_" << a << ");\n";
            } else {
                code << "                db::atomic_add_ull(d_result[" << a << "], agg_" << a << ");\n";
            }
            if (agg.isCount()) {
                code << "                db::atomic_set_valid_bit(d_result_validity, " << a << ");\n";
            }
            code << "            }\n";
            if (!agg.isCount() && !agg.needsNonNullCount()) {
                code << "            unsigned long long agg_seen_" << a << " = sycl::reduce_over_group(it.get_group(), local_seen_" << a << ", sycl::plus<unsigned long long>{});\n";
                code << "            if (tid == 0 && agg_seen_" << a << " != 0ULL) {\n";
                code << "                db::atomic_set_valid_bit(d_result_validity, " << a << ");\n";
                code << "            }\n";
            }
            if (agg.needsNonNullCount()) {
                const int cnt_slot = hiddenCountSlotForAggregate(node, a);
                code << "            unsigned long long agg_hidden_count_" << a << " = sycl::reduce_over_group(it.get_group(), local_count_hidden_" << a << ", sycl::plus<unsigned long long>{});\n";
                code << "            if (tid == 0) {\n";
                code << "                db::atomic_add_ull(d_result[" << cnt_slot << "], agg_hidden_count_" << a << ");\n";
                code << "            }\n";
            }
        }
        emitAggregateFinalizationIfNeeded(node, ctx, 1, visible_ts, storage_ts, hidden_count_slot);
        if (!fast_sparse_output) {
            emitAggregateDenseColumnarMaterialization(node, ctx, catalog_, 1, visible_ts);
        } else {
            ctx.post_execution_code << "    ctx->result_is_dense_ = false;\n";
            ctx.post_execution_code << "    ctx->result_row_count_ = 0;\n";
            ctx.post_execution_code << "    ctx->expected_result_validity_words_ = 0;\n";
        }
    } else {
        auto [hash_expr, total_size] = generatePerfectHash(node->group_by_exprs, catalog_, ctx);
        ctx.tuple_size = visible_ts;
        ctx.result_size_expr = std::to_string((uint64_t)total_size * (uint64_t)storage_ts);
        ctx.visible_result_size_expr = std::to_string((uint64_t)total_size * (uint64_t)visible_ts);
        emitAggregateInitializationIfNeeded(node, ctx, total_size, visible_ts, storage_ts);

        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        code << "                    int hash = " << hash_expr << ";\n";
        code << "                    int out = hash * " << storage_ts << ";\n";

        int slot = 0;
        for (const auto& g : node->group_by_exprs) {
            if (g->getType() == ExprType::COLUMN_REF) {
                const auto* col = static_cast<const ColumnRefExpr*>(g.get());
                const std::string reg_name = resolveLoadedRegOrThrow(col->column_name, ctx);
                std::string valid = expressionValidExpr(g.get(), ctx);
                code << "                    d_result[out + " << slot << "] = " << reg_name << "[i];\n";
                if (!fast_sparse_output) {
                    if (isTriviallyTrueExpr(valid)) {
                        code << "                    db::atomic_set_valid_bit(d_result_validity, out + " << slot << ");\n";
                    } else {
                        code << "                    if (" << valid << ") db::atomic_set_valid_bit(d_result_validity, out + " << slot << ");\n";
                    }
                }
            }
            ++slot;
        }
        for (const auto& agg : node->aggregates) {
            std::string val = aggregateInputExpr(agg, ctx, true);
            std::string valid = agg.hasStarArgument() ? "1" : expressionValidExpr(agg.agg_expr.get(), ctx);
            std::string ref = "d_result[out + " + std::to_string(slot) + "]";
            const bool valid_is_true = isTriviallyTrueExpr(valid);
            if (!valid_is_true) {
                code << "                    if (" << valid << ") {\n";
            }
            emitAtomicAggregateUpdate(code, ref, agg, val);
            if (agg.needsNonNullCount()) {
                const int cnt_slot = hiddenCountSlotForAggregate(node, slot - (int)node->group_by_exprs.size());
                code << "                    db::atomic_add_ull(d_result[out + " << cnt_slot << "], 1ULL);\n";
            } else if (!agg.isCount() && !fast_sparse_output) {
                code << "                    db::atomic_set_valid_bit(d_result_validity, out + " << slot << ");\n";
            }
            if (!valid_is_true) {
                code << "                    }\n";
            }
            if (agg.isCount() && !fast_sparse_output) {
                code << "                    db::atomic_set_valid_bit(d_result_validity, out + " << slot << ");\n";
            }
            ++slot;
        }
        code << "                }\n";
        code << "            }\n";
        emitAggregateFinalizationIfNeeded(node, ctx, total_size, visible_ts, storage_ts, hidden_count_slot);
        if (!fast_sparse_output) {
            emitAggregateDenseColumnarMaterialization(node, ctx, catalog_, total_size, visible_ts);
        } else {
            ctx.post_execution_code << "    ctx->result_is_dense_ = false;\n";
            ctx.post_execution_code << "    ctx->result_row_count_ = 0;\n";
            ctx.post_execution_code << "    ctx->expected_result_validity_words_ = 0;\n";
        }
    }
}

// ============================================================================
// consumeAggregateItem — direct atomic inside MHT expansion loop
// ============================================================================
void JITOperatorVisitor::consumeAggregateItem(const AggregateNode* node, JITContext& ctx,
                                               const OperatorNode* /*sender*/,
                                               const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    const bool has_group_by = !node->group_by_exprs.empty();
    const int visible_ts = (int)node->visibleTupleSize();
    const bool needs_hidden_count = node->needsHiddenCountSlot();
    const int storage_ts = (int)node->storageTupleSize();
    const int hidden_count_slot = visible_ts;

    if (!has_group_by) {
        ctx.tuple_size = visible_ts;
        ctx.result_size_expr = std::to_string(storage_ts);
        ctx.visible_result_size_expr = std::to_string(visible_ts);
        emitAggregateInitializationIfNeeded(node, ctx, 1, visible_ts, storage_ts);
        for (std::size_t a = 0; a < node->aggregates.size(); ++a) {
            const auto& agg = node->aggregates[a];
            std::string val = agg.hasStarArgument() ? "1" : itemExprValue(agg.agg_expr.get(), ctx);
            std::string valid = agg.hasStarArgument() ? "1" : itemExprValid(agg.agg_expr.get(), ctx);
            std::string ref = "d_result[" + std::to_string(a) + "]";
            code << "                        if (" << valid << ") {\n";
            if (agg.isMin()) {
                code << "                            db::atomic_min_ull(" << ref << ", " << val << ");\n";
            } else if (agg.isMax()) {
                code << "                            db::atomic_max_ull(" << ref << ", " << val << ");\n";
            } else {
                code << "                            db::atomic_add_ull(" << ref << ", " << val << ");\n";
            }
            if (agg.needsNonNullCount()) {
                const int cnt_slot = hiddenCountSlotForAggregate(node, a);
                code << "                            db::atomic_add_ull(d_result[" << cnt_slot << "], 1ULL);\n";
            } else {
                code << "                            db::atomic_set_valid_bit(d_result_validity, " << a << ");\n";
            }
            code << "                        }\n";
        }
        emitAggregateFinalizationIfNeeded(node, ctx, 1, visible_ts, storage_ts, hidden_count_slot);
        emitAggregateDenseColumnarMaterialization(node, ctx, catalog_, 1, visible_ts);
    } else {
        auto [hash_expr, total_size] = generatePerfectHashItem(node->group_by_exprs, catalog_, ctx);
        ctx.tuple_size = visible_ts;
        ctx.result_size_expr = std::to_string((uint64_t)total_size * (uint64_t)storage_ts);
        ctx.visible_result_size_expr = std::to_string((uint64_t)total_size * (uint64_t)visible_ts);
        emitAggregateInitializationIfNeeded(node, ctx, total_size, visible_ts, storage_ts);
        code << "                        int hash = " << hash_expr << ";\n";
        int slot = 0;
        for (const auto& g : node->group_by_exprs) {
            if (g->getType() == ExprType::COLUMN_REF) {
                const std::string value = itemExprValue(g.get(), ctx);
                std::string valid = itemExprValid(g.get(), ctx);
                code << "                        d_result[(unsigned long long)hash*" << storage_ts
                     << "+" << slot << "] = (unsigned long long)(" << value << ");\n";
                code << "                        if (" << valid << ") db::atomic_set_valid_bit(d_result_validity, (unsigned long long)hash*"
                     << storage_ts << "+" << slot << ");\n";
            }
            ++slot;
        }
        for (const auto& agg : node->aggregates) {
            std::string val = agg.hasStarArgument() ? "1" : itemExprValue(agg.agg_expr.get(), ctx);
            std::string valid = agg.hasStarArgument() ? "1" : itemExprValid(agg.agg_expr.get(), ctx);
            std::string ref = "d_result[out + " + std::to_string(slot) + "]";
            code << "                        if (" << valid << ") {\n";
            if (agg.isMin()) {
                code << "                            db::atomic_min_ull(" << ref << ", " << val << ");\n";
            } else if (agg.isMax()) {
                code << "                            db::atomic_max_ull(" << ref << ", " << val << ");\n";
            } else {
                code << "                            db::atomic_add_ull(" << ref << ", " << val << ");\n";
            }
            if (agg.needsNonNullCount()) {
                const int cnt_slot = hiddenCountSlotForAggregate(node, slot - (int)node->group_by_exprs.size());
                code << "                            db::atomic_add_ull(d_result[(unsigned long long)hash*" << storage_ts << "+" << cnt_slot << "], 1ULL);\n";
            } else if (!agg.isCount()) {
                code << "                            db::atomic_set_valid_bit(d_result_validity, (unsigned long long)hash*" << storage_ts << "+" << slot << ");\n";
            }
            code << "                        }\n";
            if (agg.isCount()) {
                code << "                        db::atomic_set_valid_bit(d_result_validity, (unsigned long long)hash*" << storage_ts << "+" << slot << ");\n";
            }
            ++slot;
        }
        emitAggregateFinalizationIfNeeded(node, ctx, total_size, visible_ts, storage_ts, hidden_count_slot);
        emitAggregateDenseColumnarMaterialization(node, ctx, catalog_, total_size, visible_ts);
    }
}


static bool projectionRequiresRowIdPayload(const HashJoinNode* join_node,
                                           const std::string& dim_table) {
    const OperatorNode* curr = join_node ? join_node->parent_ : nullptr;
    while (curr) {
        if (curr->getType() == OperatorType::PROJECTION) {
            const auto* proj = static_cast<const ProjectionNode*>(curr);
            for (const auto& expr : proj->select_exprs) {
                if (!expr) continue;
                if (expr->getType() == ExprType::STAR) return true;
                std::vector<std::string> cols;
                extractColumnsForTable(expr.get(), dim_table, cols);
                if (!cols.empty()) return true;
            }
        }
        curr = curr->parent_;
    }
    return false;
}

static void ensureProjectionExactBuffers(JITContext& ctx,
                                         const std::string& row_count_expr,
                                         int tuple_size,
                                         const std::vector<LogicalType>& result_types) {
    const std::string num_tiles_expr = "((" + row_count_expr + " + TILE_SIZE - 1) / TILE_SIZE)";
    auto has_buffer = [&](const std::string& name) {
        for (const auto& ht : ctx.hash_tables) {
            if (ht.name == name) return true;
        }
        return false;
    };
    if (!has_buffer("d_projection_counts")) {
        ctx.hash_tables.push_back({"d_projection_counts", "unsigned long long", num_tiles_expr});
    }
    if (!has_buffer("d_projection_offsets")) {
        ctx.hash_tables.push_back({"d_projection_offsets", "unsigned long long", num_tiles_expr});
    }
    if (!has_buffer("d_projection_write_counts")) {
        ctx.hash_tables.push_back({"d_projection_write_counts", "unsigned long long", num_tiles_expr});
    }
    const std::string scan_blocks_expr = "(((" + row_count_expr + " + TILE_SIZE - 1) / TILE_SIZE + 255) / 256)";
    if (!has_buffer("d_projection_block_sums")) {
        ctx.hash_tables.push_back({"d_projection_block_sums", "unsigned long long", scan_blocks_expr});
    }

    ctx.projection_exact_materialization = true;
    ctx.projection_row_count_expr = row_count_expr;
    ctx.projection_tuple_size = tuple_size;
    ctx.result_size_expr = "1";
    ctx.visible_result_size_expr = "1";
}

static void emitProjectionPrefixScanBeforeWrite(JITContext& ctx,
                                                const std::string& row_count_expr,
                                                int tuple_size,
                                                const std::vector<LogicalType>& result_types) {
    const std::string marker = "projection_exact_prefix_scan";
    if (!ctx.emitted_auxiliary_kernels.insert(marker).second) return;

    auto& out = ctx.current_pipeline->includes_and_globals;
    for (int col = 0; col < tuple_size; ++col) {
        const LogicalType type = (col < static_cast<int>(result_types.size())) ? result_types[static_cast<std::size_t>(col)] : LogicalType::UInt64;
        out << "    " << resultColumnPointerType(type) << " d_result_col_" << col << " = nullptr;\n";
        out << "    uint64_t* d_result_validity_col_" << col << " = nullptr;\n";
    }
    out << "    {\n";
    out << "        const int projection_num_tiles = (" << row_count_expr << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    out << "        const int projection_scan_blocks = (projection_num_tiles + 255) / 256;\n";
    out << "        q.submit([&](sycl::handler& h) {\n";
    out << "            h.parallel_for<class ProjectionTileBlockScan>(sycl::nd_range<1>(projection_scan_blocks * 256, 256), [=](sycl::nd_item<1> it) {\n";
    out << "                const int gid = static_cast<int>(it.get_global_linear_id());\n";
    out << "                const int lid = static_cast<int>(it.get_local_linear_id());\n";
    out << "                unsigned long long val = (gid < projection_num_tiles) ? d_projection_counts[gid] : 0ULL;\n";
    out << "                unsigned long long scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<unsigned long long>{});\n";
    out << "                if (gid < projection_num_tiles) d_projection_offsets[gid] = scanned;\n";
    out << "                if (lid == 255) d_projection_block_sums[it.get_group_linear_id()] = scanned + val;\n";
    out << "            });\n";
    out << "        });\n";
    out << "        q.submit([&](sycl::handler& h) {\n";
    out << "            h.parallel_for<class ProjectionBlockSumsScan>(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {\n";
    out << "                const int gid = static_cast<int>(it.get_global_linear_id());\n";
    out << "                unsigned long long val = (gid < projection_scan_blocks) ? d_projection_block_sums[gid] : 0ULL;\n";
    out << "                unsigned long long scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<unsigned long long>{});\n";
    out << "                if (gid < projection_scan_blocks) d_projection_block_sums[gid] = scanned;\n";
    out << "            });\n";
    out << "        });\n";
    out << "        q.submit([&](sycl::handler& h) {\n";
    out << "            h.parallel_for<class ProjectionAddBlockSums>(sycl::nd_range<1>(projection_scan_blocks * 256, 256), [=](sycl::nd_item<1> it) {\n";
    out << "                const int gid = static_cast<int>(it.get_global_linear_id());\n";
    out << "                if (gid < projection_num_tiles) {\n";
    out << "                    d_projection_offsets[gid] += d_projection_block_sums[it.get_group_linear_id()];\n";
    out << "                }\n";
    out << "            });\n";
    out << "        });\n";
    out << "        unsigned long long projection_last_offset = 0ULL;\n";
    out << "        unsigned long long projection_last_count = 0ULL;\n";
    out << "        if (projection_num_tiles > 0) {\n";
    out << "            q.memcpy(&projection_last_offset, d_projection_offsets + projection_num_tiles - 1, sizeof(unsigned long long)).wait();\n";
    out << "            q.memcpy(&projection_last_count, d_projection_counts + projection_num_tiles - 1, sizeof(unsigned long long)).wait();\n";
    out << "        }\n";
    out << "        unsigned long long projection_running = projection_last_offset + projection_last_count;\n";
    out << "        if (projection_running != 0ULL && projection_running > static_cast<unsigned long long>(std::numeric_limits<size_t>::max() / " << tuple_size << ")) {\n";
    out << "            throw std::overflow_error(\"Projection result size overflow\");\n";
    out << "        }\n";
    out << "        ctx->result_row_count_ = static_cast<size_t>(projection_running);\n";
    out << "        ctx->result_is_dense_ = true;\n";
    out << "        ctx->expected_result_size_ = static_cast<size_t>(projection_running) * " << tuple_size << ";\n";
    out << "        const size_t projection_result_bytes = ctx->expected_result_size_ * sizeof(unsigned long long);\n";
    out << "        const size_t projection_temp_bytes = static_cast<size_t>(projection_num_tiles) * 3ULL * sizeof(unsigned long long) + static_cast<size_t>(projection_scan_blocks) * sizeof(unsigned long long);\n";
    out << "        const size_t projection_total_mem = static_cast<size_t>(q.get_device().get_info<sycl::info::device::global_mem_size>());\n";
    out << "        const size_t projection_reserve_a = projection_total_mem / 10ULL;\n";
    out << "        const size_t projection_reserve_b = static_cast<size_t>(512ULL * 1024ULL * 1024ULL);\n";
    out << "        const size_t projection_reserve = projection_reserve_a > projection_reserve_b ? projection_reserve_a : projection_reserve_b;\n";
    out << "        const size_t projection_budget = projection_total_mem > projection_reserve ? projection_total_mem - projection_reserve : projection_total_mem / 2ULL;\n";
    out << "        const size_t projection_required = ctx->loaded_device_bytes_ + projection_temp_bytes + projection_result_bytes;\n";
    out << "        if (projection_required > projection_budget) {\n";
    out << "            throw std::runtime_error(\"Insufficient GPU memory after projection count: exact materialized result does not fit device memory\");\n";
    out << "        }\n";
    out << "        ctx->ensureColumnarResultCapacity(" << tuple_size << ", static_cast<size_t>(projection_running));\n";
    out << "        q.memset(d_projection_write_counts, 0, projection_num_tiles * sizeof(unsigned long long));\n";
    for (int col = 0; col < tuple_size; ++col) {
        const LogicalType type = (col < static_cast<int>(result_types.size())) ? result_types[static_cast<std::size_t>(col)] : LogicalType::UInt64;
        out << "        d_result_col_" << col << " = ctx->" << resultColumnGetter(type) << "(" << col << ");\n";
        out << "        d_result_validity_col_" << col << " = ctx->getResultColumnValidityPointer(" << col << ");\n";
    }
    out << "    }\n\n";
}

static void collectTableScans(const OperatorNode* node, std::vector<const TableScanNode*>& out) {
    if (!node) return;
    if (node->getType() == OperatorType::TABLE_SCAN) {
        out.push_back(static_cast<const TableScanNode*>(node));
    }
    for (const auto& child : node->getChildren()) collectTableScans(child.get(), out);
}

static std::vector<std::string> expandProjectionExpressions(
        const ProjectionNode* node,
        const Catalog& catalog,
        bool allow_join_star) {
    std::vector<std::string> cols;
    std::vector<const TableScanNode*> scans;
    if (!node->getChildren().empty()) collectTableScans(node->getChildren()[0].get(), scans);

    for (const auto& expr : node->select_exprs) {
        if (!expr) continue;
        if (expr->getType() == ExprType::STAR) {
            if (scans.size() > 1 && !allow_join_star) {
                throw std::runtime_error(
                    "SELECT * over joins is not supported by the current single-payload PHT projection path. "
                    "List the required columns explicitly.");
            }
            for (const auto* scan : scans) {
                const auto& meta = catalog.getTableMetadata(scan->table_name);
                for (const auto& col : meta.getColumnNames()) cols.push_back(col);
            }
        } else if (expr->getType() == ExprType::COLUMN_REF) {
            cols.push_back(static_cast<const ColumnRefExpr*>(expr.get())->column_name);
        } else {
            // Complex expression: emitted as an expression slot, not a named column.
            cols.push_back("");
        }
    }
    return cols;
}

void JITOperatorVisitor::consumeProjectionVector(const ProjectionNode* node, JITContext& ctx,
                                                  const OperatorNode* /*sender*/,
                                                  const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    std::vector<const TableScanNode*> scans;
    if (!node->getChildren().empty()) collectTableScans(node->getChildren()[0].get(), scans);

    // SELECT * over joins is implemented through row-id payloads in PHTUnique:
    // dimension columns are gathered lazily at projection time.
    (void)expandProjectionExpressions(node, catalog_, true);

    std::vector<const ExprNode*> expanded;
    for (const auto& expr : node->select_exprs) {
        if (!expr) continue;
        if (expr->getType() == ExprType::STAR) {
            for (const auto* scan : scans) {
                const auto& meta = catalog_.getTableMetadata(scan->table_name);
                for (const auto& col : meta.getColumnNames()) {
                    expanded_projection_exprs_.push_back(std::make_unique<ColumnRefExpr>(col));
                    expanded.push_back(expanded_projection_exprs_.back().get());
                }
            }
        } else {
            expanded.push_back(expr.get());
        }
    }

    const int tuple_size = (int)expanded.size();
    ctx.tuple_size = tuple_size > 0 ? tuple_size : 1;
    std::vector<LogicalType> projection_types;
    projection_types.reserve(expanded.size());
    for (const auto* expr : expanded) projection_types.push_back(resultLogicalTypeForExpr(expr));
    std::string row_count = "LO_LEN";
    if (scans.size() == 1) row_count = sizeMacroFor(scans[0]->table_name);
    ensureProjectionExactBuffers(ctx, row_count, ctx.tuple_size, projection_types);

    if (projection_pass_ == ProjectionPass::Count) {
        code << "            unsigned long long projection_local_count = 0ULL;\n";
        code << "            #pragma unroll\n";
        code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
        code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
        code << "                    ++projection_local_count;\n";
        code << "                }\n";
        code << "            }\n";
        code << "            if (projection_local_count != 0ULL) {\n";
        code << "                db::atomic_add_ull(d_projection_counts[it.get_group_linear_id()], projection_local_count);\n";
        code << "            }\n";
        return;
    }

    if (projection_pass_ == ProjectionPass::Write) {
        emitProjectionPrefixScanBeforeWrite(ctx, row_count, ctx.tuple_size, projection_types);
    }

    std::vector<std::string> cols;
    for (const auto* expr : expanded) extractAllColumns(expr, cols);
    int reg_idx = 0;
    for (const auto& col : cols) {
        if (ctx.col_to_reg.count(col) && ctx.col_to_reg[col] == col) continue;
        std::string reg_name = (reg_idx == 0) ? "items" : ("items" + std::to_string(reg_idx + 1));
        loadIntoReg(col, reg_name, ctx);
        ++reg_idx;
    }

    JITExprVisitor expr_vis(ctx, code, "flags", false, nullptr);
    code << "            #pragma unroll\n";
    code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
    code << "                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {\n";
    code << "                    unsigned long long projection_local_row = db::atomic_fetch_add_ull(d_projection_write_counts[it.get_group_linear_id()], 1ULL);\n";
    code << "                    unsigned long long out_row = d_projection_offsets[it.get_group_linear_id()] + projection_local_row;\n";
    for (int slot = 0; slot < (int)expanded.size(); ++slot) {
        std::string val = expr_vis.translateInlineExpr(expanded[slot], true);
        std::string valid = expressionValidExpr(expanded[slot], ctx);
        const LogicalType out_type = (slot < static_cast<int>(projection_types.size())) ? projection_types[static_cast<std::size_t>(slot)] : LogicalType::UInt64;
        code << "                    d_result_col_" << slot << "[out_row] = " << castToResultColumnType(val, out_type) << ";\n";
        if (valid == "1" || valid == "true" || valid == "(1)" || valid == "(true)") {
            code << "                    db::atomic_set_valid_bit(d_result_validity_col_" << slot << ", out_row);\n";
        } else if (!(valid == "0" || valid == "false" || valid == "(0)" || valid == "(false)")) {
            code << "                    if (" << valid << ") db::atomic_set_valid_bit(d_result_validity_col_" << slot << ", out_row);\n";
        }
    }
    code << "                }\n";
    code << "            }\n";
}

static std::string itemColumnValueExpr(const std::string& col_name, JITContext& ctx) {
    auto mapped = ctx.col_to_reg.find(col_name);
    if (mapped != ctx.col_to_reg.end() && !mapped->second.empty()) {
        return mapped->second + "[i]";
    }
    const std::string table = getTableName(col_name);
    auto rid = ctx.table_rowid_regs.find(table);
    ctx.external_columns.insert("d_" + col_name);
    if (ctx.nullable_columns.count(col_name)) ctx.external_null_columns.insert(nullBitmapSymbolFor(col_name));
    if (rid != ctx.table_rowid_regs.end() && !rid->second.empty()) {
        return "d_" + col_name + "[" + rid->second + "[i]]";
    }
    // Probe-side columns should have been preloaded before entering the MHT
    // scalar expansion loop.  This fallback remains safe for table scans that
    // directly drive item-mode in tests.
    return "d_" + col_name + "[tile_offset + tid + BLOCK_THREADS * i]";
}

static std::string itemColumnValidExpr(const std::string& col_name, JITContext& ctx) {
    auto mapped = ctx.col_to_valid_reg.find(col_name);
    if (mapped != ctx.col_to_valid_reg.end() && !mapped->second.empty()) {
        return mapped->second + "[i]";
    }
    if (!ctx.nullable_columns.count(col_name)) return "1";
    const std::string table = getTableName(col_name);
    auto rid = ctx.table_rowid_regs.find(table);
    ctx.external_null_columns.insert(nullBitmapSymbolFor(col_name));
    if (rid != ctx.table_rowid_regs.end() && !rid->second.empty()) {
        return "db::bitmap_valid_at(" + nullBitmapSymbolFor(col_name) + ", " + rid->second + "[i])";
    }
    return "db::bitmap_valid_at(" + nullBitmapSymbolFor(col_name) + ", tile_offset + tid + BLOCK_THREADS * i)";
}

static std::string itemExprValue(const ExprNode* e, JITContext& ctx);
static std::string itemExprValid(const ExprNode* e, JITContext& ctx);

static std::string itemExprValue(const ExprNode* e, JITContext& ctx) {
    if (!e) return "0";
    switch (e->getType()) {
        case ExprType::COLUMN_REF:
            return itemColumnValueExpr(static_cast<const ColumnRefExpr*>(e)->column_name, ctx);
        case ExprType::LITERAL_INT:
            return std::to_string(static_cast<const LiteralIntExpr*>(e)->value);
        case ExprType::LITERAL_FLOAT:
            return std::to_string(static_cast<const LiteralFloatExpr*>(e)->value);
        case ExprType::STAR:
            return "1";
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string l = itemExprValue(b->left.get(), ctx);
            const std::string r = itemExprValue(b->right.get(), ctx);
            switch (e->getType()) {
                case ExprType::OP_ADD: return "db::safe_add(" + l + ", " + r + ")";
                case ExprType::OP_SUB: return "db::safe_sub(" + l + ", " + r + ")";
                case ExprType::OP_MUL: return "db::safe_mul(" + l + ", " + r + ")";
                case ExprType::OP_DIV: return "db::safe_div(" + l + ", " + r + ")";
                default: break;
            }
            return "0";
        }
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string l = itemExprValue(b->left.get(), ctx);
            const std::string r = itemExprValue(b->right.get(), ctx);
            switch (e->getType()) {
                case ExprType::OP_EQ:  return "db::safe_eq(" + l + ", " + r + ")";
                case ExprType::OP_NEQ: return "db::safe_neq(" + l + ", " + r + ")";
                case ExprType::OP_LT:  return "db::safe_lt(" + l + ", " + r + ")";
                case ExprType::OP_LTE: return "db::safe_lte(" + l + ", " + r + ")";
                case ExprType::OP_GT:  return "db::safe_gt(" + l + ", " + r + ")";
                case ExprType::OP_GTE: return "db::safe_gte(" + l + ", " + r + ")";
                default: break;
            }
            return "0";
        }
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string v = itemExprValid(b->left.get(), ctx);
            if (isLiteralTrue(v)) return e->getType() == ExprType::OP_IS_NULL ? "0" : "1";
            if (isLiteralFalse(v)) return e->getType() == ExprType::OP_IS_NULL ? "1" : "0";
            return e->getType() == ExprType::OP_IS_NULL ? "(!(" + v + "))" : "(" + v + ")";
        }
        case ExprType::OP_NOT: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanNotValue(itemExprValue(b->left.get(), ctx));
        }
        case ExprType::OP_AND: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanAndValue(itemExprValue(b->left.get(), ctx), itemExprValue(b->right.get(), ctx));
        }
        case ExprType::OP_OR: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return booleanOrValue(itemExprValue(b->left.get(), ctx), itemExprValue(b->right.get(), ctx));
        }
    }
    return "0";
}

static std::string itemExprValid(const ExprNode* e, JITContext& ctx) {
    if (!e) return "1";
    switch (e->getType()) {
        case ExprType::STAR:
        case ExprType::LITERAL_INT:
        case ExprType::LITERAL_FLOAT:
        case ExprType::OP_IS_NULL:
        case ExprType::OP_IS_NOT_NULL:
            return "1";
        case ExprType::COLUMN_REF:
            return itemColumnValidExpr(static_cast<const ColumnRefExpr*>(e)->column_name, ctx);
        case ExprType::OP_ADD:
        case ExprType::OP_SUB:
        case ExprType::OP_MUL:
        case ExprType::OP_DIV:
        case ExprType::OP_EQ:
        case ExprType::OP_NEQ:
        case ExprType::OP_LT:
        case ExprType::OP_LTE:
        case ExprType::OP_GT:
        case ExprType::OP_GTE: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return combineAndTerms({itemExprValid(b->left.get(), ctx), itemExprValid(b->right.get(), ctx)});
        }
        case ExprType::OP_NOT: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            return itemExprValid(b->left.get(), ctx);
        }
        case ExprType::OP_AND: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string lv = itemExprValid(b->left.get(), ctx);
            const std::string rv = itemExprValid(b->right.get(), ctx);
            const std::string lval = itemExprValue(b->left.get(), ctx);
            const std::string rval = itemExprValue(b->right.get(), ctx);
            return booleanOrValue(booleanOrValue(combineAndTerms({lv, booleanNotValue(lval)}),
                                                combineAndTerms({rv, booleanNotValue(rval)})),
                                  combineAndTerms({lv, rv}));
        }
        case ExprType::OP_OR: {
            const auto* b = static_cast<const BinaryExpr*>(e);
            const std::string lv = itemExprValid(b->left.get(), ctx);
            const std::string rv = itemExprValid(b->right.get(), ctx);
            const std::string lval = itemExprValue(b->left.get(), ctx);
            const std::string rval = itemExprValue(b->right.get(), ctx);
            return booleanOrValue(booleanOrValue(combineAndTerms({lv, lval}),
                                                combineAndTerms({rv, rval})),
                                  combineAndTerms({lv, rv}));
        }
    }
    return "1";
}


static std::pair<std::string, uint64_t> generatePerfectHashItem(
        const std::vector<std::unique_ptr<ExprNode>>& group_by,
        const Catalog& catalog,
        JITContext& ctx) {
    std::string hash_expr;
    uint64_t total_size = 1;
    for (const auto& g : group_by) {
        if (!g || g->getType() != ExprType::COLUMN_REF) continue;
        const auto* col = static_cast<const ColumnRefExpr*>(g.get());
        const std::string table_name = getTableName(col->column_name);
        uint64_t min_val = 0;
        uint64_t card = 1;
        bool nullable = false;
        try {
            const auto& meta = catalog.getTableMetadata(table_name);
            nullable = meta.isColumnNullable(col->column_name);
            if (meta.hasColumnStats(col->column_name)) {
                const auto& st = meta.getColumnStats(col->column_name);
                min_val = static_cast<uint64_t>(st.min_value_);
                card = st.cardinality_;
            }
        } catch (...) {}
        const std::string value = itemExprValue(g.get(), ctx);
        const std::string valid = itemExprValid(g.get(), ctx);
        std::string term;
        if (nullable) {
            term = "((" + valid + ") ? (" + value + " - " + std::to_string(min_val) + " + 1) : 0)";
            ++card;
        } else {
            term = "(" + value + " - " + std::to_string(min_val) + ")";
        }
        if (hash_expr.empty()) hash_expr = term;
        else hash_expr = "(" + hash_expr + " * " + std::to_string(card) + " + " + term + ")";
        total_size *= card;
    }
    if (hash_expr.empty()) {
        return {"0", 1};
    }
    return {"(" + hash_expr + ") % " + std::to_string(total_size), total_size};
}

void JITOperatorVisitor::consumeProjectionItem(const ProjectionNode* node, JITContext& ctx,
                                                const OperatorNode* /*sender*/,
                                                const std::vector<std::string>& /*active_vars*/) {
    auto& code = ctx.current_pipeline->kernel_body;
    std::vector<const TableScanNode*> scans;
    if (!node->getChildren().empty()) collectTableScans(node->getChildren()[0].get(), scans);

    std::vector<const ExprNode*> expanded;
    for (const auto& expr : node->select_exprs) {
        if (!expr) continue;
        if (expr->getType() == ExprType::STAR) {
            for (const auto* scan : scans) {
                const auto& meta = catalog_.getTableMetadata(scan->table_name);
                for (const auto& col : meta.getColumnNames()) {
                    expanded_projection_exprs_.push_back(std::make_unique<ColumnRefExpr>(col));
                    expanded.push_back(expanded_projection_exprs_.back().get());
                }
            }
        } else {
            expanded.push_back(expr.get());
        }
    }

    const int tuple_size = expanded.empty() ? 1 : static_cast<int>(expanded.size());
    ctx.tuple_size = tuple_size;
    std::vector<LogicalType> projection_types;
    for (const auto* expr : expanded) projection_types.push_back(resultLogicalTypeForExpr(expr));
    std::string row_count = "LO_LEN";
    for (const auto* scan : scans) {
        try {
            if (catalog_.getTableMetadata(scan->table_name).isFactTable()) {
                row_count = sizeMacroFor(scan->table_name);
                break;
            }
        } catch (...) {}
    }
    ensureProjectionExactBuffers(ctx, row_count, tuple_size, projection_types);

    if (projection_pass_ == ProjectionPass::Count) {
        code << "                        db::atomic_add_ull(d_projection_counts[it.get_group_linear_id()], 1ULL);\n";
        return;
    }

    if (projection_pass_ == ProjectionPass::Write) {
        emitProjectionPrefixScanBeforeWrite(ctx, row_count, tuple_size, projection_types);
        code << "                        unsigned long long projection_local_row = db::atomic_fetch_add_ull(d_projection_write_counts[it.get_group_linear_id()], 1ULL);\n";
        code << "                        unsigned long long out_row = d_projection_offsets[it.get_group_linear_id()] + projection_local_row;\n";
        for (int slot = 0; slot < static_cast<int>(expanded.size()); ++slot) {
            const LogicalType out_type = (slot < static_cast<int>(projection_types.size())) ? projection_types[static_cast<std::size_t>(slot)] : LogicalType::UInt64;
            const std::string val = itemExprValue(expanded[slot], ctx);
            const std::string valid = itemExprValid(expanded[slot], ctx);
            code << "                        d_result_col_" << slot << "[out_row] = " << castToResultColumnType(val, out_type) << ";\n";
            if (isLiteralTrue(valid)) {
                code << "                        db::atomic_set_valid_bit(d_result_validity_col_" << slot << ", out_row);\n";
            } else if (!isLiteralFalse(valid)) {
                code << "                        if (" << valid << ") db::atomic_set_valid_bit(d_result_validity_col_" << slot << ", out_row);\n";
            }
        }
    }
}



JITOperatorVisitor::BuildInfo JITOperatorVisitor::computeBuildInfo(
        const std::string& dim_table,
        const FilterNode* /*filter*/,
        const HashJoinNode* join_node) const {
    BuildInfo bi;
    bi.dim_table   = dim_table;
    bi.dim_prefix  = tablePrefix(dim_table);
    bi.size_macro  = sizeMacro(dim_table);
    bi.ht_name     = "d_" + bi.dim_prefix + "_hash_table";
    bi.variant     = 1;
    bi.use_mht     = false;
    bi.key_mins    = "0";
    bi.row_id_reg  = bi.dim_prefix + "_row_id";

    // SSB dimension keys are used by the hand-written reference kernels with
    // zero-based perfect-hash domains for PART/SUPPLIER/CUSTOMER.  Keeping the
    // same domains avoids subtle off-by-one differences for encoded key columns.
    if (dim_table == "PART") {
        bi.ht_size_expr = "P_LEN";
        bi.key_mins = "0";
    } else if (dim_table == "SUPPLIER") {
        bi.ht_size_expr = "S_LEN";
        bi.key_mins = "0";
    } else if (dim_table == "CUSTOMER") {
        bi.ht_size_expr = "C_LEN";
        bi.key_mins = "0";
    } else if (dim_table == "DDATE") {
        // Leave one spare slot beyond the historic 19981230 upper bound.  This
        // prevents modulo aliasing if the loaded SSB date dictionary contains
        // 19981231, while preserving the same hash values for all earlier dates.
        bi.ht_size_expr = "61131";
        bi.key_mins = "19920101";
    }

    if (bi.ht_size_expr.empty()) {
        try {
            const auto& meta = catalog_.getTableMetadata(dim_table);
            for (const auto& col_name : meta.getColumnNames()) {
                if (col_name.find("key") != std::string::npos ||
                    col_name.find("Key") != std::string::npos) {
                    if (meta.hasColumnStats(col_name)) {
                        const auto& stats = meta.getColumnStats(col_name);
                        int64_t range = stats.max_value_ - stats.min_value_ + 1;
                        if (range > 0) {
                            bi.key_mins    = std::to_string(stats.min_value_);
                            bi.ht_size_expr = std::to_string(range);
                        }
                        break;
                    }
                }
            }
        } catch (...) {}
    }

    if (bi.ht_size_expr.empty()) bi.ht_size_expr = bi.size_macro;

    // Extract pk_col / fk_col from join condition
    if (join_node && join_node->join_condition) {
        std::function<void(const ExprNode*)> extractKeys = [&](const ExprNode* e) {
            if (!e) return;
            if (e->getType() == ExprType::OP_EQ) {
                const auto* eq = static_cast<const BinaryExpr*>(e);
                if (eq->left && eq->right && 
                    eq->left->getType() == ExprType::COLUMN_REF && 
                    eq->right->getType() == ExprType::COLUMN_REF) {
                    const auto* lc = static_cast<const ColumnRefExpr*>(eq->left.get());
                    const auto* rc = static_cast<const ColumnRefExpr*>(eq->right.get());
                    if (getTableName(lc->column_name) == dim_table) {
                        bi.pk_col = lc->column_name;
                        bi.fk_cols.push_back(rc->column_name);
                    } else {
                        bi.pk_col = rc->column_name;
                        bi.fk_cols.push_back(lc->column_name);
                    }
                }
            } else if (e->getType() == ExprType::OP_AND || e->getType() == ExprType::OP_OR) {
                const auto* bin = static_cast<const BinaryExpr*>(e);
                extractKeys(bin->left.get());
                extractKeys(bin->right.get());
            }
        };
        extractKeys(join_node->join_condition.get());
    }

    // Choose MHT only when metadata proves that the build key is non-unique.
    // SSB PK/FK joins remain on the PHT fast path.  MHT still stores row_id or
    // a single scalar payload, never wide tuples.
    if (!bi.pk_col.empty() && !isCatalogUniqueBuildKey(catalog_, dim_table, bi.pk_col)) {
        try {
            const auto& meta = catalog_.getTableMetadata(dim_table);
            if (meta.hasColumnStats(bi.pk_col)) {
                const auto& st = meta.getColumnStats(bi.pk_col);
                if (st.cardinality_ > 0 && st.cardinality_ < meta.getSize()) {
                    bi.use_mht = true;
                }
            }
        } catch (...) {}
    }

    if (projectionRequiresRowIdPayload(join_node, dim_table) || bi.use_mht) {
        // MHT always carries row_id. This is the only payload that can support
        // projection, post-join filters, GROUP BY and aggregate expressions over
        // arbitrary build-side columns after row expansion.
        bi.payload_is_row_id = true;
        bi.variant = 2;
        bi.val_col.clear();
        return bi;
    }

    // 1. Собираем колонки, которые требуются узлам ВЫШЕ джойна
    // Разделяем на те, что нужны для агрегации (приоритет) и те, что для фильтрации.
    std::vector<std::string> agg_required;
    std::vector<std::string> filter_required;
    const OperatorNode* curr = join_node->parent_;
    while (curr) {
        if (curr->getType() == OperatorType::AGGREGATE) {
            const auto* agg = static_cast<const AggregateNode*>(curr);
            // Извлекаем колонки из всех агрегатных выражений
            for (const auto& a : agg->aggregates) {
                extractAllColumns(a.agg_expr.get(), agg_required);
            }
            // Извлекаем колонки из всех выражений группировки
            for (const auto& gb : agg->group_by_exprs) {
                extractAllColumns(gb.get(), agg_required);
            }
        } else if (curr->getType() == OperatorType::FILTER) {
            const auto* flt = static_cast<const FilterNode*>(curr);
            extractAllColumns(flt->predicate.get(), filter_required);
        }
        curr = curr->parent_;
    }

    // 2. Ищем val_col: сначала среди агрегатных, потом среди фильтровых
    bi.variant = 1;
    bi.val_col = "";

    auto findMatch = [&](const std::vector<std::string>& cols) -> bool {
        for (const auto& col : cols) {
            if (getTableName(col) == dim_table && col != bi.pk_col) {
                bi.val_col = col;
                bi.variant = 2;
                return true;
            }
        }
        return false;
    };

    if (!findMatch(agg_required)) {
        findMatch(filter_required);
    }

    return bi;
}

void JITOperatorVisitor::collectAllColumnsFromTree(const OperatorNode* node) {
    if (!node) return;
    if (node->getType() == OperatorType::AGGREGATE) {
        const auto* agg = static_cast<const AggregateNode*>(node);
        for (const auto& a : agg->aggregates) {
            extractAllColumns(a.agg_expr.get(), agg_cols_);
        }
        for (const auto& gb : agg->group_by_exprs) {
            extractAllColumns(gb.get(), agg_cols_);
        }
    } else if (node->getType() == OperatorType::FILTER) {
        const auto* flt = static_cast<const FilterNode*>(node);
        extractAllColumns(flt->predicate.get(), filter_cols_);
    } else if (node->getType() == OperatorType::PROJECTION) {
        const auto* proj = static_cast<const ProjectionNode*>(node);
        for (const auto& e : proj->select_exprs) {
            extractAllColumns(e.get(), agg_cols_);
        }
    }
    
    for (const auto& child : node->getChildren()) {
        collectAllColumnsFromTree(child.get());
    }
}

// Emits block-level filter predicates for a build-side dimension kernel.
// `flags` starts fully initialised (all 1).  For each comparison in the
// filter predicate we load the column (if not already loaded) and emit the
// appropriate BlockPred* call.  Returns the column currently in `items`.
std::string JITOperatorVisitor::emitBuildFilter(
        const FilterNode* filter,
        const std::string& /*dim_table*/,
        std::stringstream& code,
        JITContext& ctx) const {
    if (!filter || !filter->predicate) return "";

    int mask_counter = 0;
    std::string current_items_col;

    auto comparisonSuffix = [](ExprType op) -> const char* {
        switch (op) {
            case ExprType::OP_EQ:  return "Eq";
            case ExprType::OP_NEQ: return "NEq";
            case ExprType::OP_LT:  return "LT";
            case ExprType::OP_LTE: return "LTE";
            case ExprType::OP_GT:  return "GT";
            case ExprType::OP_GTE: return "GTE";
            default:               return "Eq";
        }
    };

    auto invertComparison = [](ExprType op) -> ExprType {
        switch (op) {
            case ExprType::OP_LT:  return ExprType::OP_GT;
            case ExprType::OP_LTE: return ExprType::OP_GTE;
            case ExprType::OP_GT:  return ExprType::OP_LT;
            case ExprType::OP_GTE: return ExprType::OP_LTE;
            default:               return op;
        }
    };

    auto loadColumnToItems = [&](const std::string& col_name) {
        if (current_items_col == col_name) return;
        ctx.external_columns.insert("d_" + col_name);
        code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
             << "d_" << col_name << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
        current_items_col = col_name;
    };

    std::function<bool(const ExprNode*, const std::string&, bool&, bool)> emitPredicate;

    emitPredicate = [&](const ExprNode* expr,
                        const std::string& target_mask,
                        bool& first_predicate,
                        bool or_context) -> bool {
        if (!expr) return true;

        if (expr->getType() == ExprType::OP_AND) {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            if (or_context) {
                const std::string and_mask = "build_and_mask_" + std::to_string(++mask_counter);
                code << "            int " << and_mask << "[ITEMS_PER_THREAD];\n";
                code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << and_mask << ");\n";
                bool and_first = true;
                if (!emitPredicate(bin->left.get(), and_mask, and_first, false)) return false;
                if (!emitPredicate(bin->right.get(), and_mask, and_first, false)) return false;

                // OR the whole conjunction into the parent OR accumulator.
                code << "            BlockApplyMaskOr<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, "
                     << target_mask << ", " << and_mask << ");\n";
                first_predicate = false;
                return true;
            }
            return emitPredicate(bin->left.get(), target_mask, first_predicate, false) &&
                   emitPredicate(bin->right.get(), target_mask, first_predicate, false);
        }

        if (expr->getType() == ExprType::OP_OR) {
            const auto* bin = static_cast<const BinaryExpr*>(expr);
            if (or_context) {
                return emitPredicate(bin->left.get(), target_mask, first_predicate, true) &&
                       emitPredicate(bin->right.get(), target_mask, first_predicate, true);
            }

            const std::string or_mask = "build_or_mask_" + std::to_string(++mask_counter);
            code << "            int " << or_mask << "[ITEMS_PER_THREAD];\n";
            code << "            InitFlagsZero<BLOCK_THREADS, ITEMS_PER_THREAD>(" << or_mask << ");\n";
            bool or_first = true;
            if (!emitPredicate(bin->left.get(), or_mask, or_first, true)) return false;
            if (!emitPredicate(bin->right.get(), or_mask, or_first, true)) return false;
            code << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, "
                 << target_mask << ", " << or_mask << ");\n";
            first_predicate = false;
            return true;
        }

        if (expr->getType() < ExprType::OP_EQ || expr->getType() > ExprType::OP_GTE) {
            return false;
        }

        const auto* cmp = static_cast<const BinaryExpr*>(expr);
        if (!cmp->left || !cmp->right) return false;

        const ExprNode* col_node = nullptr;
        const ExprNode* lit_node = nullptr;
        ExprType op = cmp->op_type;

        if (cmp->left->getType() == ExprType::COLUMN_REF &&
            (cmp->right->getType() == ExprType::LITERAL_INT ||
             cmp->right->getType() == ExprType::LITERAL_FLOAT)) {
            col_node = cmp->left.get();
            lit_node = cmp->right.get();
        } else if (cmp->right->getType() == ExprType::COLUMN_REF &&
                   (cmp->left->getType() == ExprType::LITERAL_INT ||
                    cmp->left->getType() == ExprType::LITERAL_FLOAT)) {
            col_node = cmp->right.get();
            lit_node = cmp->left.get();
            op = invertComparison(op);
        } else {
            return false;
        }

        const auto* col = static_cast<const ColumnRefExpr*>(col_node);
        const std::string& col_name = col->column_name;
        loadColumnToItems(col_name);

        std::string lit_val;
        if (lit_node->getType() == ExprType::LITERAL_INT) {
            lit_val = std::to_string(static_cast<const LiteralIntExpr*>(lit_node)->value);
        } else {
            lit_val = std::to_string(static_cast<const LiteralFloatExpr*>(lit_node)->value);
        }

        const std::string suffix = comparisonSuffix(op);
        std::string func;
        if (or_context) {
            // The OR accumulator is zero-initialised, so the first and later
            // predicates both have to OR into it.  Using BlockPredOr* for all
            // leaves avoids accidentally replacing previous OR terms.
            if (suffix == "Eq")       func = "BlockPredOrEq";
            else if (suffix == "NEq") func = "BlockPredOrNEq";
            else                       func = "BlockPredO" + suffix;
        } else {
            if (first_predicate) {
                func = "BlockPred" + suffix;
            } else {
                if (suffix == "Eq")       func = "BlockPredAEq";
                else if (suffix == "NEq") func = "BlockPredANEq";
                else                       func = "BlockPredA" + suffix;
            }
        }

        code << "            " << func << "<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, " << target_mask << ", " << lit_val << ", num_tile_items);\n";
        first_predicate = false;
        return true;
    };

    bool first = true;
    if (!emitPredicate(filter->predicate.get(), "flags", first, false)) {
        throw std::runtime_error("Unsupported build-side filter predicate for block codegen");
    }
    return current_items_col;
}

// ============================================================================
// emitPHTBuildKernel — single-pass PHT build (mirrors q21.cpp reference pattern)
// ============================================================================
void JITOperatorVisitor::emitPHTBuildKernel(
        const BuildInfo& bi,
        const FilterNode* build_filter,
        JITContext& ctx) {

    std::string kname = ctx.getNewBuildKernelName(bi.dim_prefix);
    ctx.startNewPipeline("Build_" + bi.dim_prefix);
    auto& code = ctx.current_pipeline->kernel_body;

    // HT size: 2× for open-addressing headroom
    std::string ht_size = "2*" + bi.ht_size_expr;
    ctx.hash_tables.push_back({bi.ht_name, "int",
                               bi.variant == 1 ? ht_size : "2*" + bi.ht_size_expr});

    // Register PK column as external
    if (!bi.pk_col.empty()) {
        markColumnNullability(ctx, catalog_, bi.pk_col);
        ctx.external_columns.insert("d_" + bi.pk_col);
        if (ctx.nullable_columns.count(bi.pk_col)) ctx.external_null_columns.insert(nullBitmapSymbolFor(bi.pk_col));
    }
    if (!bi.val_col.empty() && !bi.payload_is_row_id) {
        markColumnNullability(ctx, catalog_, bi.val_col);
        ctx.external_columns.insert("d_" + bi.val_col);
        if (ctx.nullable_columns.count(bi.val_col)) ctx.external_null_columns.insert(nullBitmapSymbolFor(bi.val_col));
    }
    if (build_filter && build_filter->predicate) {
        markExpressionNullability(ctx, catalog_, build_filter->predicate.get());
    }

    code << "    q.submit([&](sycl::handler& h) {\n";
    code << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "        size_t local  = BLOCK_THREADS;\n";
    code << "        size_t global = num_tiles * BLOCK_THREADS;\n";
    code << "        h.parallel_for<class " << kname << ">"
         << "(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {\n";
    code << "            int items[ITEMS_PER_THREAD];\n";
    if (bi.variant == 2) code << "            int items2[ITEMS_PER_THREAD];\n";
    code << "            int flags[ITEMS_PER_THREAD];\n";
    code << "            int tid = it.get_local_linear_id();\n";
    code << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    code << "            int num_tiles_local = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    code << "            int num_tile_items = TILE_SIZE;\n";
    code << "            if (it.get_group_linear_id() == num_tiles_local - 1) {\n";
    code << "                num_tile_items = " << bi.size_macro << " - tile_offset;\n";
    code << "            }\n";
    code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";

    // Emit dimension filter predicates (Option A: block-level BlockPred*)
    emitBuildFilter(build_filter, bi.dim_prefix, code, ctx);

    // Load PK key column into items
    if (!bi.pk_col.empty()) {
        code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
             << "d_" << bi.pk_col << " + tile_offset, tid, tile_offset, items, num_tile_items);\n";
        if (ctx.nullable_columns.count(bi.pk_col)) {
            code << "            int build_key_valid[ITEMS_PER_THREAD];\n";
            code << "            BlockLoadValidity<BLOCK_THREADS, ITEMS_PER_THREAD>("
                 << nullBitmapSymbolFor(bi.pk_col) << ", tid, tile_offset, build_key_valid, num_tile_items);\n";
            code << "            BlockApplyValidityAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, flags, build_key_valid, num_tile_items);\n";
        }
    }

    if (bi.variant == 1) {
        // PHT_1: key-only
        code << "            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, flags, " << bi.ht_name << ", "
             << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
    } else {
        // PHT_2: key+value. For projection/materialization the value is the
        // build-side row_id; for aggregation/filter pushdown it remains a
        // single scalar payload column.
        if (bi.payload_is_row_id) {
            code << "            BlockMakeRowIds<BLOCK_THREADS, ITEMS_PER_THREAD>"
                 << "(tid, tile_offset, items2, num_tile_items);\n";
        } else if (!bi.val_col.empty()) {
            code << "            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>("
                 << "d_" << bi.val_col << " + tile_offset, tid, tile_offset, items2, num_tile_items);\n";
        }
        code << "            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>"
             << "(tid, items, items2, flags, " << bi.ht_name << ", "
             << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
    }

    code << "        });\n    });\n\n";
}

// ============================================================================
// emitMHTBuildKernels — two-pass MHT build for 1-to-N joins (future use)
// ============================================================================
void JITOperatorVisitor::emitMHTBuildKernels(
        const BuildInfo& bi,
        const FilterNode* build_filter,
        JITContext& ctx) {

    std::string counts_name    = "d_" + bi.dim_prefix + "_counts";
    std::string offsets_name   = "d_" + bi.dim_prefix + "_offsets";
    std::string write_pos_name = "d_" + bi.dim_prefix + "_write_pos";
    std::string block_sums_name = "d_" + bi.dim_prefix + "_block_sums";

    if (!bi.pk_col.empty()) {
        markColumnNullability(ctx, catalog_, bi.pk_col);
        ctx.external_columns.insert("d_" + bi.pk_col);
        if (ctx.nullable_columns.count(bi.pk_col)) ctx.external_null_columns.insert(nullBitmapSymbolFor(bi.pk_col));
    }
    if (!bi.val_col.empty() && !bi.payload_is_row_id) {
        markColumnNullability(ctx, catalog_, bi.val_col);
        ctx.external_columns.insert("d_" + bi.val_col);
        if (ctx.nullable_columns.count(bi.val_col)) ctx.external_null_columns.insert(nullBitmapSymbolFor(bi.val_col));
    }
    if (build_filter && build_filter->predicate) {
        markExpressionNullability(ctx, catalog_, build_filter->predicate.get());
    }

    ctx.hash_tables.push_back({bi.ht_name,    "int", "3*" + bi.ht_size_expr});
    ctx.hash_tables.push_back({counts_name,   "int", bi.ht_size_expr});
    ctx.hash_tables.push_back({offsets_name,  "int", bi.ht_size_expr});
    ctx.hash_tables.push_back({write_pos_name,"int", bi.ht_size_expr});

    // ---- Pass 1: Count ----
    std::string kname1 = ctx.getNewBuildKernelName(bi.dim_prefix + "_cnt");
    ctx.startNewPipeline("MHT_Count_" + bi.dim_prefix);
    auto& c1 = ctx.current_pipeline->kernel_body;

    c1 << "    q.submit([&](sycl::handler& h) {\n";
    c1 << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c1 << "        h.parallel_for<class " << kname1 << ">"
       << "(sycl::nd_range<1>(num_tiles*BLOCK_THREADS, BLOCK_THREADS), [=](sycl::nd_item<1> it) {\n";
    c1 << "            int items[ITEMS_PER_THREAD]; int flags[ITEMS_PER_THREAD];\n";
    c1 << "            int tid = it.get_local_linear_id();\n";
    c1 << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    c1 << "            int ntiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c1 << "            int num_tile_items = (it.get_group_linear_id() == ntiles-1) ? "
       << bi.size_macro << " - tile_offset : TILE_SIZE;\n";
    c1 << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
    emitBuildFilter(build_filter, bi.dim_prefix, c1, ctx);
    if (!bi.pk_col.empty()) {
        c1 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.pk_col << "+tile_offset,tid,tile_offset,items,num_tile_items);\n";
        c1 << "            int build_key_valid[ITEMS_PER_THREAD];\n";
        c1 << "            BlockLoadValidity<BLOCK_THREADS,ITEMS_PER_THREAD>("
           << nullBitmapSymbolFor(bi.pk_col) << ",tid,tile_offset,build_key_valid,num_tile_items);\n";
        c1 << "            BlockApplyValidityAnd<BLOCK_THREADS,ITEMS_PER_THREAD>(tid,flags,build_key_valid,num_tile_items);\n";
    }
    c1 << "            BlockBuildMHT_Count<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
       << "(tid,items,flags," << counts_name << "," << bi.ht_size_expr
       << "," << bi.key_mins << ",num_tile_items);\n";
    c1 << "        });\n    });\n\n";

    // ---- 3-Step Global Prefix Sum (using exclusive_scan_over_group) ----
    ctx.startNewPipeline("MHT_PfxSum_" + bi.dim_prefix);
    auto& cs = ctx.current_pipeline->kernel_body;

    cs << "    {\n";
    cs << "        int num_ps_blocks = (" << bi.ht_size_expr << " + 255) / 256;\n";
    cs << "        int num_ps_super_blocks = (num_ps_blocks + 255) / 256;\n";
    cs << "        int* " << block_sums_name << " = sycl::malloc_device<int>(num_ps_blocks, q);\n";
    cs << "        int* " << block_sums_name << "_super = sycl::malloc_device<int>(num_ps_super_blocks, q);\n";

    // Level 0: scan counts[] into offsets[], write per-block sums.
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTBlockScan_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = static_cast<int>(it.get_global_linear_id());\n";
    cs << "                int lid = static_cast<int>(it.get_local_linear_id());\n";
    cs << "                int val = (gid < " << bi.ht_size_expr << ") ? " << counts_name << "[gid] : 0;\n";
    cs << "                int scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<int>{});\n";
    cs << "                if (gid < " << bi.ht_size_expr << ") " << offsets_name << "[gid] = scanned;\n";
    cs << "                if (lid == 255) " << block_sums_name << "[it.get_group_linear_id()] = scanned + val;\n";
    cs << "            });\n        });\n";

    // Level 1: scan block_sums[] in 256-entry groups and write super sums.
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTBlockSumsScan_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_super_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = static_cast<int>(it.get_global_linear_id());\n";
    cs << "                int lid = static_cast<int>(it.get_local_linear_id());\n";
    cs << "                int val = (gid < num_ps_blocks) ? " << block_sums_name << "[gid] : 0;\n";
    cs << "                int scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<int>{});\n";
    cs << "                if (gid < num_ps_blocks) " << block_sums_name << "[gid] = scanned;\n";
    cs << "                if (lid == 255) " << block_sums_name << "_super[it.get_group_linear_id()] = scanned + val;\n";
    cs << "            });\n        });\n";

    // Level 2: scan super sums. Supports up to 256^3 hash slots.
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTSuperSumsScan_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = static_cast<int>(it.get_global_linear_id());\n";
    cs << "                int val = (gid < num_ps_super_blocks) ? " << block_sums_name << "_super[gid] : 0;\n";
    cs << "                int scanned = sycl::exclusive_scan_over_group(it.get_group(), val, sycl::plus<int>{});\n";
    cs << "                if (gid < num_ps_super_blocks) " << block_sums_name << "_super[gid] = scanned;\n";
    cs << "            });\n        });\n";

    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTAddSuperSums_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_super_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = static_cast<int>(it.get_global_linear_id());\n";
    cs << "                if (gid < num_ps_blocks) " << block_sums_name << "[gid] += " << block_sums_name << "_super[it.get_group_linear_id()];\n";
    cs << "            });\n        });\n";
    cs << "        q.submit([&](sycl::handler& h) {\n";
    cs << "            h.parallel_for<class MHTAddSums_" << bi.dim_prefix << ">"
       << "(sycl::nd_range<1>(num_ps_blocks*256,256),[=](sycl::nd_item<1> it){\n";
    cs << "                int gid = static_cast<int>(it.get_global_linear_id());\n";
    cs << "                if (gid < " << bi.ht_size_expr << ") " << offsets_name << "[gid] += " << block_sums_name << "[it.get_group_linear_id()];\n";
    cs << "            });\n        });\n";

    cs << "        int mht_last_off=0, mht_last_cnt=0;\n";
    cs << "        q.memcpy(&mht_last_off," << offsets_name << "+" << bi.ht_size_expr << "-1,sizeof(int)).wait();\n";
    cs << "        q.memcpy(&mht_last_cnt," << counts_name << "+" << bi.ht_size_expr << "-1,sizeof(int)).wait();\n";
    cs << "        int payload_sz_" << bi.dim_prefix << " = mht_last_off + mht_last_cnt;\n";
    cs << "        if (payload_sz_" << bi.dim_prefix << " == 0) payload_sz_" << bi.dim_prefix << " = 1;\n";
    cs << "        const size_t mht_payload_bytes_" << bi.dim_prefix << " = static_cast<size_t>(payload_sz_" << bi.dim_prefix << ") * sizeof(int);\n";
    cs << "        const size_t mht_total_mem_" << bi.dim_prefix << " = static_cast<size_t>(q.get_device().get_info<sycl::info::device::global_mem_size>());\n";
    cs << "        if (ctx->loaded_device_bytes_ + mht_payload_bytes_" << bi.dim_prefix << " > mht_total_mem_" << bi.dim_prefix << ") {\n";
    cs << "            throw std::runtime_error(\"Insufficient GPU memory for MHT payload allocation\");\n";
    cs << "        }\n";
    cs << "        payload_" << bi.dim_prefix << " = sycl::malloc_device<int>(payload_sz_" << bi.dim_prefix << ", q);\n";
    cs << "        q.memset(" << write_pos_name << ", 0, " << bi.ht_size_expr << "*sizeof(int));\n";
    cs << "        sycl::free(" << block_sums_name << ", q);\n";
    cs << "        sycl::free(" << block_sums_name << "_super, q);\n";
    cs << "    }\n\n";
    // ---- Pass 2: Write ----
    std::string kname2 = ctx.getNewBuildKernelName(bi.dim_prefix + "_wrt");
    ctx.startNewPipeline("MHT_Write_" + bi.dim_prefix);
    auto& c2 = ctx.current_pipeline->kernel_body;

    c2 << "    q.submit([&](sycl::handler& h) {\n";
    c2 << "        int num_tiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c2 << "        h.parallel_for<class " << kname2 << ">"
       << "(sycl::nd_range<1>(num_tiles*BLOCK_THREADS,BLOCK_THREADS),[=](sycl::nd_item<1> it){\n";
    c2 << "            int items[ITEMS_PER_THREAD]; int items2[ITEMS_PER_THREAD];\n";
    c2 << "            int flags[ITEMS_PER_THREAD];\n";
    c2 << "            int tid = it.get_local_linear_id();\n";
    c2 << "            int tile_offset = it.get_group_linear_id() * TILE_SIZE;\n";
    c2 << "            int ntiles = (" << bi.size_macro << " + TILE_SIZE - 1) / TILE_SIZE;\n";
    c2 << "            int num_tile_items = (it.get_group_linear_id() == ntiles-1) ? "
       << bi.size_macro << " - tile_offset : TILE_SIZE;\n";
    c2 << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
    emitBuildFilter(build_filter, bi.dim_prefix, c2, ctx);
    if (!bi.pk_col.empty()) {
        c2 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.pk_col << "+tile_offset,tid,tile_offset,items,num_tile_items);\n";
        c2 << "            int build_key_valid[ITEMS_PER_THREAD];\n";
        c2 << "            BlockLoadValidity<BLOCK_THREADS,ITEMS_PER_THREAD>("
           << nullBitmapSymbolFor(bi.pk_col) << ",tid,tile_offset,build_key_valid,num_tile_items);\n";
        c2 << "            BlockApplyValidityAnd<BLOCK_THREADS,ITEMS_PER_THREAD>(tid,flags,build_key_valid,num_tile_items);\n";
    }
    if (bi.payload_is_row_id) {
        c2 << "            BlockMakeRowIds<BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(tid,tile_offset,items2,num_tile_items);\n";
    } else if (!bi.val_col.empty()) {
        c2 << "            BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>"
           << "(d_" << bi.val_col << "+tile_offset,tid,tile_offset,items2,num_tile_items);\n";
    }
    c2 << "            BlockBuildMHT_Write<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>"
       << "(tid,items,items2,flags," << bi.ht_name << "," << offsets_name << ","
       << write_pos_name << "," << counts_name << ",payload_" << bi.dim_prefix
       << "," << bi.ht_size_expr << "," << bi.key_mins << ",num_tile_items);\n";
    c2 << "        });\n    });\n\n";
}

void JITOperatorVisitor::produceHashJoin(const HashJoinNode* node, JITContext& ctx) {
    const auto& children = node->getChildren();
    if (children.size() < 2) return;

    auto is_dim_op = [&](const OperatorNode* n) {
        const TableScanNode* scan = nullptr;
        if (n->getType() == OperatorType::TABLE_SCAN) {
            scan = static_cast<const TableScanNode*>(n);
        } else if (n->getType() == OperatorType::FILTER && !n->getChildren().empty() && 
                   n->getChildren()[0]->getType() == OperatorType::TABLE_SCAN) {
            scan = static_cast<const TableScanNode*>(n->getChildren()[0].get());
        }
        if (scan && !scan->table_name.empty()) {
            try {
                return !catalog_.getTableMetadata(scan->table_name).isFactTable();
            } catch (...) { return false; }
        }
        return false;
    };

    const OperatorNode* build_side = nullptr;
    const OperatorNode* probe_side = nullptr;

    if (is_dim_op(children[0].get())) {
        build_side = children[0].get();
        probe_side = children[1].get();
    } else if (is_dim_op(children[1].get())) {
        build_side = children[1].get();
        probe_side = children[0].get();
    } else {
        return; // Both complex
    }

    const FilterNode*   build_filter = nullptr;
    const TableScanNode* build_scan  = nullptr;

    if (build_side->getType() == OperatorType::FILTER) {
        build_filter = static_cast<const FilterNode*>(build_side);
        build_scan = static_cast<const TableScanNode*>(build_filter->getChildren()[0].get());
    } else {
        build_scan = static_cast<const TableScanNode*>(build_side);
    }
    if (!build_scan) return;

    const std::string& dim_table = build_scan->table_name;

    // --- Compute BuildInfo ---
    BuildInfo bi = computeBuildInfo(dim_table, build_filter, node);

    // --- Route to PHT or MHT ---
    if (!bi.use_mht) {
        bi.variant = (bi.payload_is_row_id || !bi.val_col.empty()) ? 2 : 1;
        emitPHTBuildKernel(bi, build_filter, ctx);
    } else {
        bi.variant = 2;
        emitMHTBuildKernels(bi, build_filter, ctx);
    }

    build_infos_[node] = bi;

    // --- Trigger probe pipeline ---
    produce(probe_side, ctx);
}


static bool columnRequiredAboveNode(const OperatorNode* node, const std::string& col_name) {
    const OperatorNode* curr = node ? node->parent_ : nullptr;
    std::vector<std::string> cols;
    while (curr) {
        if (curr->getType() == OperatorType::AGGREGATE) {
            const auto* agg = static_cast<const AggregateNode*>(curr);
            for (const auto& a : agg->aggregates) extractAllColumns(a.agg_expr.get(), cols);
            for (const auto& gb : agg->group_by_exprs) extractAllColumns(gb.get(), cols);
        } else if (curr->getType() == OperatorType::FILTER) {
            const auto* flt = static_cast<const FilterNode*>(curr);
            extractAllColumns(flt->predicate.get(), cols);
        } else if (curr->getType() == OperatorType::PROJECTION) {
            const auto* proj = static_cast<const ProjectionNode*>(curr);
            for (const auto& e : proj->select_exprs) extractAllColumns(e.get(), cols);
        }
        curr = curr->parent_;
    }
    return std::find(cols.begin(), cols.end(), col_name) != cols.end();
}

// ============================================================================
// consumeHashJoinVector — vector mode probe (outside scalar loop)
// PHT (1-to-1): BlockProbeAndPHT → stay in vector mode → consumeVector
// MHT (1-to-N): open scalar expansion loop → consumeItem
// ============================================================================
void JITOperatorVisitor::consumeHashJoinVector(const HashJoinNode* node, JITContext& ctx,
                                                const OperatorNode* /*sender*/,
                                                const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;

    if (!build_infos_.count(node)) return;
    const BuildInfo& bi = build_infos_[node];
    if (bi.fk_cols.empty()) return;
    const std::string& fk = bi.fk_cols.front();

    if (!bi.use_mht) {
        // ---- PHT probe: stays in vector mode ----
        // A join condition may contain several fact keys joined to the same
        // dimension PK, for example:
        //   lo_orderdate = d_datekey OR lo_commitdate = d_datekey
        // SQL OR semantics require probing every FK alternative.  If two FK
        // values are equal, the dimension row must be emitted only once.
        const std::string payload_reg = bi.payload_is_row_id ? bi.row_id_reg : (bi.val_col.empty() ? "items2" : bi.val_col);
        if (bi.payload_is_row_id) {
            ctx.table_rowid_regs[bi.dim_table] = payload_reg;
            ctx.col_to_reg["__rowid_" + bi.dim_table] = payload_reg;
        } else if (!bi.val_col.empty()) {
            ctx.col_to_reg[bi.val_col] = payload_reg;
        }

        if (bi.fk_cols.size() > 1) {
            const std::string base_flags = "join_base_flags_" + std::to_string(++ctx.mask_counter);
            code << "            int " << base_flags << "[ITEMS_PER_THREAD];\n";
            code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << base_flags << ");\n";
            code << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, "
                 << base_flags << ", flags);\n";

            for (std::size_t fk_idx = 0; fk_idx < bi.fk_cols.size(); ++fk_idx) {
                const std::string& fk_col = bi.fk_cols[fk_idx];
                loadIntoReg(fk_col, fk_col, ctx);

                code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
                code << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, flags, "
                     << base_flags << ");\n";
                if (ctx.col_to_valid_reg.count(fk_col)) {
                    code << "            BlockApplyValidityAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, flags, "
                         << ctx.col_to_valid_reg[fk_col] << ", num_tile_items);\n";
                }

                // Suppress duplicate OR matches.  If the current FK equals any
                // previous FK for the same fact row, that dimension key was
                // already probed and should not be aggregated again.
                for (std::size_t prev = 0; prev < fk_idx; ++prev) {
                    const std::string& prev_fk = bi.fk_cols[prev];
                    loadIntoReg(prev_fk, prev_fk, ctx);
                    const std::string prev_valid = ctx.col_to_valid_reg.count(prev_fk) ? ctx.col_to_valid_reg[prev_fk] + "[i]" : "1";
                    code << "            #pragma unroll\n";
                    code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
                    code << "                if (tid + BLOCK_THREADS * i < num_tile_items) {\n";
                    code << "                    flags[i] = flags[i] && (!(" << prev_valid << ") || db::safe_neq(" << fk_col << "[i], " << prev_fk << "[i]));\n";
                    code << "                }\n";
                    code << "            }\n";
                }

                if (bi.variant == 1) {
                    code << "            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                         << "(tid, " << fk_col << ", flags, " << bi.ht_name << ", "
                         << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
                } else {
                    code << "            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                         << "(tid, " << fk_col << ", " << payload_reg << ", flags, "
                         << bi.ht_name << ", " << bi.ht_size_expr << ", "
                         << bi.key_mins << ", num_tile_items);\n";
                }

                if (node->parent_) {
                    consume_mode_ = ConsumeMode::Vector;
                    consume(node->parent_, ctx, node, active_vars);
                }
            }

            // Leave flags restored to the incoming mask for any code that may
            // be appended after this join block.
            code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);\n";
            code << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, flags, "
                 << base_flags << ");\n";
            return;
        }

        // Single-FK fast path.  Use items[] as transient key scratch unless
        // an ancestor actually needs the FK value later.  This preserves the
        // hand-written SSB register shape and reduces VGPR pressure.
        const bool keep_fk_reg = columnRequiredAboveNode(node, fk);
        const std::string key_reg = keep_fk_reg ? fk : "items";
        loadIntoReg(fk, key_reg, ctx);
        if (ctx.col_to_valid_reg.count(fk)) {
            code << "            BlockApplyValidityAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, flags, "
                 << ctx.col_to_valid_reg[fk] << ", num_tile_items);\n";
        }

        if (bi.variant == 1) {
            code << "            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                 << "(tid, " << key_reg << ", flags, " << bi.ht_name << ", "
                 << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
        } else {
            code << "            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>"
                 << "(tid, " << key_reg << ", " << payload_reg << ", flags, " << bi.ht_name << ", "
                 << bi.ht_size_expr << ", " << bi.key_mins << ", num_tile_items);\n";
        }
        if (!keep_fk_reg) {
            ctx.col_to_reg.erase(fk);
            ctx.col_to_valid_reg.erase(fk);
        }

        // Still in vector mode — pass to parent
        if (node->parent_) {
            consume_mode_ = ConsumeMode::Vector;
            consume(node->parent_, ctx, node, active_vars);
        }

    } else {
        // ---- MHT probe: row-expanding 1-to-N path ----
        // Preload every probe-side column required by parents before entering
        // the scalar expansion loop.  Item-mode consumers must not emit block
        // loads inside the j-loop.
        for (const auto& col : active_vars) {
            if (getTableName(col) != bi.dim_table) {
                loadIntoReg(col, col, ctx);
            }
        }

        const std::string base_flags = "mht_base_flags_" + std::to_string(++ctx.mask_counter);
        code << "            int " << base_flags << "[ITEMS_PER_THREAD];\n";
        code << "            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(" << base_flags << ");\n";
        code << "            BlockApplyMaskAnd<BLOCK_THREADS, ITEMS_PER_THREAD>(tid, " << base_flags << ", flags);\n";

        for (std::size_t fk_idx = 0; fk_idx < bi.fk_cols.size(); ++fk_idx) {
            const std::string& fk_col = bi.fk_cols[fk_idx];
            loadIntoReg(fk_col, fk_col, ctx);
            const std::string fk_valid = ctx.col_to_valid_reg.count(fk_col) ? ctx.col_to_valid_reg[fk_col] + "[i]" : "1";

            code << "            #pragma unroll\n";
            code << "            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {\n";
            code << "                int branch_flag = " << base_flags << "[i] && (" << fk_valid << ") && (tid + BLOCK_THREADS * i < num_tile_items);\n";
            for (std::size_t prev = 0; prev < fk_idx; ++prev) {
                const std::string& prev_fk = bi.fk_cols[prev];
                loadIntoReg(prev_fk, prev_fk, ctx);
                const std::string prev_valid = ctx.col_to_valid_reg.count(prev_fk) ? ctx.col_to_valid_reg[prev_fk] + "[i]" : "1";
                code << "                branch_flag = branch_flag && (!(" << prev_valid << ") || db::safe_neq(" << fk_col << "[i], " << prev_fk << "[i]));\n";
            }
            code << "                if (branch_flag) {\n";
            code << "                    int offset = 0, count = 0;\n";
            code << "                    ProbeMultiHT(" << fk_col << "[i], offset, count, branch_flag, "
                 << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ");\n";
            code << "                    for (int j = 0; j < count; ++j) {\n";

            std::vector<std::string> new_vars = active_vars;
            if (bi.payload_is_row_id) {
                ctx.table_rowid_regs[bi.dim_table] = bi.row_id_reg;
                ctx.col_to_reg["__rowid_" + bi.dim_table] = bi.row_id_reg;
                code << "                        " << bi.row_id_reg << "[i] = payload_"
                     << bi.dim_prefix << "[offset + j];\n";
            } else if (!bi.val_col.empty()) {
                ctx.col_to_reg[bi.val_col] = bi.val_col;
                ctx.col_to_valid_reg[bi.val_col] = "";
                code << "                        " << bi.val_col << "[i] = payload_"
                     << bi.dim_prefix << "[offset + j];\n";
                new_vars.push_back(bi.val_col);
            }

            if (node->parent_) {
                consume_mode_ = ConsumeMode::Item;
                consume(node->parent_, ctx, node, new_vars);
            }

            code << "                    }\n";
            code << "                }\n";
            code << "            }\n";
        }
    }
}

// ============================================================================
// consumeHashJoinItem — PHT probe inside a scalar expansion loop (rare)
// ============================================================================
void JITOperatorVisitor::consumeHashJoinItem(const HashJoinNode* node, JITContext& ctx,
                                              const OperatorNode* /*sender*/,
                                              const std::vector<std::string>& active_vars) {
    auto& code = ctx.current_pipeline->kernel_body;
    if (!build_infos_.count(node)) return;
    const BuildInfo& bi = build_infos_[node];
    if (bi.fk_cols.empty()) return;
    const std::string& fk = bi.fk_cols.front();

    // In item mode, we already have a scalar 'i' in scope.
    if (bi.variant == 1) {
        code << "                        if (ProbePHT_1(" << fk << "[i], "
             << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ")) {\n";
    } else {
        code << "                        int ht_val = 0;\n";
        code << "                        if (ProbePHT_2(" << fk << "[i], ht_val, "
             << bi.ht_name << ", " << bi.ht_size_expr << ", " << bi.key_mins << ")) {\n";
        if (!bi.val_col.empty())
            code << "                            " << bi.val_col << "[i] = ht_val;\n";
    }
    std::vector<std::string> new_vars = active_vars;
    if (!bi.val_col.empty()) new_vars.push_back(bi.val_col);
    if (node->parent_) {
        consume_mode_ = ConsumeMode::Item;
        consume(node->parent_, ctx, node, new_vars);
    }
    code << "                        }\n";
}



// ============================================================================
// generateCode
// ============================================================================

std::string JITOperatorVisitor::generateCode() const {
    std::stringstream code;

    code << "#include \"core/execution.h\"\n";
    code << "#include <sycl/sycl.hpp>\n";
    code << "#include <limits>\n";
    code << "#include <stdexcept>\n";
    code << "#include <vector>\n";
    code << "#include <cstddef>\n";
    code << "#include <chrono>\n";
    code << "#include \"crystal/load.h\"\n";
    code << "#include \"crystal/pred.h\"\n";
    code << "#include \"crystal/join.h\"\n";
    code << "#include \"crystal/utils.h\"\n";
    code << "#include \"core/inline_math.h\"\n\n";
    code << "using namespace sycl;\n\n";
    code << "constexpr int BLOCK_THREADS = 128;\n";
    code << "constexpr int ITEMS_PER_THREAD = 4;\n";
    code << "constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;\n\n";

    // Forward-declare only kernel classes that are actually used as
    // parallel_for<class ...> names.  Build_* pipeline labels are internal
    // bookkeeping and must not leak into generated C++.
    for (const auto& p : ctx_.pipelines) {
        if (p.kernel_name.rfind("Scan_", 0) == 0) {
            code << "class " << p.kernel_name << ";\n";
        }
    }
    for (const auto& kname : ctx_.kernel_class_names) {
        code << "class " << kname << ";\n";
    }

    code << "\nextern \"C\" void execute_query(db::ExecutionContext* ctx) {\n";
    code << "    auto __crystal_generated_start = std::chrono::high_resolution_clock::now();\n";
    code << "    sycl::queue& q = *(ctx->q_);\n";

    for (const auto& col : ctx_.external_columns) {
        code << "    int* " << col << " = ctx->getBuffer<int>(\"" << col << "\");\n";
    }
    for (const auto& ncol : ctx_.external_null_columns) {
        std::string data_name = ncol;
        if (data_name.rfind("n_", 0) == 0) data_name = "d_" + data_name.substr(2);
        code << "    uint64_t* " << ncol << " = ctx->getNullBitmap(\"" << data_name << "\");\n";
    }

    code << "    unsigned long long* d_result = ctx->getResultPointer();\n";
    code << "    uint64_t* d_result_validity = ctx->getResultValidityPointer();\n";
    for (const auto& pair : build_infos_) {
        const auto& bi = pair.second;
        if (bi.use_mht) {
            code << "    int* payload_" << bi.dim_prefix << " = nullptr;\n";
        }
    }

    for (const auto& ht : ctx_.hash_tables) {
        if (ht.type == "int") {
            code << "    int* " << ht.name << " = ctx->config_.reuse_scratch_buffers ? ctx->getScratchIntBuffer(\"" << ht.name << "\", static_cast<size_t>(" << ht.size_expr << ")) : sycl::malloc_device<int>(" << ht.size_expr << ", q);\n";
        } else if (ht.type == "unsigned long long") {
            code << "    unsigned long long* " << ht.name << " = ctx->config_.reuse_scratch_buffers ? ctx->getScratchUInt64Buffer(\"" << ht.name << "\", static_cast<size_t>(" << ht.size_expr << ")) : sycl::malloc_device<unsigned long long>(" << ht.size_expr << ", q);\n";
        } else {
            code << "    " << ht.type << "* " << ht.name
                 << " = sycl::malloc_device<" << ht.type << ">(" 
                 << ht.size_expr << ", q);\n";
        }
        code << "    q.memset(" << ht.name << ", 0, ("
             << ht.size_expr << ") * sizeof(" << ht.type << "));\n";
    }



    code << "    q.memset(d_result, 0, " << ctx_.result_size_expr
         << " * sizeof(unsigned long long));\n\n";

    for (const auto& p : ctx_.pipelines) {
        code << p.includes_and_globals.str();
        code << p.kernel_body.str();
    }

    code << ctx_.post_execution_code.str();

    code << "    q.wait();\n\n";
    for (const auto& ht : ctx_.hash_tables) {
        if (ht.type == "int" || ht.type == "unsigned long long") {
            code << "    if (!ctx->config_.reuse_scratch_buffers) sycl::free(" << ht.name << ", q);\n";
        } else {
            code << "    sycl::free(" << ht.name << ", q);\n";
        }
    }



    // Free MHT-only allocations (payload chunks)
    for (const auto& pair : build_infos_) {
        const auto& bi = pair.second;
        if (bi.use_mht) {
            code << "    if (payload_" << bi.dim_prefix << ") sycl::free(payload_" << bi.dim_prefix << ", q);\n";
        }
    }

    code << "    auto __crystal_generated_end = std::chrono::high_resolution_clock::now();\n";
    code << "    ctx->timing_.gpu_execute_ms = std::chrono::duration<double, std::milli>(__crystal_generated_end - __crystal_generated_start).count();\n";
    code << "    ctx->timing_.jit_execute_ms = ctx->timing_.gpu_execute_ms;\n";
    code << "    ctx->tuple_size_ = " << ctx_.tuple_size << ";\n";
    code << "}\n";

    return code.str();
}

} // namespace db
