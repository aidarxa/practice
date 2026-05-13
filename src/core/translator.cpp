#include "core/translator.h"

#include <algorithm>
#include <cctype>
#include <ostream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// ============================================================================
// Block 1: ExprPrinter implementation (definitions живут здесь, а не в .h)
// ============================================================================

namespace db {

// --- accept() definitions ---

void ColumnRefExpr::accept(ExprVisitor& visitor) const  { visitor.visit(*this); }
void LiteralIntExpr::accept(ExprVisitor& visitor) const { visitor.visit(*this); }
void LiteralFloatExpr::accept(ExprVisitor& visitor) const { visitor.visit(*this); }
void LiteralNullExpr::accept(ExprVisitor& visitor) const { visitor.visit(*this); }
void BinaryExpr::accept(ExprVisitor& visitor) const     { visitor.visit(*this); }
void CaseWhenExpr::accept(ExprVisitor& visitor) const   { visitor.visit(*this); }
void StarExpr::accept(ExprVisitor& visitor) const       { visitor.visit(*this); }

// --- ExprPrinter ---

void ExprPrinter::printIndent() const {
    for (int i = 0; i < indent_; ++i) out_ << "  ";
}

const char* ExprPrinter::exprTypeName(ExprType t) {
    switch (t) {
        case ExprType::COLUMN_REF:    return "COLUMN_REF";
        case ExprType::LITERAL_INT:   return "LITERAL_INT";
        case ExprType::LITERAL_FLOAT: return "LITERAL_FLOAT";
        case ExprType::LITERAL_NULL:  return "LITERAL_NULL";
        case ExprType::OP_AND:        return "AND";
        case ExprType::OP_OR:         return "OR";
        case ExprType::OP_NOT:        return "NOT";
        case ExprType::OP_EQ:         return "=";
        case ExprType::OP_NEQ:        return "<>";
        case ExprType::OP_LT:         return "<";
        case ExprType::OP_LTE:        return "<=";
        case ExprType::OP_GT:         return ">";
        case ExprType::OP_GTE:        return ">=";
        case ExprType::OP_ADD:        return "+";
        case ExprType::OP_SUB:        return "-";
        case ExprType::OP_MUL:        return "*";
        case ExprType::OP_DIV:        return "/";
        case ExprType::OP_IS_NULL:    return "IS NULL";
        case ExprType::OP_IS_NOT_NULL:return "IS NOT NULL";
        case ExprType::CASE_WHEN:     return "CASE WHEN";
        case ExprType::STAR:          return "*";
        default:                      return "UNKNOWN";
    }
}

void ExprPrinter::visit(const ColumnRefExpr& node) {
    printIndent();
    out_ << "[ColumnRef] ";
    if (!node.table_name.empty()) out_ << node.table_name << ".";
    out_ << node.column_name << "\n";
}

void ExprPrinter::visit(const LiteralIntExpr& node) {
    printIndent();
    out_ << "[LiteralInt] " << node.value << "\n";
}

void ExprPrinter::visit(const LiteralFloatExpr& node) {
    printIndent();
    out_ << "[LiteralFloat] " << node.value << "\n";
}

void ExprPrinter::visit(const LiteralNullExpr& /*node*/) {
    printIndent();
    out_ << "[LiteralNull] NULL\n";
}

void ExprPrinter::visit(const BinaryExpr& node) {
    printIndent();
    out_ << "[BinaryExpr] " << exprTypeName(node.op_type) << "\n";
    if (node.left) {
        ExprPrinter lp(out_, indent_ + 1);
        node.left->accept(lp);
    }
    if (node.right) {
        ExprPrinter rp(out_, indent_ + 1);
        node.right->accept(rp);
    }
}


void ExprPrinter::visit(const CaseWhenExpr& node) {
    printIndent();
    out_ << "[CaseWhen]\n";
    printIndent();
    out_ << "  WHEN:\n";
    {
        ExprPrinter cp(out_, indent_ + 2);
        node.condition->accept(cp);
    }
    printIndent();
    out_ << "  THEN:\n";
    {
        ExprPrinter tp(out_, indent_ + 2);
        node.then_expr->accept(tp);
    }
    printIndent();
    out_ << "  ELSE:\n";
    {
        ExprPrinter ep(out_, indent_ + 2);
        node.else_expr->accept(ep);
    }
}

void ExprPrinter::visit(const StarExpr& /*node*/) {
    printIndent();
    out_ << "[Star] *\n";
}

// ============================================================================
// Block 2: OperatorNode accept() definitions + OperatorPrinter
// ============================================================================

void TableScanNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }
void FilterNode::accept(OperatorVisitor& visitor) const    { visitor.visit(*this); }
void HashJoinNode::accept(OperatorVisitor& visitor) const  { visitor.visit(*this); }
void AggregateNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }
void ProjectionNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }
void SortLimitNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }

// --- OperatorPrinter ---

void OperatorPrinter::printIndent() const {
    for (int i = 0; i < indent_; ++i) out_ << "  ";
}

void OperatorPrinter::visitChildren(const OperatorNode& node) {
    for (const auto& child : node.getChildren()) {
        if (child) {
            OperatorPrinter cp(out_, indent_ + 1);
            child->accept(cp);
        }
    }
}

void OperatorPrinter::visit(const TableScanNode& node) {
    printIndent();
    out_ << "[TableScan] " << node.table_name;
    if (!node.table_alias.empty()) out_ << " AS " << node.table_alias;
    out_ << "\n";
    // TableScan — листовой узел, детей нет
}

void OperatorPrinter::visit(const FilterNode& node) {
    printIndent();
    out_ << "[Filter]\n";
    if (node.predicate) {
        ExprPrinter ep(out_, indent_ + 1);
        node.predicate->accept(ep);
    }
    visitChildren(node);
}

void OperatorPrinter::visit(const HashJoinNode& node) {
    printIndent();
    out_ << "[HashJoin]";
    if (node.join_condition) {
        out_ << " on:\n";
        ExprPrinter ep(out_, indent_ + 1);
        node.join_condition->accept(ep);
    } else {
        out_ << " (no condition — naive phase)\n";
    }
    visitChildren(node);
}

void OperatorPrinter::visit(const ProjectionNode& node) {
    printIndent();
    out_ << "[Projection]\n";
    for (std::size_t i = 0; i < node.select_exprs.size(); ++i) {
        const auto& expr = node.select_exprs[i];
        if (expr) {
            ExprPrinter ep(out_, indent_ + 1);
            expr->accept(ep);
            if (i < node.output_aliases.size() && !node.output_aliases[i].empty()) {
                printIndent();
                out_ << "  AS " << node.output_aliases[i] << "\n";
            }
        }
    }
    visitChildren(node);
}

void OperatorPrinter::visit(const SortLimitNode& node) {
    printIndent();
    out_ << "[SortLimit]";
    if (!node.sort_keys.empty()) {
        out_ << " ORDER BY";
        for (const auto& key : node.sort_keys) {
            out_ << " #" << key.column_index << (key.descending ? " DESC" : " ASC");
        }
    }
    if (node.has_limit) out_ << " LIMIT " << node.limit;
    out_ << "\n";
    visitChildren(node);
}

void OperatorPrinter::visit(const AggregateNode& node) {
    printIndent();
    out_ << "[Aggregate]\n";

    // Вывод GROUP BY
    if (!node.group_by_exprs.empty()) {
        printIndent();
        out_ << "  GROUP BY:\n";
        for (const auto& g : node.group_by_exprs) {
            ExprPrinter ep(out_, indent_ + 2);
            g->accept(ep);
        }
    }

    // Вывод агрегаций
    if (!node.aggregates.empty()) {
        printIndent();
        out_ << "  AGGREGATIONS:\n";
        for (const auto& agg : node.aggregates) {
            printIndent();
            out_ << "    " << agg.func_name << "(\n";
            if (agg.agg_expr) {
                ExprPrinter ep(out_, indent_ + 3);
                agg.agg_expr->accept(ep);
            }
            printIndent();
            out_ << "    )\n";
        }
    }

    if (node.having_predicate) {
        printIndent();
        out_ << "  HAVING:\n";
        ExprPrinter ep(out_, indent_ + 2);
        node.having_predicate->accept(ep);
    }

    visitChildren(node);
}

// ============================================================================
// Block 3: QueryTranslator
// ============================================================================

// Маппинг HSQL бинарных операторов → ExprType
ExprType QueryTranslator::mapBinaryOp(hsql::OperatorType op) {
    switch (op) {
        case hsql::kOpAnd:       return ExprType::OP_AND;
        case hsql::kOpOr:        return ExprType::OP_OR;
        case hsql::kOpEquals:    return ExprType::OP_EQ;
        case hsql::kOpNotEquals: return ExprType::OP_NEQ;
        case hsql::kOpLess:      return ExprType::OP_LT;
        case hsql::kOpLessEq:    return ExprType::OP_LTE;
        case hsql::kOpGreater:   return ExprType::OP_GT;
        case hsql::kOpGreaterEq: return ExprType::OP_GTE;
        case hsql::kOpPlus:      return ExprType::OP_ADD;
        case hsql::kOpMinus:     return ExprType::OP_SUB;
        case hsql::kOpAsterisk:  return ExprType::OP_MUL;
        case hsql::kOpSlash:     return ExprType::OP_DIV;
        default:
            throw std::runtime_error("QueryTranslator: unsupported binary operator type: "
                                     + std::to_string(static_cast<int>(op)));
    }
}


static std::string translatorUpperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

static bool isAggregateFunctionName(const std::string& name) {
    const std::string upper = translatorUpperAscii(name);
    return upper == "COUNT" || upper == "SUM" || upper == "MIN" ||
           upper == "MAX" || upper == "AVG";
}

namespace {

struct HsqlCaseArm {
    const hsql::Expr* when_expr;
    const hsql::Expr* then_expr;
};

static bool isCaseListElementNode(const hsql::Expr* expr) {
    return expr && expr->type == hsql::kExprOperator &&
           expr->opType == hsql::kOpCaseListElement;
}

static void collectCaseArmsFromNode(const hsql::Expr* node,
                                    std::vector<HsqlCaseArm>& out) {
    if (!node) return;

    if (isCaseListElementNode(node)) {
        if (!node->expr || !node->expr2) {
            throw std::runtime_error("QueryTranslator: malformed CASE WHEN arm");
        }
        out.push_back({node->expr, node->expr2});
        return;
    }

    if (node->exprList) {
        for (const hsql::Expr* item : *node->exprList) {
            collectCaseArmsFromNode(item, out);
        }
        return;
    }

    // Some hyrise/sql-parser revisions build the case-list as a linked
    // operator node rather than exposing Expr::exprList on the top-level CASE.
    if (node->type == hsql::kExprOperator && node->opType == hsql::kOpCase) {
        if (node->exprList) {
            for (const hsql::Expr* item : *node->exprList) collectCaseArmsFromNode(item, out);
            return;
        }
        if (isCaseListElementNode(node->expr)) {
            collectCaseArmsFromNode(node->expr, out);
            return;
        }
    }
}

static bool nodeLooksLikeCaseList(const hsql::Expr* node) {
    if (!node) return false;
    if (isCaseListElementNode(node)) return true;
    if (node->exprList && node->type == hsql::kExprOperator &&
        (node->opType == hsql::kOpCase || node->opType == hsql::kOpCaseListElement)) {
        return true;
    }
    return false;
}

} // namespace

std::unique_ptr<ExprNode> QueryTranslator::translateCaseExpr(const hsql::Expr* expr) {
    if (!expr || expr->type != hsql::kExprOperator || expr->opType != hsql::kOpCase) {
        throw std::runtime_error("QueryTranslator::translateCaseExpr: expected Hyrise kOpCase expression");
    }

    std::vector<HsqlCaseArm> arms;
    if (expr->exprList) {
        for (const hsql::Expr* item : *expr->exprList) collectCaseArmsFromNode(item, arms);
    }

    const hsql::Expr* base_expr = nullptr;
    if (expr->expr) {
        if (nodeLooksLikeCaseList(expr->expr)) {
            collectCaseArmsFromNode(expr->expr, arms);
        } else {
            base_expr = expr->expr;
        }
    }

    if (arms.empty()) {
        throw std::runtime_error("QueryTranslator: CASE expression contains no WHEN arms");
    }

    std::unique_ptr<ExprNode> else_expr = expr->expr2
        ? translateExpr(expr->expr2)
        : std::make_unique<LiteralNullExpr>();

    for (auto it = arms.rbegin(); it != arms.rend(); ++it) {
        std::unique_ptr<ExprNode> condition;
        if (base_expr) {
            condition = std::make_unique<BinaryExpr>(
                ExprType::OP_EQ,
                translateExpr(base_expr),
                translateExpr(it->when_expr));
        } else {
            condition = translateExpr(it->when_expr);
        }
        else_expr = std::make_unique<CaseWhenExpr>(
            std::move(condition),
            translateExpr(it->then_expr),
            std::move(else_expr));
    }

    return else_expr;
}

// Рекурсивная трансляция hsql::Expr* → ExprNode
std::unique_ptr<ExprNode> QueryTranslator::translateExpr(const hsql::Expr* expr) {
    if (!expr) {
        throw std::runtime_error("QueryTranslator::translateExpr: null expression");
    }

    switch (expr->type) {
        // ---- SQL star ----
        case hsql::kExprStar:
            return std::make_unique<StarExpr>();

        // ---- Ссылка на колонку ----
        case hsql::kExprColumnRef: {
            std::string col  = expr->name  ? std::string(expr->name)  : "";
            std::string tbl  = expr->table ? std::string(expr->table) : "";
            if (col == "*") return std::make_unique<StarExpr>();
            tbl = resolveTableQualifier(tbl);
            return std::make_unique<ColumnRefExpr>(std::move(col), std::move(tbl));
        }

        // ---- Целочисленный литерал ----
        case hsql::kExprLiteralInt:
            return std::make_unique<LiteralIntExpr>(expr->ival);

        // ---- Вещественный литерал ----
        case hsql::kExprLiteralFloat:
            return std::make_unique<LiteralFloatExpr>(expr->fval);

        // ---- SQL NULL literal ----
        case hsql::kExprLiteralNull:
            return std::make_unique<LiteralNullExpr>();

        // ---- Бинарные/логические операторы ----
        case hsql::kExprOperator: {
            if (expr->opType == hsql::kOpCase) {
                return translateCaseExpr(expr);
            }

            // SQL NULL predicates. Hyrise SQL parser represents both
            // `x IS NULL` and `x IS NOT NULL` as unary operators in Expr.
            if (expr->opType == hsql::kOpIsNull) {
                if (!expr->expr) {
                    throw std::runtime_error("QueryTranslator: IS NULL missing operand");
                }
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_IS_NULL, translateExpr(expr->expr), nullptr);
            }
            if (expr->opType == hsql::kOpNot && expr->expr &&
                expr->expr->type == hsql::kExprOperator &&
                expr->expr->opType == hsql::kOpIsNull) {
                if (!expr->expr->expr) {
                    throw std::runtime_error("QueryTranslator: IS NOT NULL missing operand");
                }
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_IS_NOT_NULL, translateExpr(expr->expr->expr), nullptr);
            }
            if (expr->opType == hsql::kOpNot) {
                if (!expr->expr) {
                    throw std::runtime_error("QueryTranslator: NOT missing operand");
                }
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_NOT, translateExpr(expr->expr), nullptr);
            }

            // BETWEEN не обрабатываем — должно быть развёрнуто в >= и <= на уровне SQL
            if (expr->opType == hsql::kOpBetween) {
                throw std::runtime_error(
                    "QueryTranslator: BETWEEN is not supported. "
                    "Rewrite as col >= low AND col <= high.");
            }

            ExprType op = mapBinaryOp(expr->opType);

            // kOpUnaryMinus, kOpNot и прочие унарные — не поддерживаем
            if (!expr->expr) {
                throw std::runtime_error(
                    "QueryTranslator: unsupported unary operator or missing left operand");
            }
            if (!expr->expr2) {
                throw std::runtime_error(
                    "QueryTranslator: missing right operand in binary expression");
            }

            auto left  = translateExpr(expr->expr);
            auto right = translateExpr(expr->expr2);
            return std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
        }

        // ---- Scalar function references ----
        case hsql::kExprFunctionRef: {
            std::string func = expr->name ? std::string(expr->name) : "";
            func = translatorUpperAscii(func);
            if (func == "CASE_WHEN") {
                // Backward-compatible internal scalar kept for older benchmark SQL.
                if (!expr->exprList || expr->exprList->size() != 3) {
                    throw std::runtime_error("CASE_WHEN requires exactly 3 arguments: condition, then, else");
                }
                return std::make_unique<CaseWhenExpr>(
                    translateExpr((*expr->exprList)[0]),
                    translateExpr((*expr->exprList)[1]),
                    translateExpr((*expr->exprList)[2]));
            }
            throw std::runtime_error(
                "QueryTranslator: unsupported scalar function reference: " + func);
        }

        // ---- Прочие типы ----
        default:
            throw std::runtime_error(
                "QueryTranslator::translateExpr: unsupported expression type: "
                + std::to_string(static_cast<int>(expr->type)));
    }
}

// Трансляция FROM-секции → поддерево OperatorNode
std::unique_ptr<OperatorNode> QueryTranslator::translateFrom(const hsql::TableRef* ref) {
    if (!ref) {
        throw std::runtime_error("QueryTranslator::translateFrom: null TableRef");
    }

    switch (ref->type) {
        // Простая именованная таблица
        case hsql::kTableName: {
            std::string name = ref->name ? std::string(ref->name) : "";
            // Переводим в верхний регистр для совместимости с Catalog (LINEORDER, DDATE, ...)
            std::transform(name.begin(), name.end(), name.begin(), ::toupper);
            std::string alias;
            if (ref->alias && ref->alias->name) {
                alias = std::string(ref->alias->name);
                std::transform(alias.begin(), alias.end(), alias.begin(), ::toupper);
            }
            table_alias_to_name_[name] = name;
            if (!alias.empty()) table_alias_to_name_[alias] = name;
            return std::make_unique<TableScanNode>(std::move(name), std::move(alias));
        }

        // Кросс-декартово произведение (FROM t1, t2, t3, ...)
        // Строим левоассоциативное дерево HashJoinNode без условий:
        //   JOIN(JOIN(t1, t2), t3)
        case hsql::kTableCrossProduct: {
            if (!ref->list || ref->list->empty()) {
                throw std::runtime_error("QueryTranslator: kTableCrossProduct with empty list");
            }

            // Транслируем первую таблицу (обычно это таблица фактов, например LINEORDER)
            std::unique_ptr<OperatorNode> root = translateFrom((*ref->list)[0]);

            // Строим Right-Deep Tree: измерения всегда слева (build-side), факты справа (probe-side)
            for (std::size_t i = 1; i < ref->list->size(); ++i) {
                auto rhs = translateFrom((*ref->list)[i]);
                auto join = std::make_unique<HashJoinNode>(nullptr);
                join->addChild(std::move(rhs));   // build-side (левый потомок) - Измерение
                join->addChild(std::move(root));  // probe-side (правый потомок) - Таблица фактов / поддерево
                root = std::move(join);
            }
            return root;
        }

        // Явный JOIN с условием (kTableJoin)
        case hsql::kTableJoin: {
            if (!ref->join) {
                throw std::runtime_error(
                    "QueryTranslator: kTableJoin without JoinDefinition");
            }
            auto left = translateFrom(ref->join->left);
            auto right = translateFrom(ref->join->right);
            std::unique_ptr<ExprNode> cond = nullptr;
            if (ref->join->condition) {
                cond = translateExpr(ref->join->condition);
            }
            auto join = std::make_unique<HashJoinNode>(std::move(cond));
            join->addChild(std::move(left));
            join->addChild(std::move(right));
            return join;
        }

        // Подзапросы в FROM — не поддерживаются
        case hsql::kTableSelect:
            throw std::runtime_error(
                "QueryTranslator: subqueries in FROM are not supported");

        default:
            throw std::runtime_error(
                "QueryTranslator::translateFrom: unsupported TableRef type: "
                + std::to_string(static_cast<int>(ref->type)));
    }
}

static std::string translatorUpperCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

std::string QueryTranslator::resolveTableQualifier(const std::string& qualifier) const {
    if (qualifier.empty()) return "";
    std::string key = translatorUpperCopy(qualifier);
    auto it = table_alias_to_name_.find(key);
    if (it != table_alias_to_name_.end()) return it->second;
    return key;
}

void QueryTranslator::registerSelectAliases(const hsql::SelectStatement* stmt) {
    select_alias_to_select_index_.clear();
    if (!stmt || !stmt->selectList) return;
    for (std::size_t i = 0; i < stmt->selectList->size(); ++i) {
        const hsql::Expr* sel = (*stmt->selectList)[i];
        if (!sel || !sel->alias) continue;
        std::string alias = translatorUpperCopy(std::string(sel->alias));
        if (alias.empty()) continue;
        if (select_alias_to_select_index_.count(alias)) {
            throw std::runtime_error("QueryTranslator: duplicate SELECT alias: " + alias);
        }
        select_alias_to_select_index_[alias] = i;
    }
}

namespace {

static std::string upperCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

static bool hsqlExprEquivalent(const hsql::Expr* a, const hsql::Expr* b) {
    if (a == b) return true;
    if (!a || !b || a->type != b->type) return false;

    switch (a->type) {
        case hsql::kExprStar:
            return true;
        case hsql::kExprColumnRef: {
            const std::string an = a->name ? a->name : "";
            const std::string bn = b->name ? b->name : "";
            const std::string at = a->table ? a->table : "";
            const std::string bt = b->table ? b->table : "";
            return upperCopy(an) == upperCopy(bn) && (at.empty() || bt.empty() || upperCopy(at) == upperCopy(bt));
        }
        case hsql::kExprLiteralInt:
            return a->ival == b->ival;
        case hsql::kExprLiteralFloat:
            return a->fval == b->fval;
        case hsql::kExprLiteralNull:
            return true;
        case hsql::kExprFunctionRef: {
            const std::string an = a->name ? a->name : "";
            const std::string bn = b->name ? b->name : "";
            if (upperCopy(an) != upperCopy(bn)) return false;
            const auto asz = a->exprList ? a->exprList->size() : 0U;
            const auto bsz = b->exprList ? b->exprList->size() : 0U;
            if (asz != bsz) return false;
            for (std::size_t i = 0; i < asz; ++i) {
                if (!hsqlExprEquivalent((*a->exprList)[i], (*b->exprList)[i])) return false;
            }
            return true;
        }
        case hsql::kExprOperator: {
            if (a->opType != b->opType) return false;
            const auto asz = a->exprList ? a->exprList->size() : 0U;
            const auto bsz = b->exprList ? b->exprList->size() : 0U;
            if (asz != bsz) return false;
            for (std::size_t i = 0; i < asz; ++i) {
                if (!hsqlExprEquivalent((*a->exprList)[i], (*b->exprList)[i])) return false;
            }
            return hsqlExprEquivalent(a->expr, b->expr) &&
                   hsqlExprEquivalent(a->expr2, b->expr2);
        }
        default:
            return false;
    }
}

static std::size_t checkedOrderOrdinal(const hsql::Expr* expr,
                                       std::size_t visible_columns) {
    if (!expr || expr->type != hsql::kExprLiteralInt) {
        throw std::runtime_error("ORDER BY expression is not a select-list ordinal");
    }
    if (expr->ival <= 0 || static_cast<std::uint64_t>(expr->ival) > visible_columns) {
        throw std::runtime_error("ORDER BY ordinal is outside the SELECT result width");
    }
    return static_cast<std::size_t>(expr->ival - 1);
}

static bool resolveSelectAliasReference(
        const hsql::Expr* expr,
        const std::unordered_map<std::string, std::size_t>& alias_map,
        std::size_t& select_index) {
    if (!expr || expr->type != hsql::kExprColumnRef || expr->table) return false;
    const std::string name = expr->name ? upperCopy(std::string(expr->name)) : "";
    auto it = alias_map.find(name);
    if (it == alias_map.end()) return false;
    select_index = it->second;
    return true;
}

static std::size_t aggregateOutputSlotForSelectIndex(
        const hsql::SelectStatement* stmt,
        std::size_t select_index) {
    if (!stmt || !stmt->selectList || select_index >= stmt->selectList->size()) {
        throw std::runtime_error("SELECT alias points outside SELECT list");
    }
    const hsql::Expr* selected = (*stmt->selectList)[select_index];
    const std::size_t group_count = (stmt->groupBy && stmt->groupBy->columns)
        ? stmt->groupBy->columns->size()
        : 0U;

    if (selected && selected->isType(hsql::kExprFunctionRef) &&
        isAggregateFunctionName(selected->name ? std::string(selected->name) : std::string())) {
        std::size_t agg_idx = 0;
        for (std::size_t i = 0; i < stmt->selectList->size(); ++i) {
            const hsql::Expr* sel = (*stmt->selectList)[i];
            if (!sel || !sel->isType(hsql::kExprFunctionRef) ||
                !isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) continue;
            if (i == select_index) return group_count + agg_idx;
            ++agg_idx;
        }
    }

    if (stmt->groupBy && stmt->groupBy->columns) {
        for (std::size_t i = 0; i < stmt->groupBy->columns->size(); ++i) {
            if (hsqlExprEquivalent(selected, (*stmt->groupBy->columns)[i])) return i;
        }
    }

    throw std::runtime_error(
        "SELECT alias used in ORDER BY/HAVING must refer to a GROUP BY expression or aggregate output");
}

static std::size_t resolveProjectionOrderKey(
        const hsql::Expr* order_expr,
        const hsql::SelectStatement* stmt,
        const std::unordered_map<std::string, std::size_t>& alias_map) {
    if (!stmt->selectList || stmt->selectList->empty()) {
        throw std::runtime_error("ORDER BY requires a non-empty SELECT list");
    }
    if (order_expr && order_expr->type == hsql::kExprLiteralInt) {
        return checkedOrderOrdinal(order_expr, stmt->selectList->size());
    }
    std::size_t alias_select_index = 0;
    if (resolveSelectAliasReference(order_expr, alias_map, alias_select_index)) {
        if (alias_select_index >= stmt->selectList->size()) {
            throw std::runtime_error("ORDER BY alias points outside SELECT list");
        }
        return alias_select_index;
    }
    for (std::size_t i = 0; i < stmt->selectList->size(); ++i) {
        const hsql::Expr* sel = (*stmt->selectList)[i];
        if (sel && sel->type == hsql::kExprStar) {
            continue;
        }
        if (hsqlExprEquivalent(order_expr, sel)) return i;
    }
    throw std::runtime_error(
        "ORDER BY is supported only for visible SELECT expressions in projection queries");
}

static std::size_t resolveAggregateOrderKey(
        const hsql::Expr* order_expr,
        const hsql::SelectStatement* stmt,
        const std::unordered_map<std::string, std::size_t>& alias_map) {
    const std::size_t group_count = (stmt->groupBy && stmt->groupBy->columns)
        ? stmt->groupBy->columns->size()
        : 0U;
    std::size_t aggregate_count = 0;
    if (stmt->selectList) {
        for (const auto* sel : *stmt->selectList) {
            if (sel && sel->isType(hsql::kExprFunctionRef) &&
                isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) ++aggregate_count;
        }
    }
    const std::size_t visible_columns = group_count + aggregate_count;
    if (order_expr && order_expr->type == hsql::kExprLiteralInt) {
        return checkedOrderOrdinal(order_expr, visible_columns);
    }
    std::size_t alias_select_index = 0;
    if (resolveSelectAliasReference(order_expr, alias_map, alias_select_index)) {
        return aggregateOutputSlotForSelectIndex(stmt, alias_select_index);
    }

    if (stmt->groupBy && stmt->groupBy->columns) {
        for (std::size_t i = 0; i < stmt->groupBy->columns->size(); ++i) {
            if (hsqlExprEquivalent(order_expr, (*stmt->groupBy->columns)[i])) return i;
        }
    }

    std::size_t agg_idx = 0;
    if (stmt->selectList) {
        for (const auto* sel : *stmt->selectList) {
            if (!sel || !sel->isType(hsql::kExprFunctionRef) ||
                !isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) continue;
            if (hsqlExprEquivalent(order_expr, sel)) return group_count + agg_idx;
            ++agg_idx;
        }
    }
    throw std::runtime_error(
        "ORDER BY is supported only for GROUP BY expressions, aggregate expressions, or SELECT-list ordinals");
}

static std::size_t parseLimitValue(const hsql::LimitDescription* limit) {
    if (!limit || !limit->limit) return 0;
    if (limit->limit->type != hsql::kExprLiteralInt || limit->limit->ival < 0) {
        throw std::runtime_error("LIMIT currently requires a non-negative integer literal");
    }
    if (limit->offset) {
        if (limit->offset->type != hsql::kExprLiteralInt || limit->offset->ival != 0) {
            throw std::runtime_error("LIMIT OFFSET is not supported by the GPU ORDER BY/LIMIT path");
        }
    }
    return static_cast<std::size_t>(limit->limit->ival);
}

} // namespace

std::unique_ptr<ExprNode> QueryTranslator::translateHavingCaseExpr(
        const hsql::Expr* expr,
        const hsql::SelectStatement* stmt) {
    if (!expr || expr->type != hsql::kExprOperator || expr->opType != hsql::kOpCase) {
        throw std::runtime_error("QueryTranslator::translateHavingCaseExpr: expected Hyrise kOpCase expression");
    }

    std::vector<HsqlCaseArm> arms;
    if (expr->exprList) {
        for (const hsql::Expr* item : *expr->exprList) collectCaseArmsFromNode(item, arms);
    }

    const hsql::Expr* base_expr = nullptr;
    if (expr->expr) {
        if (nodeLooksLikeCaseList(expr->expr)) {
            collectCaseArmsFromNode(expr->expr, arms);
        } else {
            base_expr = expr->expr;
        }
    }

    if (arms.empty()) {
        throw std::runtime_error("QueryTranslator: HAVING CASE expression contains no WHEN arms");
    }

    std::unique_ptr<ExprNode> else_expr = expr->expr2
        ? translateHavingExpr(expr->expr2, stmt)
        : std::make_unique<LiteralNullExpr>();

    for (auto it = arms.rbegin(); it != arms.rend(); ++it) {
        std::unique_ptr<ExprNode> condition;
        if (base_expr) {
            condition = std::make_unique<BinaryExpr>(
                ExprType::OP_EQ,
                translateHavingExpr(base_expr, stmt),
                translateHavingExpr(it->when_expr, stmt));
        } else {
            condition = translateHavingExpr(it->when_expr, stmt);
        }
        else_expr = std::make_unique<CaseWhenExpr>(
            std::move(condition),
            translateHavingExpr(it->then_expr, stmt),
            std::move(else_expr));
    }

    return else_expr;
}

std::unique_ptr<ExprNode> QueryTranslator::translateHavingExpr(
        const hsql::Expr* expr,
        const hsql::SelectStatement* stmt) {
    if (!expr) {
        throw std::runtime_error("QueryTranslator::translateHavingExpr: null expression");
    }

    auto result_slot_expr = [](std::size_t slot) {
        return std::make_unique<ColumnRefExpr>("__result_col_" + std::to_string(slot), "__RESULT__");
    };

    switch (expr->type) {
        case hsql::kExprLiteralInt:
            return std::make_unique<LiteralIntExpr>(expr->ival);
        case hsql::kExprLiteralFloat:
            return std::make_unique<LiteralFloatExpr>(expr->fval);
        case hsql::kExprLiteralNull:
            return std::make_unique<LiteralNullExpr>();
        case hsql::kExprColumnRef: {
            std::size_t alias_select_index = 0;
            if (resolveSelectAliasReference(expr, select_alias_to_select_index_, alias_select_index)) {
                return result_slot_expr(aggregateOutputSlotForSelectIndex(stmt, alias_select_index));
            }
            if (stmt->groupBy && stmt->groupBy->columns) {
                const std::string col = expr->name ? std::string(expr->name) : "";
                const std::string table = resolveTableQualifier(expr->table ? std::string(expr->table) : "");
                for (std::size_t i = 0; i < stmt->groupBy->columns->size(); ++i) {
                    const hsql::Expr* group_expr = (*stmt->groupBy->columns)[i];
                    if (!group_expr || group_expr->type != hsql::kExprColumnRef) continue;
                    const std::string group_col = group_expr->name ? std::string(group_expr->name) : "";
                    const std::string group_table = resolveTableQualifier(group_expr->table ? std::string(group_expr->table) : "");
                    if (upperCopy(group_col) == upperCopy(col) &&
                        (table.empty() || group_table.empty() || table == group_table)) {
                        return result_slot_expr(i);
                    }
                }
            }
            throw std::runtime_error(
                "HAVING column reference must be a GROUP BY expression or SELECT alias");
        }
        case hsql::kExprFunctionRef: {
            std::string func = expr->name ? std::string(expr->name) : "";
            func = translatorUpperAscii(func);
            if (func == "CASE_WHEN") {
                if (!expr->exprList || expr->exprList->size() != 3) {
                    throw std::runtime_error("HAVING CASE_WHEN requires exactly 3 arguments");
                }
                return std::make_unique<CaseWhenExpr>(
                    translateHavingExpr((*expr->exprList)[0], stmt),
                    translateHavingExpr((*expr->exprList)[1], stmt),
                    translateHavingExpr((*expr->exprList)[2], stmt));
            }
            if (!stmt->selectList) {
                throw std::runtime_error("HAVING aggregate requires a SELECT list aggregate");
            }
            for (std::size_t i = 0; i < stmt->selectList->size(); ++i) {
                const hsql::Expr* sel = (*stmt->selectList)[i];
                if (sel && sel->isType(hsql::kExprFunctionRef) && hsqlExprEquivalent(expr, sel)) {
                    return result_slot_expr(aggregateOutputSlotForSelectIndex(stmt, i));
                }
            }
            throw std::runtime_error(
                "HAVING aggregate expression must also appear in SELECT list");
        }
        case hsql::kExprOperator: {
            if (expr->opType == hsql::kOpCase) {
                return translateHavingCaseExpr(expr, stmt);
            }
            if (expr->opType == hsql::kOpIsNull) {
                if (!expr->expr) throw std::runtime_error("QueryTranslator: HAVING IS NULL missing operand");
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_IS_NULL, translateHavingExpr(expr->expr, stmt), nullptr);
            }
            if (expr->opType == hsql::kOpNot && expr->expr &&
                expr->expr->type == hsql::kExprOperator &&
                expr->expr->opType == hsql::kOpIsNull) {
                if (!expr->expr->expr) throw std::runtime_error("QueryTranslator: HAVING IS NOT NULL missing operand");
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_IS_NOT_NULL, translateHavingExpr(expr->expr->expr, stmt), nullptr);
            }
            if (expr->opType == hsql::kOpNot) {
                if (!expr->expr) throw std::runtime_error("QueryTranslator: HAVING NOT missing operand");
                return std::make_unique<BinaryExpr>(
                    ExprType::OP_NOT, translateHavingExpr(expr->expr, stmt), nullptr);
            }
            ExprType op = mapBinaryOp(expr->opType);
            if (!expr->expr || !expr->expr2) {
                throw std::runtime_error("QueryTranslator: malformed binary expression in HAVING");
            }
            return std::make_unique<BinaryExpr>(
                op, translateHavingExpr(expr->expr, stmt), translateHavingExpr(expr->expr2, stmt));
        }
        default:
            throw std::runtime_error(
                "QueryTranslator::translateHavingExpr: unsupported expression type: " +
                std::to_string(static_cast<int>(expr->type)));
    }
}

// Главная точка входа: SELECT → операторное дерево
std::unique_ptr<OperatorNode> QueryTranslator::translate(const hsql::SelectStatement* stmt) {
    if (!stmt) {
        throw std::runtime_error("QueryTranslator::translate: null SelectStatement");
    }

    table_alias_to_name_.clear();
    select_alias_to_select_index_.clear();

    // ---- 1. Трансляция FROM → поддерево scan/join ----
    std::unique_ptr<OperatorNode> root = translateFrom(stmt->fromTable);
    registerSelectAliases(stmt);

    // ---- 2. WHERE → FilterNode поверх поддерева FROM ----
    if (stmt->whereClause) {
        auto filter = std::make_unique<FilterNode>(translateExpr(stmt->whereClause));
        filter->addChild(std::move(root));
        root = std::move(filter);
    }

    // ---- 3. Анализируем SELECT LIST на предмет агрегаций ----
    bool has_group_by = (stmt->groupBy != nullptr &&
                         stmt->groupBy->columns != nullptr &&
                         !stmt->groupBy->columns->empty());

    bool has_aggregation = false;
    if (stmt->selectList) {
        for (const auto* sel : *stmt->selectList) {
            if (sel && sel->isType(hsql::kExprFunctionRef) &&
                isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) {
                has_aggregation = true;
                break;
            }
        }
    }

    if (has_group_by || has_aggregation) {
        auto agg_node = std::make_unique<AggregateNode>();

        // GROUP BY выражения
        if (has_group_by) {
            for (const auto* g : *stmt->groupBy->columns) {
                agg_node->group_by_exprs.push_back(translateExpr(g));
            }
        }

        // Агрегатные функции из SELECT LIST
        if (stmt->selectList) {
            for (const auto* sel : *stmt->selectList) {
                if (!sel || !sel->isType(hsql::kExprFunctionRef) ||
                    !isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) continue;

                std::string func = sel->name ? std::string(sel->name) : "SUM";
                std::transform(func.begin(), func.end(), func.begin(), ::toupper);
                if (func != "COUNT" && func != "SUM" && func != "MIN" &&
                    func != "MAX" && func != "AVG") {
                    throw std::runtime_error("QueryTranslator: unsupported aggregate function: " + func);
                }

                std::unique_ptr<ExprNode> agg_expr;
                if (sel->exprList && !sel->exprList->empty()) {
                    if (sel->exprList->size() > 1) {
                        throw std::runtime_error("QueryTranslator: aggregate functions accept at most one argument");
                    }
                    const hsql::Expr* arg = (*sel->exprList)[0];
                    if (arg) agg_expr = translateExpr(arg);
                }

                if (!agg_expr) {
                    throw std::runtime_error(
                        "QueryTranslator: aggregate function " + func +
                        " requires exactly one SQL argument; use COUNT(*) for row count");
                }
                if (agg_expr->getType() == ExprType::STAR && func != "COUNT") {
                    throw std::runtime_error(
                        "QueryTranslator: aggregate function " + func +
                        " does not accept '*' according to SQL semantics");
                }

                agg_node->aggregates.emplace_back(std::move(func), std::move(agg_expr));
            }
        }

        agg_node->output_aliases.resize(static_cast<std::size_t>(agg_node->visibleTupleSize()));
        if (stmt->selectList) {
            for (std::size_t i = 0; i < stmt->selectList->size(); ++i) {
                const hsql::Expr* sel = (*stmt->selectList)[i];
                if (!sel || !sel->alias) continue;
                const std::size_t slot = aggregateOutputSlotForSelectIndex(stmt, i);
                if (slot < agg_node->output_aliases.size()) {
                    agg_node->output_aliases[slot] = std::string(sel->alias);
                }
            }
        }
        if (stmt->groupBy && stmt->groupBy->having) {
            agg_node->having_predicate = translateHavingExpr(stmt->groupBy->having, stmt);
        }

        agg_node->addChild(std::move(root));
        root = std::move(agg_node);
    } else if (stmt->selectList) {
        auto projection = std::make_unique<ProjectionNode>();
        for (const auto* sel : *stmt->selectList) {
            if (!sel) continue;
            if (sel->isType(hsql::kExprFunctionRef) &&
                isAggregateFunctionName(sel->name ? std::string(sel->name) : std::string())) {
                throw std::runtime_error("QueryTranslator: aggregate function in non-aggregate SELECT list is not supported");
            }
            projection->select_exprs.push_back(translateExpr(sel));
            projection->output_aliases.push_back(sel->alias ? std::string(sel->alias) : std::string());
        }
        if (!projection->select_exprs.empty()) {
            projection->addChild(std::move(root));
            root = std::move(projection);
        }
    }

    const bool has_order_by = stmt->order && !stmt->order->empty();
    const bool has_limit = stmt->limit && stmt->limit->limit;
    if (has_order_by || has_limit) {
        auto sort_limit = std::make_unique<SortLimitNode>();
        if (has_limit) {
            sort_limit->has_limit = true;
            sort_limit->limit = parseLimitValue(stmt->limit);
        }
        if (has_order_by) {
            for (const auto* ord : *stmt->order) {
                if (!ord || !ord->expr) {
                    throw std::runtime_error("ORDER BY contains an empty expression");
                }
                SortKeyDef key;
                key.column_index = (has_group_by || has_aggregation)
                    ? resolveAggregateOrderKey(ord->expr, stmt, select_alias_to_select_index_)
                    : resolveProjectionOrderKey(ord->expr, stmt, select_alias_to_select_index_);
                key.descending = (ord->type == hsql::kOrderDesc);
                sort_limit->sort_keys.push_back(key);
            }
        }
        sort_limit->addChild(std::move(root));
        root = std::move(sort_limit);
    }

    return root;
}

} // namespace db
