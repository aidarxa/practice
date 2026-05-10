#include "core/translator.h"

#include <algorithm>
#include <ostream>
#include <sstream>
#include <iostream>

// ============================================================================
// Block 1: ExprPrinter implementation (definitions живут здесь, а не в .h)
// ============================================================================

namespace db {

// --- accept() definitions ---

void ColumnRefExpr::accept(ExprVisitor& visitor) const  { visitor.visit(*this); }
void LiteralIntExpr::accept(ExprVisitor& visitor) const { visitor.visit(*this); }
void LiteralFloatExpr::accept(ExprVisitor& visitor) const { visitor.visit(*this); }
void BinaryExpr::accept(ExprVisitor& visitor) const     { visitor.visit(*this); }

// --- ExprPrinter ---

void ExprPrinter::printIndent() const {
    for (int i = 0; i < indent_; ++i) out_ << "  ";
}

const char* ExprPrinter::exprTypeName(ExprType t) {
    switch (t) {
        case ExprType::COLUMN_REF:    return "COLUMN_REF";
        case ExprType::LITERAL_INT:   return "LITERAL_INT";
        case ExprType::LITERAL_FLOAT: return "LITERAL_FLOAT";
        case ExprType::OP_AND:        return "AND";
        case ExprType::OP_OR:         return "OR";
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

// ============================================================================
// Block 2: OperatorNode accept() definitions + OperatorPrinter
// ============================================================================

void TableScanNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }
void FilterNode::accept(OperatorVisitor& visitor) const    { visitor.visit(*this); }
void HashJoinNode::accept(OperatorVisitor& visitor) const  { visitor.visit(*this); }
void AggregateNode::accept(OperatorVisitor& visitor) const { visitor.visit(*this); }

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
    out_ << "[TableScan] " << node.table_name << "\n";
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

// Рекурсивная трансляция hsql::Expr* → ExprNode
std::unique_ptr<ExprNode> QueryTranslator::translateExpr(const hsql::Expr* expr) {
    if (!expr) {
        throw std::runtime_error("QueryTranslator::translateExpr: null expression");
    }

    switch (expr->type) {
        // ---- Ссылка на колонку ----
        case hsql::kExprColumnRef: {
            std::string col  = expr->name  ? std::string(expr->name)  : "";
            std::string tbl  = expr->table ? std::string(expr->table) : "";
            return std::make_unique<ColumnRefExpr>(std::move(col), std::move(tbl));
        }

        // ---- Целочисленный литерал ----
        case hsql::kExprLiteralInt:
            return std::make_unique<LiteralIntExpr>(expr->ival);

        // ---- Вещественный литерал ----
        case hsql::kExprLiteralFloat:
            return std::make_unique<LiteralFloatExpr>(expr->fval);

        // ---- Бинарные/логические операторы ----
        case hsql::kExprOperator: {
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

        // ---- Агрегатные функции в WHERE — неожиданно ----
        case hsql::kExprFunctionRef:
            throw std::runtime_error(
                "QueryTranslator: function references in WHERE/predicates are not supported");

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
            return std::make_unique<TableScanNode>(std::move(name));
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
            std::unique_ptr<ExprNode> cond = nullptr;
            if (ref->join->condition) {
                cond = translateExpr(ref->join->condition);
            }
            auto join = std::make_unique<HashJoinNode>(std::move(cond));
            join->addChild(translateFrom(ref->join->left));
            join->addChild(translateFrom(ref->join->right));
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

// Главная точка входа: SELECT → операторное дерево
std::unique_ptr<OperatorNode> QueryTranslator::translate(const hsql::SelectStatement* stmt) {
    if (!stmt) {
        throw std::runtime_error("QueryTranslator::translate: null SelectStatement");
    }

    // ---- 1. Трансляция FROM → поддерево scan/join ----
    std::unique_ptr<OperatorNode> root = translateFrom(stmt->fromTable);

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
            if (sel && sel->isType(hsql::kExprFunctionRef)) {
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
                if (!sel || !sel->isType(hsql::kExprFunctionRef)) continue;

                std::string func = sel->name ? std::string(sel->name) : "SUM";
                std::transform(func.begin(), func.end(), func.begin(), ::toupper);
                // Аргумент агрегатной функции
                std::unique_ptr<ExprNode> agg_expr;
                if (sel->exprList && !sel->exprList->empty()) {
                    const hsql::Expr* arg = (*sel->exprList)[0];
                    if (arg) {
                        agg_expr = translateExpr(arg);
                    }
                }

                agg_node->aggregates.emplace_back(std::move(func), std::move(agg_expr));
            }
        }

        agg_node->addChild(std::move(root));
        root = std::move(agg_node);
    }

    return root;
}

} // namespace db
