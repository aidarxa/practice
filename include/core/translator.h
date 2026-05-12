#pragma once

#include "operators.h"
#include "expressions.h"
#include "../../deps/include/SQLParser.h"
#include "sql/SelectStatement.h"
#include "sql/Table.h"
#include "sql/Expr.h"

#include <memory>
#include <stdexcept>

namespace db {

// ============================================================================
// QueryTranslator
//
// Транслирует HSQL AST (hsql::SelectStatement*) в наивное дерево операторов.
// Это "первая фаза" — дерево неоптимизировано:
//   - HashJoinNode без условий (join_condition == nullptr)
//   - FilterNode содержит всё WHERE-дерево целиком (один большой And)
//   - Оптимизатор позже выполнит predicate pushdown и связку join-условий
//
// Не владеет AST-указателями: все они живут в hsql::SQLParserResult.
// ============================================================================
class QueryTranslator {
public:
    QueryTranslator() = default;

    // Главная точка входа. Возвращает корень операторного дерева.
    // Бросает std::runtime_error при неподдерживаемых конструкциях.
    std::unique_ptr<OperatorNode> translate(const hsql::SelectStatement* stmt);

private:
    // -----------------------------------------------------------------------
    // Трансляция FROM — строит поддерево scan/join
    // -----------------------------------------------------------------------

    // Рекурсивно транслирует hsql::TableRef в узел оператора.
    std::unique_ptr<OperatorNode> translateFrom(const hsql::TableRef* ref);

    // -----------------------------------------------------------------------
    // Трансляция выражений — строит поддерево ExprNode
    // -----------------------------------------------------------------------

    // Рекурсивно транслирует hsql::Expr* в дерево ExprNode.
    // Бросает std::runtime_error для неподдерживаемых типов выражений.
    std::unique_ptr<ExprNode> translateExpr(const hsql::Expr* expr);

    // Отображение HSQL OperatorType → ExprType
    // Бросает std::runtime_error для неподдерживаемых операторов.
    static ExprType mapBinaryOp(hsql::OperatorType op);
};

} // namespace db
