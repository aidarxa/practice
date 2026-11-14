#pragma once

#include "operators.h"
#include "expressions.h"
#include "query_template.h"
#include "../../deps/include/SQLParser.h"
#include "sql/SelectStatement.h"
#include "sql/Table.h"
#include "sql/Expr.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

    /**
     * @brief Resolve SQL references against a catalog, then extract value parameters.
     * @details The returned plan is not optimized. Each call owns its resolver
     * state and values; AST pointers are used only as temporary occurrence IDs.
     * This is the frontend entry point for the migration; translate() remains a
     * literal-plan entry point for legacy tests and tools without a catalog.
     */
    PreparedQuery prepare(const hsql::SelectStatement* stmt, const Catalog& catalog) const;

private:
    /** @brief A projection star includes text too; COUNT(*) deliberately does not call this check. */
    void validateProjectionStorage(const ExprNode& expression) const;
    // -----------------------------------------------------------------------
    // Трансляция FROM — строит поддерево scan/join
    // -----------------------------------------------------------------------

    // Рекурсивно транслирует hsql::TableRef в узел оператора.
    std::unique_ptr<OperatorNode> translateFrom(const hsql::TableRef* ref);

    // Table aliases are resolved in the translator so optimizer/JIT receive
    // canonical catalog table names in ColumnRefExpr::table_name.
    std::unordered_map<std::string, std::string> table_alias_to_name_;
    std::unordered_map<std::string, std::size_t> select_alias_to_select_index_;
    const Catalog* catalog_ = nullptr;
    std::vector<std::string> from_tables_;
    std::unordered_map<const hsql::Expr*, std::size_t> literal_origins_;

    std::string resolveTableQualifier(const std::string& qualifier) const;
    /** @brief Bind a column once, rejecting unknown or ambiguous catalog references. */
    ColumnRefExpr resolveColumnReference(const std::string& column, const std::string& qualifier) const;
    /** @brief Resolve one validated qualifier in the FROM scope, rejecting ambiguity. */
    ColumnRefExpr resolveCatalogColumn(const std::string& column, const std::string& table) const;
    std::size_t literalOrigin(const hsql::Expr* expr);
    /** @brief Normalize a signed numeric literal before parameter extraction; not general unary arithmetic. */
    std::unique_ptr<ExprNode> translateNegativeLiteral(const hsql::Expr* expr);
    void registerSelectAliases(const hsql::SelectStatement* stmt);
    /** @brief Compare resolved references before literal values become parameters. */
    bool hsqlExprEquivalent(const hsql::Expr* left, const hsql::Expr* right) const;
    std::size_t aggregateOutputSlotForSelectIndex(
        const hsql::SelectStatement* stmt, std::size_t select_index) const;
    std::size_t resolveProjectionOrderKey(
        const hsql::Expr* expr, const hsql::SelectStatement* stmt) const;
    /** @brief Expanded SELECT offsets resolve ORDER BY positions before parameter extraction. */
    std::vector<std::size_t> projectionSelectOffsets(const hsql::SelectStatement* stmt) const;
    std::size_t resolveAggregateOrderKey(
        const hsql::Expr* expr, const hsql::SelectStatement* stmt) const;
    std::unique_ptr<ExprNode> translateHavingExpr(const hsql::Expr* expr, const hsql::SelectStatement* stmt);
    std::unique_ptr<ExprNode> translateCaseExpr(const hsql::Expr* expr);
    std::unique_ptr<ExprNode> translateHavingCaseExpr(const hsql::Expr* expr, const hsql::SelectStatement* stmt);

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
