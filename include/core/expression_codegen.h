#pragma once
/** @file @brief Pure SQL expression lowering; values and validity travel together. */
#include "core/expressions.h"
#include "core/parameter_value.h"
#include <functional>

namespace db {
/** @brief Device expressions, not evaluated values; invalid payloads are never SQL truth. */
struct ExpressionCode { std::string value; std::string valid; };
using ColumnExpression = std::function<ExpressionCode(const ColumnRefExpr&)>;
using ColumnType = std::function<ParameterType(const ColumnRefExpr&)>;

/** @brief Emit one expression using caller-provided column bindings and independent ABI slots. */
ExpressionCode emitSqlExpression(const ExprNode* expression, const ColumnExpression& column, const ParameterSchema& schema = {});
/** @brief Infer numeric output type from structure/schema only; no value specialization. */
ParameterType sqlExpressionType(const ExprNode* expression, const ParameterSchema& schema, const ColumnType& column_type = {});
/** @brief Apply the schema's common CASE type even when a tagged branch currently holds an integer.
 * @details SQL type follows both branches, not the branch selected by this row. An
 * integer branch of a Float64 CASE must not cause integer division in fallback code.
 */
std::string emitCaseValue(const ExprNode* expression, const std::string& condition,
                          const std::string& then_value, const std::string& else_value,
                          const ParameterSchema& schema, const ColumnType& column_type = {});
/** @brief Stable symbolic parameter name; the module generator normalizes global slot numbers. */
std::string parameterName(size_t slot);
/** @brief Determine a fixed CASE domain from leaf occurrences, not their current integer range. */
uint64_t caseLeafCount(const ExprNode* expression);
/** @brief Encode equal CASE outputs into the first matching leaf bucket, including NULL. */
std::string emitCaseGroupCode(const ExprNode* expression, const ColumnExpression& column,
                             const ParameterSchema& schema = {});
} // namespace db
