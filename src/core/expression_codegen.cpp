#include "core/expression_codegen.h"
#include "core/size_expression.h"
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace db {
namespace {
bool truth(const std::string& text) { return text == "1" || text == "true"; }
bool falsity(const std::string& text) { return text == "0" || text == "false"; }
std::string conjunction(const std::string& a, const std::string& b) {
    if (falsity(a) || falsity(b)) return "0";
    if (truth(a)) return b;
    if (truth(b)) return a;
    return "(" + a + " && " + b + ")";
}
std::string disjunction(const std::string& a, const std::string& b) {
    if (truth(a) || truth(b)) return "1";
    if (falsity(a)) return b;
    if (falsity(b)) return a;
    return "(" + a + " || " + b + ")";
}
std::string negate(const std::string& a) {
    if (truth(a)) return "0";
    if (falsity(a)) return "1";
    return "(!(" + a + "))";
}
const char* arithmeticHelper(ExprType type) {
    if (type == ExprType::OP_ADD) return "safe_add";
    if (type == ExprType::OP_SUB) return "safe_sub";
    if (type == ExprType::OP_MUL) return "safe_mul";
    return "safe_div";
}
const char* comparisonHelper(ExprType type) {
    static const char* names[] = {"safe_eq", "safe_neq", "safe_lt", "safe_lte", "safe_gt", "safe_gte"};
    return names[static_cast<size_t>(type) - static_cast<size_t>(ExprType::OP_EQ)];
}
bool arithmetic(ExprType type) { return type >= ExprType::OP_ADD && type <= ExprType::OP_DIV; }
bool comparison(ExprType type) { return type >= ExprType::OP_EQ && type <= ExprType::OP_GTE; }
std::string call(const char* helper, const ExpressionCode& a, const ExpressionCode& b) {
    return "db::" + std::string(helper) + "(" + a.value + ", " + b.value + ")";
}
ExpressionCode binaryCode(const BinaryExpr& expr, const ColumnExpression& column, const ParameterSchema& schema) {
    const auto a = emitSqlExpression(expr.left.get(), column, schema);
    if (expr.op_type == ExprType::OP_IS_NULL) return {negate(a.valid), "1"};
    if (expr.op_type == ExprType::OP_IS_NOT_NULL) return {a.valid, "1"};
    if (expr.op_type == ExprType::OP_NOT) return {negate(a.value), a.valid};
    const auto b = emitSqlExpression(expr.right.get(), column, schema);
    const auto valid = conjunction(a.valid, b.valid);
    if (arithmetic(expr.op_type)) return {call(arithmeticHelper(expr.op_type), a, b), valid};
    if (comparison(expr.op_type)) return {call(comparisonHelper(expr.op_type), a, b), valid};
    if (expr.op_type == ExprType::OP_AND) {
        const auto known_false = disjunction(conjunction(a.valid, negate(a.value)), conjunction(b.valid, negate(b.value)));
        return {conjunction(a.value, b.value), disjunction(known_false, valid)};
    }
    if (expr.op_type == ExprType::OP_OR) {
        const auto known_true = disjunction(conjunction(a.valid, a.value), conjunction(b.valid, b.value));
        return {disjunction(a.value, b.value), disjunction(known_true, valid)};
    }
    throw std::runtime_error("Unsupported SQL expression operator");
}
ExpressionCode caseCode(const CaseWhenExpr& expr, const ColumnExpression& column, const ParameterSchema& schema) {
    const auto condition = emitSqlExpression(expr.condition.get(), column, schema);
    const auto a = emitSqlExpression(expr.then_expr.get(), column, schema);
    const auto b = emitSqlExpression(expr.else_expr.get(), column, schema);
    const auto selected = conjunction(condition.valid, condition.value);
    return {emitCaseValue(&expr, selected, a.value, b.value, schema),
            disjunction(conjunction(selected, a.valid), conjunction(negate(selected), b.valid))};
}
std::string floatLiteral(double value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << std::scientific << value;
    return out.str();
}
ParameterType commonNumeric(ParameterType a, ParameterType b) {
    return commonParameterType(a, b);
}
void caseLeaves(const ExprNode* expr, std::vector<const ExprNode*>& leaves) {
    if (!expr) throw std::runtime_error("Absent GROUP BY CASE leaf");
    if (expr->getType() == ExprType::CASE_WHEN) {
        const auto* branch = static_cast<const CaseWhenExpr*>(expr);
        caseLeaves(branch->then_expr.get(), leaves);
        caseLeaves(branch->else_expr.get(), leaves);
    } else {
        const auto type = expr->getType();
        if (type != ExprType::PARAMETER_REF && type != ExprType::LITERAL_INT && type != ExprType::LITERAL_NULL)
            throw std::runtime_error("GROUP BY CASE requires integer/NULL leaf parameters");
        leaves.push_back(expr);
    }
}
std::optional<ExpressionCode> scalarCode(const ExprNode* expr,const ColumnExpression& column) {
    const auto type = expr->getType();
    if (type == ExprType::PARAMETER_REF) {
        const auto slot = static_cast<const ParameterRefExpr*>(expr)->slot;
        return ExpressionCode{parameterName(slot), parameterName(slot) + "_valid"};
    }
    if (type == ExprType::COLUMN_REF) return column(static_cast<const ColumnRefExpr&>(*expr));
    if (type == ExprType::LITERAL_INT) {
        const auto* value = static_cast<const LiteralIntExpr*>(expr);
        const auto text = value->value == INT64_MIN ? "(-9223372036854775807LL - 1LL)" : std::to_string(value->value) + "LL";
        return ExpressionCode{text, "1"};
    }
    if (type == ExprType::LITERAL_FLOAT) return ExpressionCode{floatLiteral(static_cast<const LiteralFloatExpr*>(expr)->value), "1"};
    if (type == ExprType::LITERAL_NULL) return ExpressionCode{"0LL", "0"};
    if (type == ExprType::STAR) return ExpressionCode{"1", "1"};
    return {};
}
void validateCaseTypes(const std::vector<const ExprNode*>& leaves,const ParameterSchema& schema) {
    for (const auto* leaf:leaves) {
        if (leaf->getType()!=ExprType::PARAMETER_REF || schema.empty()) continue;
        if (schema.at(static_cast<const ParameterRefExpr*>(leaf)->slot)==ParameterType::Float64)
            throw std::runtime_error("GROUP BY CASE requires integer/NULL leaf parameters");
    }
}
} // namespace

std::string parameterName(size_t slot) { return "db_parameter_" + std::to_string(slot); }

ExpressionCode emitSqlExpression(const ExprNode* expr, const ColumnExpression& column, const ParameterSchema& schema) {
    if (!expr) return {"0", "0"};
    if (const auto scalar=scalarCode(expr,column)) return *scalar;
    const auto type=expr->getType();
    if (type == ExprType::CASE_WHEN) return caseCode(static_cast<const CaseWhenExpr&>(*expr), column, schema);
    if (type >= ExprType::OP_AND && type <= ExprType::OP_IS_NOT_NULL) return binaryCode(static_cast<const BinaryExpr&>(*expr), column, schema);
    throw std::runtime_error("Unsupported SQL expression node");
}

ParameterType sqlExpressionType(const ExprNode* expr, const ParameterSchema& schema, const ColumnType& column_type) {
    if (!expr || expr->getType() == ExprType::LITERAL_NULL) return ParameterType::NullOnly;
    if (expr->getType() == ExprType::LITERAL_FLOAT) return ParameterType::Float64;
    if (expr->getType() == ExprType::LITERAL_TEXT) return ParameterType::Text;
    if (expr->getType() == ExprType::PARAMETER_REF) return schema.at(static_cast<const ParameterRefExpr*>(expr)->slot);
    if (expr->getType() == ExprType::COLUMN_REF && column_type) return column_type(static_cast<const ColumnRefExpr&>(*expr));
    if (expr->getType() == ExprType::COLUMN_REF) return static_cast<const ColumnRefExpr&>(*expr).value_type;
    if (expr->getType() == ExprType::CASE_WHEN) {
        const auto& branch = static_cast<const CaseWhenExpr&>(*expr);
        return commonNumeric(sqlExpressionType(branch.then_expr.get(), schema, column_type), sqlExpressionType(branch.else_expr.get(), schema, column_type));
    }
    if (arithmetic(expr->getType())) {
        const auto& binary = static_cast<const BinaryExpr&>(*expr);
        return commonNumeric(sqlExpressionType(binary.left.get(), schema, column_type), sqlExpressionType(binary.right.get(), schema, column_type));
    }
    return ParameterType::Int64;
}

std::string emitCaseValue(const ExprNode* expr, const std::string& condition, const std::string& a,
                          const std::string& b, const ParameterSchema& schema, const ColumnType& column_type) {
    const auto value = "((" + condition + ") ? (" + a + ") : (" + b + "))";
    if (schema.empty()) return value; // literal-only legacy callers already have C++ common-type promotion
    const auto type = sqlExpressionType(expr, schema, column_type);
    if (type == ParameterType::Text) return value;
    return "static_cast<" + std::string(type == ParameterType::Float64 ? "double" : "std::int64_t") + ">(" + value + ")";
}

uint64_t caseLeafCount(const ExprNode* expr) {
    std::vector<const ExprNode*> leaves;
    caseLeaves(expr, leaves);
    return leaves.size();
}

std::string emitCaseGroupCode(const ExprNode* expr, const ColumnExpression& column,const ParameterSchema& schema) {
    std::vector<const ExprNode*> leaves;
    caseLeaves(expr, leaves);
    validateCaseTypes(leaves,schema);
    const auto selected = emitSqlExpression(expr, column, schema);
    std::string code = "0ULL";
    // Earlier equal leaves win. Branch IDs alone would incorrectly split equal SQL groups.
    for (size_t i = leaves.size(); i > 0; --i) {
        const auto leaf = emitSqlExpression(leaves[i - 1], column);
        const auto equal_null = conjunction(negate(selected.valid), negate(leaf.valid));
        const auto equal_value = conjunction(conjunction(selected.valid, leaf.valid), call("safe_eq", selected, leaf));
        code = "((" + disjunction(equal_null, equal_value) + ") ? " + std::to_string(i - 1) + "ULL : (" + code + "))";
    }
    return code;
}
} // namespace db
