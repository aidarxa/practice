#include "core/query_template.h"

#include <unordered_map>

namespace db {

ParameterValues bindParameterValues(const ParameterSchema& schema, const ParameterValues& values) {
    if (schema.size() != values.size()) throw std::runtime_error("Parameter count does not match schema");
    ParameterValues result;
    result.reserve(values.size());
    for (std::size_t i = 0; i < schema.size(); ++i) {
        const auto& value = values[i];
        if (value.type() == ParameterType::NullOnly) {
            result.push_back(ParameterValue::null(schema[i]));
        } else if (value.type() == schema[i]) {
            result.push_back(value);
        } else {
            throw std::runtime_error("Parameter type mismatch at slot " + std::to_string(i) + "; prepare another schema");
        }
    }
    return result;
}

namespace {

ParameterType commonType(ParameterType left, ParameterType right) {
    return commonParameterType(left, right);
}

bool isArithmetic(ExprType type) {
    return type == ExprType::OP_ADD || type == ExprType::OP_SUB ||
           type == ExprType::OP_MUL || type == ExprType::OP_DIV;
}

using ResultTypes = std::unordered_map<std::string, ParameterType>;

/** @brief Infer logical types only; never inspect the value or validity of a parameter. */
ParameterType expressionType(const ExprNode* expr, const ParameterSchema& schema, const ResultTypes& results) {
    if (!expr) return ParameterType::NullOnly;
    switch (expr->getType()) {
        case ExprType::LITERAL_NULL: return ParameterType::NullOnly;
        case ExprType::LITERAL_FLOAT: return ParameterType::Float64;
        case ExprType::LITERAL_TEXT: return ParameterType::Text;
        case ExprType::PARAMETER_REF:
            return schema.at(static_cast<const ParameterRefExpr&>(*expr).slot);
        case ExprType::COLUMN_REF: {
            const auto& col = static_cast<const ColumnRefExpr&>(*expr);
            if (col.table_name == "__RESULT__") return results.at(col.column_name);
            return col.value_type;
        }
        case ExprType::CASE_WHEN: {
            const auto& branch = static_cast<const CaseWhenExpr&>(*expr);
            return commonType(expressionType(branch.then_expr.get(), schema, results),
                              expressionType(branch.else_expr.get(), schema, results));
        }
        default: break;
    }
    if (isArithmetic(expr->getType())) {
        const auto& binary = static_cast<const BinaryExpr&>(*expr);
        return commonType(expressionType(binary.left.get(), schema, results),
                          expressionType(binary.right.get(), schema, results));
    }
    return ParameterType::Int64;
}

ParameterValue literalValue(const ExprNode& expr, ParameterType null_type) {
    switch (expr.getType()) {
        case ExprType::LITERAL_INT: return ParameterValue::integer(static_cast<const LiteralIntExpr&>(expr).value);
        case ExprType::LITERAL_FLOAT: return ParameterValue::floating(static_cast<const LiteralFloatExpr&>(expr).value);
        case ExprType::LITERAL_TEXT: return ParameterValue::text(static_cast<const LiteralTextExpr&>(expr).value);
        case ExprType::LITERAL_NULL: return ParameterValue::null(null_type);
        default: throw std::runtime_error("Expected literal during parameter extraction");
    }
}

/** @brief Request-owned extraction state; identities never depend on equal values. */
class ParameterExtractor {
public:
    PreparedQuery run(std::unique_ptr<OperatorNode> plan) {
        if (!plan) throw std::runtime_error("Cannot parameterize an absent plan");
        extractPlan(*plan);
        return {{std::move(plan), std::move(schema_)}, std::move(values_)};
    }

private:
    ParameterSchema schema_;
    ParameterValues values_;
    std::unordered_map<std::size_t, std::size_t> origin_slots_;

    void extractLiteral(std::unique_ptr<ExprNode>& expr, ParameterType hint) {
        const auto& literal = static_cast<const LiteralExprNode&>(*expr);
        auto value = literalValue(*expr, hint);
        std::size_t slot = values_.size();
        if (literal.origin) {
            const auto inserted = origin_slots_.emplace(*literal.origin, slot);
            slot = inserted.first->second;
        }
        if (slot == values_.size()) {
            values_.push_back(value);
            schema_.push_back(value.type());
        } else if (value.isNull()) {
            schema_[slot] = commonType(schema_[slot], value.type());
            values_[slot] = ParameterValue::null(schema_[slot]);
        }
        expr = std::make_unique<ParameterRefExpr>(slot);
    }

    void extractExpr(std::unique_ptr<ExprNode>& expr, ParameterType hint, const ResultTypes& results) {
        if (!expr) return;
        if (dynamic_cast<const LiteralExprNode*>(expr.get())) {
            extractLiteral(expr, hint);
        } else if (auto* branch = dynamic_cast<CaseWhenExpr*>(expr.get())) {
            requireBoolean(branch->condition.get(), results);
            auto output_type = expressionType(expr.get(), schema_, results);
            if (output_type == ParameterType::NullOnly) output_type = hint;
            extractExpr(branch->condition, ParameterType::Int64, results);
            extractExpr(branch->then_expr, output_type, results);
            extractExpr(branch->else_expr, output_type, results);
        } else if (auto* binary = dynamic_cast<BinaryExpr*>(expr.get())) {
            extractBinary(*binary, hint, results);
        } else if (expr->getType() == ExprType::COLUMN_REF) {
            auto& column = static_cast<ColumnRefExpr&>(*expr);
            if (column.table_name == "__RESULT__") column.value_type = results.at(column.column_name);
        } else if (expr->getType() == ExprType::PARAMETER_REF) {
            throw std::runtime_error("Cannot extract an already parameterized expression");
        }
    }

    void extractBinary(BinaryExpr& binary, ParameterType hint, const ResultTypes& results) {
        auto operand_type = commonType(expressionType(binary.left.get(), schema_, results),
                                       expressionType(binary.right.get(), schema_, results));
        if (isArithmetic(binary.op_type) && operand_type == ParameterType::Text)
            throw std::runtime_error("Text arithmetic is not supported");
        if (isArithmetic(binary.op_type) && operand_type == ParameterType::NullOnly) operand_type = hint;
        if (binary.op_type == ExprType::OP_AND || binary.op_type == ExprType::OP_OR || binary.op_type == ExprType::OP_NOT) {
            requireBoolean(binary.left.get(), results);
            requireBoolean(binary.right.get(), results);
            operand_type = ParameterType::Int64;
        }
        extractExpr(binary.left, operand_type, results);
        extractExpr(binary.right, operand_type, results);
    }

    /** @brief Text has no implicit numeric/Boolean conversion, including CASE and NOT. */
    void requireBoolean(const ExprNode* expression, const ResultTypes& results) const {
        if (expressionType(expression, schema_, results) == ParameterType::Text)
            throw std::runtime_error("Text is not a Boolean expression; implicit Text/number conversion is forbidden");
    }

    void extractAggregate(AggregateNode& node) {
        ResultTypes results;
        auto add_result = [&](ParameterType type) {
            results.emplace("__result_col_" + std::to_string(results.size()), type);
        };
        for (auto& group : node.group_by_exprs) {
            extractExpr(group, ParameterType::NullOnly, {});
            add_result(expressionType(group.get(), schema_, {}));
        }
        for (auto& aggregate : node.aggregates) {
            extractExpr(aggregate.agg_expr, ParameterType::NullOnly, {});
            auto type = expressionType(aggregate.agg_expr.get(), schema_, {});
            if (type == ParameterType::Text && (aggregate.isSum() || aggregate.isAvg()))
                throw std::runtime_error("SUM/AVG do not accept Text");
            if (aggregate.isAvg()) type = ParameterType::Float64;
            if (aggregate.isCount()) type = ParameterType::Int64;
            add_result(type);
        }
        requireBoolean(node.having_predicate.get(), results);
        extractExpr(node.having_predicate, ParameterType::Int64, results);
    }

    void extractPlan(OperatorNode& node) {
        switch (node.getType()) {
            case OperatorType::TABLE_SCAN: break;
            case OperatorType::FILTER:
                requireBoolean(static_cast<FilterNode&>(node).predicate.get(), {});
                extractExpr(static_cast<FilterNode&>(node).predicate, ParameterType::Int64, {});
                break;
            case OperatorType::HASH_JOIN:
                extractExpr(static_cast<HashJoinNode&>(node).join_condition, ParameterType::Int64, {});
                break;
            case OperatorType::PROJECTION:
                for (auto& expr : static_cast<ProjectionNode&>(node).select_exprs) {
                    extractExpr(expr, ParameterType::NullOnly, {});
                }
                break;
            case OperatorType::AGGREGATE:
                extractAggregate(static_cast<AggregateNode&>(node));
                break;
            case OperatorType::SORT_LIMIT:
                extractExpr(static_cast<SortLimitNode&>(node).limit_expr, ParameterType::Int64, {});
                break;
        }
        for (auto& child : node.getChildrenMutable()) extractPlan(*child);
    }
};

/** @brief Unambiguous serialization; every token carries its byte length. */
class KeyWriter {
public:
    explicit KeyWriter(const ParameterSchema& schema) : schema_(schema) {}
    std::string take() const { return output_; }
    void token(const std::string& value) {
        output_ += std::to_string(value.size()) + ":" + value;
    }
    void number(std::size_t value) { token(std::to_string(value)); }

    void expression(const ExprNode* expr) {
        if (!expr) { token("absent"); return; }
        number(static_cast<std::size_t>(expr->getType()));
        if (const auto* col = dynamic_cast<const ColumnRefExpr*>(expr)) {
            token(col->table_name);
            token(col->column_name);
        } else if (const auto* parameter = dynamic_cast<const ParameterRefExpr*>(expr)) {
            if (parameter->slot >= schema_.size()) throw std::runtime_error("Parameter slot is outside schema");
            number(parameter->slot);
        } else if (const auto* binary = dynamic_cast<const BinaryExpr*>(expr)) {
            expression(binary->left.get());
            expression(binary->right.get());
        } else if (const auto* branch = dynamic_cast<const CaseWhenExpr*>(expr)) {
            expression(branch->condition.get());
            expression(branch->then_expr.get());
            expression(branch->else_expr.get());
        } else if (expr->getType() != ExprType::STAR) {
            throw std::runtime_error("Unextracted literal in query template");
        }
    }

    void plan(const OperatorNode& node) {
        number(static_cast<std::size_t>(node.getType()));
        switch (node.getType()) {
            case OperatorType::TABLE_SCAN: token(static_cast<const TableScanNode&>(node).table_name); break;
            case OperatorType::FILTER: expression(static_cast<const FilterNode&>(node).predicate.get()); break;
            case OperatorType::HASH_JOIN: expression(static_cast<const HashJoinNode&>(node).join_condition.get()); break;
            case OperatorType::PROJECTION: {
                const auto& projection = static_cast<const ProjectionNode&>(node);
                expressions(projection.select_exprs);
                aliases(projection.output_aliases);
                break;
            }
            case OperatorType::AGGREGATE: aggregate(static_cast<const AggregateNode&>(node)); break;
            case OperatorType::SORT_LIMIT: sort(static_cast<const SortLimitNode&>(node)); break;
        }
        number(node.getChildren().size());
        for (const auto& child : node.getChildren()) plan(*child);
    }

private:
    const ParameterSchema& schema_;
    std::string output_;

    void expressions(const std::vector<std::unique_ptr<ExprNode>>& exprs) {
        number(exprs.size());
        for (const auto& expr : exprs) expression(expr.get());
    }
    void aliases(const std::vector<std::string>& names) {
        number(names.size());
        for (const auto& name : names) token(name);
    }
    void aggregate(const AggregateNode& node) {
        expressions(node.group_by_exprs);
        number(node.aggregates.size());
        for (const auto& aggregate : node.aggregates) {
            token(aggregate.func_name);
            expression(aggregate.agg_expr.get());
        }
        expression(node.having_predicate.get());
        number(node.select_output_slots.size());
        for (const auto slot : node.select_output_slots) number(slot);
        aliases(node.output_aliases);
    }
    void sort(const SortLimitNode& node) {
        number(node.sort_keys.size());
        for (const auto& key : node.sort_keys) {
            number(key.column_index);
            number(key.descending ? 1 : 0);
        }
        expression(node.limit_expr.get());
    }
};

std::unique_ptr<ExprNode> boundLiteral(const ParameterValue& value) {
    if (value.isNull()) return std::make_unique<LiteralNullExpr>();
    if (value.type() == ParameterType::Int64) return std::make_unique<LiteralIntExpr>(value.asInt64());
    if (value.type() == ParameterType::Text) return std::make_unique<LiteralTextExpr>(value.asText());
    return std::make_unique<LiteralFloatExpr>(value.asFloat64());
}

std::unique_ptr<ExprNode> bindExpression(const ExprNode* expr, const ParameterValues& values) {
    if (!expr) return nullptr;
    if (const auto* parameter = dynamic_cast<const ParameterRefExpr*>(expr)) {
        return boundLiteral(values.at(parameter->slot));
    }
    if (const auto* binary = dynamic_cast<const BinaryExpr*>(expr)) {
        return std::make_unique<BinaryExpr>(binary->op_type,
            bindExpression(binary->left.get(), values), bindExpression(binary->right.get(), values));
    }
    if (const auto* branch = dynamic_cast<const CaseWhenExpr*>(expr)) {
        return std::make_unique<CaseWhenExpr>(bindExpression(branch->condition.get(), values),
            bindExpression(branch->then_expr.get(), values), bindExpression(branch->else_expr.get(), values));
    }
    return expr->clone();
}

/** @brief Bind grouped inputs and post-aggregate references without changing output slots. */
std::unique_ptr<AggregateNode> bindAggregate(const AggregateNode& source, const ParameterValues& values) {
    auto result = std::make_unique<AggregateNode>();
    for (const auto& expr : source.group_by_exprs) result->group_by_exprs.push_back(bindExpression(expr.get(), values));
    for (const auto& aggregate : source.aggregates) {
        result->aggregates.emplace_back(aggregate.func_name, bindExpression(aggregate.agg_expr.get(), values));
    }
    result->having_predicate = bindExpression(source.having_predicate.get(), values);
    result->select_output_slots = source.select_output_slots;
    result->output_aliases = source.output_aliases;
    return result;
}

/** @brief Clone one operator's attributes; children are attached by the caller. */
std::unique_ptr<OperatorNode> bindOperator(const OperatorNode& node, const ParameterValues& values) {
    switch (node.getType()) {
        case OperatorType::TABLE_SCAN: {
            const auto& scan = static_cast<const TableScanNode&>(node);
            return std::make_unique<TableScanNode>(scan.table_name, scan.table_alias);
        }
        case OperatorType::FILTER:
            return std::make_unique<FilterNode>(bindExpression(static_cast<const FilterNode&>(node).predicate.get(), values));
        case OperatorType::HASH_JOIN:
            return std::make_unique<HashJoinNode>(bindExpression(static_cast<const HashJoinNode&>(node).join_condition.get(), values));
        case OperatorType::PROJECTION: {
            const auto& source = static_cast<const ProjectionNode&>(node);
            auto result = std::make_unique<ProjectionNode>();
            for (const auto& expr : source.select_exprs) result->select_exprs.push_back(bindExpression(expr.get(), values));
            result->output_aliases = source.output_aliases;
            return result;
        }
        case OperatorType::AGGREGATE:
            return bindAggregate(static_cast<const AggregateNode&>(node), values);
        case OperatorType::SORT_LIMIT: {
            const auto& source = static_cast<const SortLimitNode&>(node);
            auto result = std::make_unique<SortLimitNode>();
            result->sort_keys = source.sort_keys;
            result->limit_expr = bindExpression(source.limit_expr.get(), values);
            if (result->hasLimit()) (void)result->literalLimit();
            return result;
        }
    }
    throw std::runtime_error("Unknown operator in legacy binding");
}

std::unique_ptr<OperatorNode> bindTree(const OperatorNode& node, const ParameterValues& values) {
    auto result = bindOperator(node, values);
    for (const auto& child : node.getChildren()) result->addChild(bindTree(*child, values));
    return result;
}
} // namespace

PreparedQuery parameterizeQuery(std::unique_ptr<OperatorNode> resolved_plan) {
    return ParameterExtractor{}.run(std::move(resolved_plan));
}

std::string QueryTemplate::structureKey() const {
    if (!plan) throw std::runtime_error("Query template has no plan");
    KeyWriter writer(parameter_schema);
    writer.token("sql-structure-v1");
    writer.plan(*plan);
    return writer.take();
}

std::string QueryTemplate::schemaKey() const {
    KeyWriter writer(parameter_schema);
    writer.token("sql-parameters-v1");
    writer.number(parameter_schema.size());
    for (const auto type : parameter_schema) writer.number(static_cast<std::size_t>(type));
    return writer.take();
}

std::string QueryTemplate::cacheKey() const {
    return structureKey() + schemaKey();
}

std::unique_ptr<OperatorNode> bindLegacyPlan(const QueryTemplate& query, const ParameterValues& values) {
    if (!query.plan) throw std::runtime_error("Query template has no plan");
    return bindTree(*query.plan, bindParameterValues(query.parameter_schema, values));
}

} // namespace db
