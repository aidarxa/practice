#include "core/result_layout.h"
#include "core/operators.h"

namespace db {
LogicalType expressionResultType(const ExprNode* expression, const ParameterSchema& schema) {
    if (!expression) return LogicalType::UInt64;
    const auto type = expression->getType();
    if (type >= ExprType::OP_AND && type <= ExprType::OP_GTE) return LogicalType::UInt64;
    if (type == ExprType::OP_IS_NULL || type == ExprType::OP_IS_NOT_NULL) return LogicalType::UInt64;
    if (type == ExprType::LITERAL_NULL) return LogicalType::UInt64;
    const auto value_type = sqlExpressionType(expression, schema);
    if (value_type == ParameterType::Text) return LogicalType::Text;
    return value_type == ParameterType::Float64 ? LogicalType::Float64 : LogicalType::Int64;
}
LogicalType aggregateStateType(const AggregateDef& aggregate, const ParameterSchema& schema) {
    if (aggregate.isCount()) return LogicalType::UInt64;
    const auto type = sqlExpressionType(aggregate.agg_expr.get(), schema);
    if (type == ParameterType::Text) return LogicalType::Text;
    return type == ParameterType::Float64 ? LogicalType::Float64 : LogicalType::Int64;
}
LogicalType aggregateResultType(const AggregateDef& aggregate, const ParameterSchema& schema) {
    return aggregate.isAvg() ? LogicalType::Float64 : aggregateStateType(aggregate, schema);
}
LogicalType aggregateSlotType(const AggregateNode& node, size_t slot, const ParameterSchema& schema) {
    if (slot < node.group_by_exprs.size()) {
        const auto* group = node.group_by_exprs[slot].get();
        return expressionResultType(group, schema);
    }
    return aggregateResultType(node.aggregates.at(slot - node.group_by_exprs.size()), schema);
}
} // namespace db
