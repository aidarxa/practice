#pragma once
/** @file @brief Schema-only numeric layout shared by execution, aggregate codegen, HAVING and sorting. */
#include "core/parameter_value.h"
#include "core/memory.h"
#include <cstddef>

namespace db {
class ExprNode;
struct AggregateDef;
class AggregateNode;
/** @brief Infer an existing numeric result type without reading parameter values or validity. */
LogicalType expressionResultType(const ExprNode* expression, const ParameterSchema& schema = {});
/** @brief COUNT uses UInt64; other states preserve signed integer or fractional input. */
LogicalType aggregateStateType(const AggregateDef& aggregate, const ParameterSchema& schema = {});
/** @brief AVG exposes Float64 independently of its sum-state type. */
LogicalType aggregateResultType(const AggregateDef& aggregate, const ParameterSchema& schema = {});
/** @brief Resolve one group-first visible slot; GROUP BY CASE remains integer/NULL only. */
LogicalType aggregateSlotType(const AggregateNode& node, size_t slot, const ParameterSchema& schema = {});
} // namespace db
