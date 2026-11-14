#pragma once
/** @file @brief Typed, collision-safe physical lowering for plans that consume or return Text. */
#include "core/operators.h"
#include "core/physical_planner.h"
namespace db {
/** @brief Detect Text from resolved schema, never from parameter bytes or selectivity. */
bool usesTextPlan(const OperatorNode& root, const Catalog& catalog, const ParameterSchema& schema);
/** @brief Lower Text operators to ordinary module contracts and explicit runtime operations. */
PlannedQuery lowerTextPlan(const OperatorNode& root, const Catalog& catalog, const ParameterSchema& schema);
}
