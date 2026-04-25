#pragma once
#include "../deps/include/SQLParser.h"
#include <vector>

inline void flattenAndConditions(hsql::Expr* expr, std::vector<hsql::Expr*>& flat_list) {
    if (!expr) return;
    if (expr->isType(hsql::kExprOperator) && expr->opType == hsql::kOpAnd) {
        flattenAndConditions(expr->expr, flat_list);
        flattenAndConditions(expr->expr2, flat_list);
    } else {
        flat_list.push_back(expr);
    }
}