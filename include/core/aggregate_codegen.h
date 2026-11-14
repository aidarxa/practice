#pragma once
/** @file @brief Typed aggregate expressions; register/block/atomic paths share one numeric policy. */
#include "core/expression_codegen.h"
#include "core/result_layout.h"
#include <string>
#include <vector>

namespace db {
/** @brief Schema-derived source fragments, never a holder of current parameter values. */
class AggregateCode {
public:
    AggregateCode(const AggregateDef& aggregate, const ParameterSchema& schema);
    std::string type;
    std::string scratch;
    bool tracks_validity;
    std::string identity() const;
    std::string merge(const std::string& left, const std::string& right) const;
    std::string atomic(const std::string& target, const std::string& value) const;
    std::string reduce(const std::string& value) const;
    std::string read(const std::string& raw) const;
private:
    std::string operation_;
};
/** @brief Emit one fused scalar aggregation, with a validity count independent of its numeric identity. */
std::string emitScalarAggregate(const AggregateNode& node, const ParameterSchema& schema,
                                const std::vector<ExpressionCode>& inputs);
} // namespace db
