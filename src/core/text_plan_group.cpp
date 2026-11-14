#include "text_plan_detail.h"

namespace db { namespace text_plan {
namespace {
struct State { std::string count, value; };
std::vector<State> states(Builder& builder, const AggregateNode& node, const std::string& capacity) {
    std::vector<State> result;
    for (const auto& aggregate : node.aggregates) {
        State state{builder.name("non_null_count"), builder.name("aggregate")};
        builder.scratch(state.count, DB_TYPE_UINT64, variable(capacity), true);
        builder.scratch(state.value, DB_TYPE_UINT64, variable(capacity), true);
        if (aggregate.isMin() || aggregate.isMax())
            builder.physical.plan.steps.emplace_back(FillBuffer{state.value, 0, maximum(variable(capacity), 1), 255});
        result.push_back(std::move(state));
    }
    return result;
}
std::string groupHash(Builder& builder, const AggregateNode& node, const Batch& input) {
    std::string code = "unsigned long long hash=14695981039346656037ULL;\n";
    for (const auto& key : node.group_by_exprs) {
        const auto type = sqlExpressionType(key.get(), builder.schema);
        if (type == ParameterType::Float64) throw std::runtime_error("Float64 GROUP BY remains outside the supported contract");
        code += "hash=(hash^(" + hashValue(builder.expression(key.get(), input, "r"), type) + "))*1099511628211ULL;\n";
    }
    return code;
}
std::string groupEqual(Builder& builder, const AggregateNode& node, const Batch& input) {
    std::string equal = "true";
    for (const auto& key : node.group_by_exprs) {
        const auto a = builder.expression(key.get(), input, "r"), b = builder.expression(key.get(), input, "other");
        equal += " && (((!(" + a.valid + ")) && (!(" + b.valid + "))) || ((" + a.valid + ") && (" + b.valid + ") && db::safe_eq(" + a.value + "," + b.value + ")))";
    }
    return equal;
}
std::string membership(Builder& builder, const AggregateNode& node, const Batch& input,
                       const std::string& groups, const std::string& capacity, const std::string& error) {
    if (node.group_by_exprs.empty()) return "const unsigned long long slot=0;\n";
    return groupHash(builder, node, input) + "unsigned long long slot=hash%" + capacity + ",attempt=0;\n"
        "for(;attempt<" + capacity + ";++attempt) {const auto other=atomicCAS(" + groups + "+slot,~0ULL,r);\n"
        "if(other==~0ULL || (" + groupEqual(builder, node, input) + ")) break;\n"
        "slot=slot+1==" + capacity + "?0:slot+1; }\n"
        "if(attempt==" + capacity + ") {atomicExch(" + error + ",1ULL);return;}\n";
}
std::string extremum(Builder& builder, const AggregateDef& aggregate, const Batch& input,
                     const ExpressionCode& value, const State& state) {
    const auto other = builder.expression(aggregate.agg_expr.get(), input, "other");
    const auto compare = aggregate.isMin() ? "safe_lt" : "safe_gt";
    return "auto other=atomicAdd(" + state.value + "+slot,0ULL);\n"
        "while(other==~0ULL || db::" + compare + "(" + value.value + "," + other.value + ")) {\n"
        "const auto previous=atomicCAS(" + state.value + "+slot,other,r); if(previous==other) break; other=previous; }\n";
}
std::string updates(Builder& builder, const AggregateNode& node, const Batch& input, const std::vector<State>& states) {
    std::string code;
    for (size_t i = 0; i < node.aggregates.size(); ++i) {
        const auto& aggregate = node.aggregates[i];
        const auto value = builder.expression(aggregate.agg_expr.get(), input, "r");
        code += "if(" + value.valid + ") {atomicAdd(" + states[i].count + "+slot,1ULL);\n";
        if (aggregate.isMin() || aggregate.isMax()) code += extremum(builder, aggregate, input, value, states[i]);
        else if (!aggregate.isCount()) {
            const auto type = cppType(storageType(aggregateStateType(aggregate, builder.schema)));
            code += "db::atomicAggregate<" + type + ",db::AggregateOp::Sum>(" + states[i].value + "[slot],static_cast<" + type + ">(" + value.value + "));\n";
        }
        code += "}\n";
    }
    return code;
}
ExpressionCode finalAggregate(Builder& builder, const AggregateDef& aggregate, const Batch& input, const State& state) {
    const auto count = state.count + "[r]", raw = state.value + "[r]";
    if (aggregate.isCount()) return {count, "true"};
    if (aggregate.isMin() || aggregate.isMax()) {
        const auto value = builder.expression(aggregate.agg_expr.get(), input, raw);
        const auto zero = sqlExpressionType(aggregate.agg_expr.get(), builder.schema) == ParameterType::Text ? "db::TextView{nullptr,0}" : "0";
        return {"((" + count + ")? (" + value.value + ") : (" + zero + "))", "(" + count + "!=0)"};
    }
    const auto type = cppType(storageType(aggregateStateType(aggregate, builder.schema)));
    const auto value = "db::aggregateValue<" + type + ">(" + raw + ")";
    if (aggregate.isAvg()) return {"((" + count + ")?static_cast<double>(" + value + ")/static_cast<double>(" + count + "):0.0)", "(" + count + "!=0)"};
    return {value, "(" + count + "!=0)"};
}
std::string finalize(Builder& builder, const AggregateNode& node, const Batch& input,
                      const std::vector<State>& states, const Columns& columns,
                      const std::string& groups, const std::string& flags) {
    std::string code = flags + "[r]=0;\n";
    if (!node.group_by_exprs.empty()) code += "const auto representative=" + groups + "[r]; if(representative==~0ULL) return;\n";
    std::vector<ExpressionCode> values;
    for (const auto& key : node.group_by_exprs) values.push_back(builder.expression(key.get(), input, "representative"));
    for (size_t i=0;i<states.size();++i) values.push_back(finalAggregate(builder, node.aggregates[i], input, states[i]));
    for (size_t i=0;i<values.size();++i) {
        code += "const " + cppType(columns.types[i]) + " result_value_" + std::to_string(i) + "=" + values[i].value + ";\n";
        code += "const bool result_valid_" + std::to_string(i) + "=" + values[i].valid + ";\n";
    }
    const auto having = emitSqlExpression(node.having_predicate.get(), [&](const ColumnRefExpr& column) {
        const auto slot = std::stoull(column.column_name.substr(std::string("__result_col_").size()));
        if (slot >= values.size()) throw std::runtime_error("Text HAVING reference exceeds canonical result");
        return ExpressionCode{"result_value_" + std::to_string(slot), "result_valid_" + std::to_string(slot)};
    }, builder.schema);
    if (node.having_predicate) code += "if(!" + trueValue(having) + ") return;\n";
    code += flags + "[r]=1;\n";
    for (size_t i=0;i<values.size();++i)
        code += columns.values[i] + "[r]=result_value_" + std::to_string(i) + "; " + columns.valid[i] + "[r]=result_valid_" + std::to_string(i) + ";\n";
    return code;
}
Columns compactGroups(Builder& builder, const Columns& source, const std::string& flags) {
    const auto offsets = builder.name("group_offsets"), count = builder.name("dense_groups");
    builder.scratch(offsets, DB_TYPE_UINT64, variable(source.count));
    builder.physical.plan.steps.emplace_back(DevicePrefix{flags, offsets, variable(source.count), count});
    auto output = builder.allocateColumns(source.types, count);
    std::string code = "if(!" + flags + "[r]) return; const auto p=" + offsets + "[r];\n";
    for (size_t i=0;i<source.types.size();++i)
        code += output.values[i] + "[p]=" + source.values[i] + "[r]; " + output.valid[i] + "[p]=" + source.valid[i] + "[r];\n";
    builder.launch("compact_groups", code, variable(source.count));
    return output;
}
}
Columns Builder::aggregate(const AggregateNode& node, const Batch& input) {
    const auto capacity = scalar("group_capacity", node.group_by_exprs.empty() ? SizeExpr{1} : maximum(variable(input.count) * 2, 1));
    const auto groups = name("group_representative"), error = name("group_error");
    scratch(groups, DB_TYPE_UINT64, variable(capacity));
    physical.plan.steps.emplace_back(FillBuffer{groups, 0, variable(capacity), 255});
    scratch(error, DB_TYPE_UINT64, 1, true);
    const auto state = states(*this, node, capacity);
    const auto code = membership(*this, node, input, groups, capacity, error) + updates(*this, node, input, state);
    launch("aggregate", code, variable(input.count), {capacity});
    const auto status = name("group_status");
    physical.plan.steps.emplace_back(ReadScalar{status, error, 0});
    physical.plan.steps.emplace_back(Require{equal(variable(status), 0), "Text group table exhausted bounded probes"});
    std::vector<uint32_t> types;
    for (const auto& key : node.group_by_exprs) types.push_back(storageType(expressionResultType(key.get(), schema)));
    for (const auto& value : node.aggregates) types.push_back(storageType(aggregateResultType(value, schema)));
    auto columns = allocateColumns(types, capacity);
    const auto flags = name("group_selected");
    scratch(flags, DB_TYPE_UINT64, variable(capacity));
    launch("group_finalize", finalize(*this, node, input, state, columns, groups, flags), variable(capacity));
    auto output = compactGroups(*this, columns, flags);
    output.visible = node.select_output_slots;
    return output;
}
}} // namespace db::text_plan
