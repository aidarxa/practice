#include "core/aggregate_codegen.h"
#include "core/operators.h"
#include <sstream>

namespace db {
AggregateCode::AggregateCode(const AggregateDef& aggregate, const ParameterSchema& schema)
    : tracks_validity(!aggregate.isCount()) {
    const auto state = aggregateStateType(aggregate, schema);
    type = state == LogicalType::Float64 ? "double" : state == LogicalType::Int64 ? "std::int64_t" : "unsigned long long";
    scratch = state == LogicalType::Float64 ? "db_collective_f64" : state == LogicalType::Int64 ? "db_collective_i64" : "db_collective_u64";
    operation_ = aggregate.isMin() ? "Minimum" : aggregate.isMax() ? "Maximum" : "Sum";
}
std::string AggregateCode::identity() const {
    return "db::aggregateIdentity<" + type + ",db::AggregateOp::" + operation_ + ">()";
}
std::string AggregateCode::merge(const std::string& left, const std::string& right) const {
    return "db::AggregateMerge<" + type + ",db::AggregateOp::" + operation_ + ">{}(" + left + ",static_cast<" + type + ">(" + right + "))";
}
std::string AggregateCode::atomic(const std::string& target, const std::string& value) const {
    return "db::atomicAggregate<" + type + ",db::AggregateOp::" + operation_ + ">(" + target + ",static_cast<" + type + ">(" + value + "));\n";
}
std::string AggregateCode::reduce(const std::string& value) const {
    return "crystal::blockReduce<" + type + ">(" + value + "," + scratch + ",threadIdx.x,blockDim.x,db::AggregateMerge<" + type + ",db::AggregateOp::" + operation_ + ">{})";
}
std::string AggregateCode::read(const std::string& raw) const {
    return "db::aggregateValue<" + type + ">(" + raw + ")";
}

namespace {
void scalarRegisters(std::ostream& out, const std::vector<AggregateCode>& states) {
    for (size_t i = 0; i < states.size(); ++i) {
        out << states[i].type << " local_" << i << "=" << states[i].identity() << ";\n";
        if (states[i].tracks_validity) out << "unsigned long long local_seen_" << i << "=0ULL;\n";
    }
}
void scalarInputs(std::ostream& out, const std::vector<AggregateCode>& states, const std::vector<ExpressionCode>& inputs) {
    out << "for(int i=0;i<ITEMS_PER_THREAD;++i) {\nif(flags[i] && tid+BLOCK_THREADS*i<num_tile_items) {\n";
    for (size_t i = 0; i < states.size(); ++i) {
        out << "if(" << inputs.at(i).valid << ") {local_" << i << "="
            << states[i].merge("local_" + std::to_string(i), inputs[i].value) << ";";
        if (states[i].tracks_validity) out << "++local_seen_" << i << ";";
        out << "}\n";
    }
    out << "}}\n";
}
void scalarOutputs(std::ostream& out, const AggregateNode& node, const std::vector<AggregateCode>& states) {
    size_t hidden = node.visibleTupleSize();
    for (size_t i = 0; i < states.size(); ++i) {
        out << "const auto agg_" << i << "=" << states[i].reduce("local_" + std::to_string(i)) << ";\n";
        if (!states[i].tracks_validity) {
            out << "if(tid==0) {" << states[i].atomic("d_result[" + std::to_string(i) + "]", "agg_" + std::to_string(i))
                << "db::atomic_set_valid_bit(d_result_validity," << i << ");}\n";
            continue;
        }
        out << "const auto seen_" << i << "=crystal::blockReduce<unsigned long long>(local_seen_" << i
            << ",db_collective_u64,threadIdx.x,blockDim.x,crystal::Sum<unsigned long long>{});\n"
            << "if(tid==0) {\nif(seen_" << i << ") {" << states[i].atomic("d_result[" + std::to_string(i) + "]", "agg_" + std::to_string(i));
        if (node.aggregates[i].isAvg()) out << "db::atomic_add_ull(d_result[" << hidden++ << "],seen_" << i << ");\n";
        else out << "db::atomic_set_valid_bit(d_result_validity," << i << ");\n";
        out << "}\n";
        out << "}\n";
    }
}
} // namespace
std::string emitScalarAggregate(const AggregateNode& node, const ParameterSchema& schema, const std::vector<ExpressionCode>& inputs) {
    if (inputs.size() != node.aggregates.size()) throw std::runtime_error("aggregate input count differs from layout");
    std::vector<AggregateCode> states;
    for (const auto& aggregate : node.aggregates) states.emplace_back(aggregate, schema);
    std::ostringstream out;
    scalarRegisters(out, states);
    scalarInputs(out, states, inputs);
    scalarOutputs(out, node, states);
    return out.str();
}
} // namespace db
