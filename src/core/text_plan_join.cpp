#include "text_plan_detail.h"
#include "core/optimizer_rules.h"

namespace db { namespace text_plan {
namespace {
/** @brief Each OR branch owns a bucket index; suppress pairs satisfying earlier branches. */
void alternatives(const ExprNode* expression, std::vector<const ExprNode*>& result) {
    if (expression && expression->getType() == ExprType::OP_OR) {
        const auto& binary = static_cast<const BinaryExpr&>(*expression);
        alternatives(binary.left.get(), result);
        alternatives(binary.right.get(), result);
    } else if (expression) result.push_back(expression);
}
bool belongs(const ExprNode* expression, const Batch& batch) {
    if (!expression || expression->getType() != ExprType::COLUMN_REF) return false;
    return batch.rows.count(static_cast<const ColumnRefExpr&>(*expression).table_name) != 0;
}
struct Keys { const ExprNode* build; const ExprNode* probe; };
Keys equality(const ExprNode* expression, const Batch& build, const Batch& probe) {
    if (expression && expression->getType() == ExprType::OP_EQ) {
        const auto& binary = static_cast<const BinaryExpr&>(*expression);
        if (belongs(binary.left.get(), build) && belongs(binary.right.get(), probe)) return {binary.left.get(), binary.right.get()};
        if (belongs(binary.right.get(), build) && belongs(binary.left.get(), probe)) return {binary.right.get(), binary.left.get()};
    }
    if (expression && expression->getType() == ExprType::OP_AND) {
        const auto& binary = static_cast<const BinaryExpr&>(*expression);
        try { return equality(binary.left.get(), build, probe); }
        catch (const std::runtime_error&) { return equality(binary.right.get(), build, probe); }
    }
    throw std::runtime_error("Text inner join requires an equikey in every OR branch");
}
struct Buckets {
    Keys keys;
    std::string counts, offsets, payload, capacity;
};
Batch combine(const Batch& build, const Batch& probe) {
    Batch result = build;
    for (const auto& table : probe.rows)
        if (!result.rows.emplace(table).second) throw std::runtime_error("Repeated table identity in Text join");
    return result;
}
/** @brief Remap tuples to their distinct input positions before evaluating a join expression. */
Batch candidates(const Batch& build, const Batch& probe) {
    auto batch = combine(build, probe);
    for (const auto& table : build.rows) batch.rows[table.first] = "candidate_" + table.first;
    for (const auto& table : probe.rows) batch.rows[table.first] = "candidate_" + table.first;
    return batch;
}
std::string candidateRows(Builder& builder, const Batch& build, const Batch& probe) {
    std::string result;
    for (const auto& table : build.rows)
        result += "const unsigned long long candidate_" + table.first + "[1]={" + builder.row(build, table.first, "b") + "};\n";
    for (const auto& table : probe.rows)
        result += "const unsigned long long candidate_" + table.first + "[1]={" + builder.row(probe, table.first, "r") + "};\n";
    return result;
}
Buckets buildBuckets(Builder& builder, const Batch& input, Keys keys, const std::string& capacity) {
    Buckets table{keys, builder.name("bucket_counts"), builder.name("bucket_offsets"), builder.name("bucket_rows"), capacity};
    const auto position = builder.name("bucket_positions"), total = builder.name("build_total");
    builder.scratch(table.counts, DB_TYPE_UINT64, variable(capacity), true);
    builder.scratch(table.offsets, DB_TYPE_UINT64, variable(capacity));
    builder.scratch(position, DB_TYPE_UINT64, variable(capacity), true);
    builder.scratch(table.payload, DB_TYPE_UINT64, variable(input.count));
    const auto key = builder.expression(keys.build, input, "r");
    const auto hash = hashValue(key, sqlExpressionType(keys.build, builder.schema));
    const auto prefix = "if(!(" + key.valid + ")) return; const auto bucket=(" + hash + ")%" + capacity + ";\n";
    builder.launch("bucket_count", prefix + "atomicAdd(" + table.counts + "+bucket,1ULL);", variable(input.count), {capacity});
    builder.physical.plan.steps.emplace_back(DevicePrefix{table.counts, table.offsets, variable(capacity), total});
    builder.launch("bucket_write", prefix + table.payload + "[" + table.offsets + "[bucket]+atomicAdd(" + position + "+bucket,1ULL)]=r;",
                   variable(input.count), {capacity});
    return table;
}
std::string probeCode(Builder& builder, const Batch& build, const Batch& probe,
                      const std::vector<const ExprNode*>& branches, const std::vector<Buckets>& indexes,
                      const std::string& position, const Batch* output) {
    const auto candidate = candidates(build, probe);
    std::string result = "unsigned long long matches=0;\n";
    for (size_t branch = 0; branch < branches.size(); ++branch) {
        const auto& index = indexes[branch];
        const auto key = builder.expression(index.keys.probe, probe, "r");
        const auto hash = hashValue(key, sqlExpressionType(index.keys.probe, builder.schema));
        result += "if(" + key.valid + ") { const auto bucket=(" + hash + ")%" + index.capacity + ";\n";
        result += "for(unsigned long long j=0;j<" + index.counts + "[bucket];++j) {\n";
        result += "const auto b=" + index.payload + "[" + index.offsets + "[bucket]+j];\n" + candidateRows(builder, build, probe);
        std::string selected = trueValue(builder.expression(branches[branch], candidate, "0"));
        for (size_t earlier = 0; earlier < branch; ++earlier)
            selected += " && !(" + trueValue(builder.expression(branches[earlier], candidate, "0")) + ")";
        result += "if(" + selected + ") {\n";
        if (output) {
            result += "const auto p=" + position + "[r]+matches;\n";
            for (const auto& table : output->rows)
                result += table.second + "[p]=candidate_" + table.first + "[0];\n";
        }
        result += "++matches;\n}\n}\n}\n";
    }
    if (!output) result += position + "[r]=matches;\n";
    return result;
}
}
Batch Builder::join(const HashJoinNode& node) {
    const auto build = relation(*node.getChildren().at(0));
    const auto probe = relation(*node.getChildren().at(1));
    // A pair can be emitted once despite OR alternatives. Bound every device count before launch.
    scalar("maximum_join_pairs", variable(build.count) * variable(probe.count));
    std::vector<const ExprNode*> branches;
    alternatives(node.join_condition.get(), branches);
    if (branches.empty()) throw std::runtime_error("Text cross join is not an equijoin");
    const auto capacity = scalar("buckets", maximum(variable(build.count), 1));
    std::vector<Buckets> indexes;
    for (const auto* branch : branches) indexes.push_back(buildBuckets(*this, build, equality(branch, build, probe), capacity));
    const auto counts = name("join_counts"), offsets = name("join_offsets"), total = name("join_rows");
    scratch(counts, DB_TYPE_UINT64, variable(probe.count));
    scratch(offsets, DB_TYPE_UINT64, variable(probe.count));
    launch("probe_count", probeCode(*this, build, probe, branches, indexes, counts, nullptr), variable(probe.count), {capacity});
    physical.plan.steps.emplace_back(DevicePrefix{counts, offsets, variable(probe.count), total});
    Batch output = combine(build, probe);
    output.count = total;
    for (auto& table : output.rows) {
        table.second = name("joined_row_ids");
        scratch(table.second, DB_TYPE_UINT64, variable(total));
    }
    launch("probe_write", probeCode(*this, build, probe, branches, indexes, offsets, &output), variable(probe.count), {capacity});
    return output;
}
}} // namespace db::text_plan
