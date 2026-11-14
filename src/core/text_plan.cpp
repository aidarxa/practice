#include "text_plan_detail.h"
#include "core/optimizer_rules.h"
#include <climits>

namespace db { namespace text_plan {
uint32_t storageType(LogicalType type) {
    if (type == LogicalType::Text) return DB_TYPE_TEXT;
    if (type == LogicalType::Int64) return DB_TYPE_INT64;
    if (type == LogicalType::Float64) return DB_TYPE_FLOAT64;
    return DB_TYPE_UINT64;
}
std::string cppType(uint32_t type) {
    if (type == DB_TYPE_TEXT) return "db::TextView";
    if (type == DB_TYPE_INT64) return "std::int64_t";
    if (type == DB_TYPE_FLOAT64) return "double";
    return "unsigned long long";
}
std::string trueValue(const ExpressionCode& expression) {
    return "((" + expression.valid + ") && (" + expression.value + "))";
}
std::string hashValue(const ExpressionCode& expression, ParameterType type) {
    const auto hash = type == ParameterType::Text ? "db::hashText(" + expression.value + ")" :
        "(static_cast<unsigned long long>(" + expression.value + ")*11400714819323198485ULL)";
    return "((" + expression.valid + ") ? (" + hash + ") : 0x9e3779b97f4a7c15ULL)";
}
std::string Builder::name(const std::string& prefix) { return "text_" + prefix + "_" + std::to_string(serial_++); }
std::string Builder::scalar(const std::string& prefix, SizeExpr value) {
    const auto result = name(prefix);
    physical.plan.steps.emplace_back(SetScalar{result, std::move(value)});
    return result;
}
void Builder::scratch(const std::string& buffer, uint32_t type, SizeExpr count, bool clear) {
    physical.scratch(physical.plan.steps, buffer, type, count, clear);
}
void Builder::launch(const std::string& prefix, const std::string& code, SizeExpr count, std::vector<std::string> captures) {
    physical.range(physical.plan.steps, name(prefix), "const auto r=gid[0];\n" + code, count, std::move(captures));
}
std::string Builder::row(const Batch& batch, const std::string& table, const std::string& index) const {
    const auto& buffer = batch.rows.at(table);
    return buffer.empty() ? "(" + index + ")" : buffer + "[" + index + "]";
}
ExpressionCode Builder::column(const ColumnRefExpr& column, const Batch& batch, const std::string& index) {
    const auto& table = catalog.getTableMetadata(column.table_name);
    const auto row_id = row(batch, column.table_name, index);
    const auto& logical = column.column_name;
    const auto storage = table.getColumnStorage(logical);
    const bool text = !storage.isNumeric();
    const auto validity = "text_null_" + logical;
    if (table.isColumnNullable(logical))
        physical.declareBuffer({validity, DB_TYPE_UINT64, text ? BufferRole::TextValidity : BufferRole::NullBitmap,
                                text ? logical : "d_" + logical, 0, true, table.getSize()/64+(table.getSize()%64!=0), "std::uint64_t"});
    const auto valid = table.isColumnNullable(logical) ? "db::bitmap_valid_at(" + validity + "," + row_id + ")" : "true";
    if (!text) {
        physical.declareBuffer({"d_" + logical, DB_TYPE_INT32, BufferRole::External, "d_" + logical, 0, false, table.getSize()});
        return {"static_cast<std::int64_t>(d_" + logical + "[" + row_id + "])", valid};
    }
    const auto bytes = "text_bytes_" + logical, offsets = "text_offsets_" + logical;
    const auto offset_type = checkedSizeMultiply(table.getSize(), storage.width) <= INT32_MAX ? DB_TYPE_INT32 : DB_TYPE_UINT64;
    physical.declareBuffer({bytes, DB_TYPE_UINT8, BufferRole::TextBytes, logical});
    physical.declareBuffer({offsets, static_cast<uint32_t>(offset_type), BufferRole::TextOffsets, logical, 0, false, 0,
                            offset_type == DB_TYPE_UINT64 ? "std::uint64_t" : "int"});
    return {"db::textAt(" + bytes + "," + offsets + "," + row_id + ")", valid};
}
ExpressionCode Builder::expression(const ExprNode* node, const Batch& batch, const std::string& index) {
    return emitSqlExpression(node, [&](const ColumnRefExpr& reference) { return column(reference, batch, index); }, schema);
}
Batch Builder::relation(const OperatorNode& node) {
    if (node.getType() == OperatorType::TABLE_SCAN) {
        const auto& scan = static_cast<const TableScanNode&>(node);
        return {{{scan.table_name, ""}}, scalar("scan_rows", catalog.getTableMetadata(scan.table_name).getSize())};
    }
    if (node.getType() == OperatorType::FILTER)
        return filter(static_cast<const FilterNode&>(node), relation(*node.getChildren().at(0)));
    if (node.getType() == OperatorType::HASH_JOIN) return join(static_cast<const HashJoinNode&>(node));
    throw std::runtime_error("Text relation requires a scan, filter or supported inner equijoin");
}
Batch Builder::compact(Batch input, const std::string& counts) {
    const auto offsets = name("offsets"), total = name("rows");
    scratch(offsets, DB_TYPE_UINT64, variable(input.count));
    physical.plan.steps.emplace_back(DevicePrefix{counts, offsets, variable(input.count), total});
    Batch output{{}, total};
    std::string code = "if(!" + counts + "[r]) return; const auto p=" + offsets + "[r];\n";
    for (const auto& table : input.rows) {
        const auto buffer = name("row_ids");
        scratch(buffer, DB_TYPE_UINT64, variable(total));
        output.rows.emplace(table.first, buffer);
        code += buffer + "[p]=" + row(input, table.first, "r") + ";\n";
    }
    launch("compact", code, variable(input.count));
    return output;
}
Batch Builder::filter(const FilterNode& node, Batch input) {
    if (sqlExpressionType(node.predicate.get(), schema) == ParameterType::Text)
        throw std::runtime_error("Text is not a Boolean WHERE predicate");
    const auto counts = name("selected");
    scratch(counts, DB_TYPE_UINT64, variable(input.count));
    const auto predicate = expression(node.predicate.get(), input, "r");
    launch("filter", counts + "[r]=" + trueValue(predicate) + ";", variable(input.count));
    return compact(std::move(input), counts);
}
Columns Builder::allocateColumns(const std::vector<uint32_t>& types, const std::string& count) {
    Columns output{types, {}, {}, count};
    for (auto type : types) {
        output.values.push_back(name("value"));
        output.valid.push_back(name("valid"));
        scratch(output.values.back(), type, variable(count));
        scratch(output.valid.back(), DB_TYPE_UINT8, variable(count));
    }
    return output;
}
SizeExpr Builder::limit(const SortLimitNode* sort, SizeExpr count) {
    if (!sort || !sort->limit_expr) return count;
    const auto* slot = dynamic_cast<const ParameterRefExpr*>(sort->limit_expr.get());
    if (!slot) throw std::runtime_error("Text LIMIT must be parameterized");
    physical.plan.size_parameters.push_back(slot->slot);
    return minimum(count, variable("db_size_parameter_" + std::to_string(slot->slot)));
}
Columns Builder::projection(const ProjectionNode& node, const Batch& input, const SortLimitNode* sort) {
    std::vector<const TableScanNode*> scans;
    const auto collect = [&](const auto& self, const OperatorNode& current) -> void {
        if (current.getType() == OperatorType::TABLE_SCAN) scans.push_back(static_cast<const TableScanNode*>(&current));
        for (const auto& child : current.getChildren()) self(self, *child);
    };
    collect(collect, node);
    std::vector<std::unique_ptr<ExprNode>> expressions;
    for (const auto& expression : node.select_exprs) {
        if (expression->getType() != ExprType::STAR) { expressions.push_back(expression->clone()); continue; }
        for (const auto* scan : scans) {
            const auto& metadata = catalog.getTableMetadata(scan->table_name);
            for (const auto& name : metadata.getColumnNames()) {
                auto column = std::make_unique<ColumnRefExpr>(name, scan->table_name);
                column->value_type = metadata.getColumnStorage(name).isNumeric() ? ParameterType::Int64 : ParameterType::Text;
                expressions.push_back(std::move(column));
            }
        }
    }
    std::vector<uint32_t> types;
    for (const auto& expression : expressions) types.push_back(storageType(expressionResultType(expression.get(), schema)));
    const auto count = scalar("project_rows", !sort || sort->sort_keys.empty() ? limit(sort, variable(input.count)) : variable(input.count));
    auto output = allocateColumns(types, count);
    std::string code;
    for (size_t i = 0; i < expressions.size(); ++i) {
        const auto value = expression(expressions[i].get(), input, "r");
        code += output.values[i] + "[r]=" + value.value + ";\n" + output.valid[i] + "[r]=" + value.valid + ";\n";
    }
    launch("projection", code, variable(count));
    return output;
}
PlannedQuery Builder::finish(const OperatorNode& root) {
    physical.declareBuffer({"d_result", DB_TYPE_UINT64, BufferRole::Result});
    const SortLimitNode* sorting = root.getType() == OperatorType::SORT_LIMIT ? static_cast<const SortLimitNode*>(&root) : nullptr;
    const auto& producer = sorting ? *root.getChildren().at(0) : root;
    auto input = relation(*producer.getChildren().at(0));
    Columns columns;
    if (producer.getType() == OperatorType::PROJECTION)
        columns = projection(static_cast<const ProjectionNode&>(producer), input, sorting);
    else if (producer.getType() == OperatorType::AGGREGATE)
        columns = aggregate(static_cast<const AggregateNode&>(producer), input);
    else throw std::runtime_error("Text plan requires a projection or aggregation");
    output(columns, sorting);
    return coalesceScratch(physical.finish(output_width_, 1, schema));
}
}} // namespace db::text_plan

namespace db {
namespace {
bool textExpression(const ExprNode* expression, const ParameterSchema& schema) {
    if (!expression) return false;
    if (expression->getType() == ExprType::COLUMN_REF)
        return static_cast<const ColumnRefExpr&>(*expression).value_type == ParameterType::Text;
    if (expression->getType() == ExprType::PARAMETER_REF)
        return schema.at(static_cast<const ParameterRefExpr&>(*expression).slot) == ParameterType::Text;
    if (const auto* binary = dynamic_cast<const BinaryExpr*>(expression))
        return textExpression(binary->left.get(), schema) || textExpression(binary->right.get(), schema);
    if (const auto* branch = dynamic_cast<const CaseWhenExpr*>(expression))
        return textExpression(branch->condition.get(), schema) || textExpression(branch->then_expr.get(), schema) || textExpression(branch->else_expr.get(), schema);
    return expression->getType() == ExprType::LITERAL_TEXT;
}
}
bool usesTextPlan(const OperatorNode& root, const Catalog& catalog, const ParameterSchema& schema) {
    if (root.getType() == OperatorType::FILTER && textExpression(static_cast<const FilterNode&>(root).predicate.get(), schema)) return true;
    if (root.getType() == OperatorType::HASH_JOIN && textExpression(static_cast<const HashJoinNode&>(root).join_condition.get(), schema)) return true;
    if (const auto* project = dynamic_cast<const ProjectionNode*>(&root)) {
        for (const auto& expression : project->select_exprs) {
            if (textExpression(expression.get(), schema)) return true;
            if (expression->getType() != ExprType::STAR) continue;
            for (const auto& table : collectTableNames(&root))
                for (const auto& column : catalog.getTableMetadata(table).getColumnNames())
                    if (!catalog.getTableMetadata(table).getColumnStorage(column).isNumeric()) return true;
        }
    }
    if (const auto* aggregate = dynamic_cast<const AggregateNode*>(&root)) {
        for (const auto& group : aggregate->group_by_exprs) if (textExpression(group.get(), schema)) return true;
        for (const auto& value : aggregate->aggregates) if (textExpression(value.agg_expr.get(), schema)) return true;
        if (textExpression(aggregate->having_predicate.get(), schema)) return true;
    }
    for (const auto& child : root.getChildren()) if (usesTextPlan(*child, catalog, schema)) return true;
    return false;
}
PlannedQuery lowerTextPlan(const OperatorNode& root, const Catalog& catalog, const ParameterSchema& schema) {
    return text_plan::Builder(catalog, schema).finish(root);
}
} // namespace db
