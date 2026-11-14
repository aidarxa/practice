#include "text_plan_detail.h"

namespace db { namespace text_plan {
namespace {
/** @brief Same full comparator for every merge; NULL stays last for ASC and DESC. */
std::string comparator(const Columns& columns, const std::vector<SortKeyDef>& keys) {
    std::string code = "auto before=[&](unsigned long long a,unsigned long long b) {\n";
    for (const auto& key : keys) {
        const auto i = key.column_index;
        if (i >= columns.types.size()) throw std::runtime_error("Text sort key outside visible columns");
        const auto a = columns.values[i] + "[a]", b = columns.values[i] + "[b]";
        code += "{const bool av=" + columns.valid[i] + "[a],bv=" + columns.valid[i] + "[b]; if(av!=bv) return av; if(av) {\n";
        if (columns.types[i] == DB_TYPE_TEXT) {
            code += "const auto c=db::compareText(" + a + "," + b + "); if(c) return c" + (key.descending ? ">0;\n" : "<0;\n");
        } else {
            code += "if(" + a + "<" + b + ") return " + (key.descending ? "false;" : "true;") + "\n";
            code += "if(" + b + "<" + a + ") return " + (key.descending ? "true;" : "false;") + "\n";
        }
        code += "}}\n";
    }
    return code + "return a<b;\n};\n";
}
}
void Builder::sort(const Columns& columns, const std::vector<SortKeyDef>& keys, const std::string& permutation) {
    const auto temporary = name("merge_output");
    scratch(temporary, DB_TYPE_UINT64, variable(columns.count));
    const auto width = scalar("merge_width", 1);
    const auto start = physical.plan.steps.size();
    const auto code = comparator(columns, keys) +
        "const unsigned long long base=(r/(2*" + width + "))*(2*" + width + ");\n"
        "const auto middle=base+" + width + "<db_range_items?base+" + width + ":db_range_items;\n"
        "const auto end=middle+" + width + "<db_range_items?middle+" + width + ":db_range_items;\n"
        "const bool left=r<middle; const auto begin=left?middle:base; const auto finish=left?end:middle;\n"
        "const auto value=" + permutation + "[r]; unsigned long long lo=begin,hi=finish;\n"
        "while(lo<hi) {const auto m=lo+(hi-lo)/2; if(before(" + permutation + "[m],value)) lo=m+1; else hi=m;}\n"
        + temporary + "[base+(r-(left?base:middle))+(lo-begin)]=value;\n";
    launch("merge", code, variable(columns.count), {width});
    physical.plan.steps.emplace_back(CopyBuffer{permutation, temporary, 0, 0, variable(columns.count)});
    physical.plan.steps.emplace_back(SetScalar{width, minimum(variable(columns.count), variable(width) * 2)});
    PlanBlock loop;
    loop.insert(loop.end(), physical.plan.steps.begin() + start, physical.plan.steps.end());
    physical.plan.steps.erase(physical.plan.steps.begin() + start, physical.plan.steps.end());
    physical.plan.steps.emplace_back(Loop{less(variable(width), variable(columns.count)), std::move(loop), 64});
}
void Builder::output(const Columns& columns, const SortLimitNode* sorting) {
    const auto permutation = name("permutation");
    scratch(permutation, DB_TYPE_UINT64, variable(columns.count));
    launch("identity", permutation + "[r]=r;", variable(columns.count));
    if (sorting && !sorting->sort_keys.empty()) sort(columns, sorting->sort_keys, permutation);
    const auto rows = scalar("output_rows", limit(sorting, variable(columns.count)));
    std::vector<size_t> slots = columns.visible;
    if (slots.empty()) for (size_t i=0;i<columns.types.size();++i) slots.push_back(i);
    std::vector<uint32_t> types;
    for (auto slot : slots) types.push_back(columns.types.at(slot));
    output_width_ = types.size();
    physical.columns(physical.plan.steps, types, variable(rows));
    std::string code = "const auto p=" + permutation + "[r];\n";
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto slot = slots[i];
        code += "d_result_col_" + std::to_string(i) + "[r]=" + columns.values[slot] + "[p];\n";
        code += "if(" + columns.valid[slot] + "[p]) db::atomic_set_valid_bit(d_result_validity_col_" + std::to_string(i) + ",r);\n";
    }
    launch("output", code, variable(rows));
    physical.plan.steps.emplace_back(PublishResult{variable(rows), variable(rows) * slots.size(), true, 0});
}
}} // namespace db::text_plan
