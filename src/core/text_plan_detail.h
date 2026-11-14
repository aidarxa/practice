#pragma once
/** @file @brief Internal typed lowering vocabulary; no runtime pointers or parameter values. */
#include "core/text_plan.h"
#include "core/result_layout.h"
#include <map>
#include <sstream>

namespace db { namespace text_plan {
/** @brief GPU-resident tuple identities; an empty row-array name denotes an identity scan. */
struct Batch {
    std::map<std::string, std::string> rows;
    std::string count;
};
/** @brief Dense typed GPU columns and byte validity, prior to final bitmap publication. */
struct Columns {
    std::vector<uint32_t> types;
    std::vector<std::string> values, valid;
    std::string count;
    std::vector<size_t> visible; ///< Optional SELECT mapping; sorting/HAVING retain canonical slots.
};
uint32_t storageType(LogicalType type);
std::string cppType(uint32_t type);
std::string trueValue(const ExpressionCode& expression);
std::string hashValue(const ExpressionCode& expression, ParameterType type);
/** @brief Coalesce non-overlapping typed scratch lifetimes; Text views borrow only base/parameter bytes. */
PlannedQuery coalesceScratch(PlannedQuery query);
/** @brief Own structural lowering state only; PhysicalPlanBuilder owns contracts, not GPU memory. */
class Builder {
public:
    Builder(const Catalog& catalog, const ParameterSchema& schema) : catalog(catalog), schema(schema) {}
    const Catalog& catalog;
    const ParameterSchema& schema;
    PhysicalPlanBuilder physical;
    std::string name(const std::string& prefix);
    std::string scalar(const std::string& prefix, SizeExpr value);
    void scratch(const std::string& name, uint32_t type, SizeExpr count, bool clear = false);
    void launch(const std::string& prefix, const std::string& code, SizeExpr count, std::vector<std::string> captures = {});
    ExpressionCode expression(const ExprNode* expression, const Batch& batch, const std::string& row);
    ExpressionCode column(const ColumnRefExpr& column, const Batch& batch, const std::string& row);
    std::string row(const Batch& batch, const std::string& table, const std::string& index) const;
    Batch relation(const OperatorNode& node);
    Batch filter(const FilterNode& node, Batch input);
    Batch join(const HashJoinNode& node);
    Batch compact(Batch input, const std::string& counts);
    Columns projection(const ProjectionNode& node, const Batch& input, const SortLimitNode* sort);
    Columns aggregate(const AggregateNode& node, const Batch& input);
    Columns allocateColumns(const std::vector<uint32_t>& types, const std::string& count);
    void output(const Columns& columns, const SortLimitNode* sort);
    void sort(const Columns& columns, const std::vector<SortKeyDef>& keys, const std::string& permutation);
    SizeExpr limit(const SortLimitNode* sort, SizeExpr count);
    PlannedQuery finish(const OperatorNode& root);
private:
    uint64_t serial_ = 0;
    size_t output_width_ = 1;
};
}} // namespace db::text_plan
