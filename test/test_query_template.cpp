/** @file
 * @brief Behavioural tests for the value-free SQL frontend and legacy binding boundary.
 */
#include "core/query_template.h"
#include "core/optimizer_rules.h"
#include "core/translator.h"

#include <SQLParser.h>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <cmath>
#include <sstream>

namespace {
using namespace db;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function action, const std::string& diagnostic) {
    try {
        action();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(diagnostic) != std::string::npos,
                "Wrong diagnostic: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected rejection: " + diagnostic);
}

Catalog catalog() {
    Catalog result;
    result.emplaceTableMetadata("LINEORDER",
        std::vector<std::string>{"lo_quantity", "lo_revenue", "lo_custkey"}, 100, true);
    result.emplaceTableMetadata("CUSTOMER",
        std::vector<std::string>{"c_custkey", "c_region"}, 10, false);
    result.emplaceTableMetadata("T", std::vector<std::string>{"a", "b"}, 4, false);
    result.emplaceTableMetadata("U", std::vector<std::string>{"a", "c"}, 4, false);
    return result;
}

PreparedQuery prepare(const std::string& sql, const Catalog& metadata = catalog()) {
    hsql::SQLParserResult parsed;
    hsql::SQLParser::parse(sql, &parsed);
    require(parsed.isValid() && parsed.size() == 1, "Test SQL did not parse: " + sql);
    const auto* stmt = static_cast<const hsql::SelectStatement*>(parsed.getStatement(0));
    QueryTranslator translator;
    auto result = translator.prepare(stmt, metadata);
    Optimizer optimizer;
    result.query.plan = optimizer.optimize(std::move(result.query.plan));
    return result;
}

const ProjectionNode& projection(const OperatorNode& node) {
    if (node.getType() == OperatorType::PROJECTION) {
        return static_cast<const ProjectionNode&>(node);
    }
    require(node.getChildren().size() == 1, "Expected unary path to projection");
    return projection(*node.getChildren()[0]);
}

const AggregateNode& aggregate(const OperatorNode& node) {
    if (node.getType() == OperatorType::AGGREGATE) {
        return static_cast<const AggregateNode&>(node);
    }
    require(node.getChildren().size() == 1, "Expected unary path to aggregate");
    return aggregate(*node.getChildren()[0]);
}

void test_template_identity() {
    const auto first = prepare("SELECT a FROM t WHERE a > 10");
    const auto changed = prepare("SELECT a FROM t WHERE a > 15");
    const auto operation = prepare("SELECT a FROM t WHERE a < 10");
    require(first.query.cacheKey() == changed.query.cacheKey(), "Values must not enter a template key");
    require(first.query.structureKey() != operation.query.structureKey(), "Comparison is structural");
    require(first.parameters.at(0).asInt64() == 10 && changed.parameters.at(0).asInt64() == 15,
            "Values must remain separate from the template");
    require(first.query.structureKey() == prepare(" select T.A from T where T.A > 999 ").query.structureKey(),
            "Resolved identifiers, not SQL spelling, define structure");
}

void test_all_value_locations() {
    const auto first = prepare(
        "SELECT SUM(lo_revenue + 2) AS total FROM lineorder WHERE lo_quantity > 10 "
        "GROUP BY lo_quantity HAVING total > 100 ORDER BY total DESC LIMIT 3");
    const auto changed = prepare(
        "SELECT SUM(lo_revenue + 5) AS total FROM lineorder WHERE lo_quantity > 20 "
        "GROUP BY lo_quantity HAVING total > 200 ORDER BY total DESC LIMIT 8");
    require(first.parameters.size() == 4, "WHERE, arithmetic, HAVING and LIMIT need four slots");
    require(first.query.cacheKey() == changed.query.cacheKey(), "All ordinary values must be extracted");
    const auto branch = prepare(
        "SELECT CASE WHEN a > 10 THEN 1 ELSE 2 END AS bucket FROM t");
    const auto branch_changed = prepare(
        "SELECT CASE WHEN a > 15 THEN 100 ELSE 100 END AS bucket FROM t");
    require(branch.parameters.size() == 3, "CASE condition and both results are parameters");
    require(branch.query.cacheKey() == branch_changed.query.cacheKey(), "Equal CASE values cannot specialize a plan");
    const auto group = prepare(
        "SELECT SUM(a) FROM t GROUP BY CASE WHEN a > 10 THEN 1 ELSE 2 END");
    const auto group_changed = prepare(
        "SELECT SUM(a) FROM t GROUP BY CASE WHEN a > 15 THEN 100 ELSE 100 END");
    require(group.query.cacheKey() == group_changed.query.cacheKey(), "GROUP BY CASE domain is not structural");
}

void test_independent_and_cloned_slots() {
    auto prepared = prepare("SELECT a + 10, b + 10 FROM t WHERE a > 10");
    require(prepared.parameters.size() == 3, "Equal independent literals must not merge");
    const auto& selected = projection(*prepared.query.plan).select_exprs;
    const auto& first = static_cast<const BinaryExpr&>(*selected[0]);
    const auto& second = static_cast<const BinaryExpr&>(*selected[1]);
    const auto& p = static_cast<const ParameterRefExpr&>(*first.right);
    const auto& q = static_cast<const ParameterRefExpr&>(*second.right);
    require(p.slot != q.slot, "Separate SELECT expressions must have separate slots");
    auto cloned = first.clone();
    const auto& copy = static_cast<const BinaryExpr&>(*cloned);
    require(static_cast<const ParameterRefExpr&>(*copy.right).slot == p.slot, "Clone must preserve slot identity");
    const auto simple_case = prepare("SELECT CASE (a + 7) WHEN 1 THEN 10 WHEN 2 THEN 20 END FROM t");
    require(simple_case.parameters.size() == 6, "Simple CASE expansion must preserve its single base parameter");
}

void test_reference_identity_before_values() {
    const auto same = prepare(
        "SELECT SUM(a + 10) AS x, SUM(a + 10) AS y FROM t GROUP BY b HAVING y > 10 ORDER BY y");
    const auto changed = prepare(
        "SELECT SUM(a + 10) AS x, SUM(a + 20) AS y FROM t GROUP BY b HAVING y > 30 ORDER BY y");
    require(same.query.cacheKey() == changed.query.cacheKey(), "Independent selected expressions cannot collapse by value");
    require(same.parameters.size() == 3, "HAVING alias must not copy aggregate parameters");
    const auto& having = static_cast<const BinaryExpr&>(*aggregate(*same.query.plan).having_predicate);
    const auto& reference = static_cast<const ColumnRefExpr&>(*having.left);
    require(reference.column_name == "__result_col_2", "HAVING must reference the second aggregate output");

    const auto function_reference = prepare(
        "SELECT SUM(l.lo_quantity + 1), SUM(l.lo_quantity + 2) FROM lineorder l "
        "GROUP BY lo_revenue HAVING SUM(lineorder.lo_quantity + 2) > 3 ORDER BY SUM(lineorder.lo_quantity + 1)");
    require(function_reference.parameters.size() == 3, "Resolved aggregate references do not create value slots");
    const auto& predicate = static_cast<const BinaryExpr&>(*aggregate(*function_reference.query.plan).having_predicate);
    require(static_cast<const ColumnRefExpr&>(*predicate.left).column_name == "__result_col_2",
            "Aggregate reference must be resolved using original values");
    rejects([] { prepare("SELECT SUM(a + 1) FROM t GROUP BY b HAVING SUM(a + 2) > 3"); }, "must also appear");
    const auto grouped_case = prepare(
        "SELECT SUM(b) FROM t GROUP BY CASE WHEN a > 1 THEN 10 ELSE 20 END "
        "HAVING CASE WHEN a > 1 THEN 10 ELSE 20 END > 15");
    require(grouped_case.parameters.size() == 4, "HAVING must reference a grouped expression, not copy its parameters");
}

void test_order_positions_are_structural() {
    const auto first = prepare("SELECT a + 10, b + 20 FROM t ORDER BY 1 LIMIT 3");
    const auto second = prepare("SELECT a + 10, b + 20 FROM t ORDER BY 2 LIMIT 3");
    require(first.parameters.size() == 3, "ORDER BY ordinal must not become a fourth parameter");
    require(first.query.structureKey() != second.query.structureKey(), "Resolved ORDER BY output identity is structural");
    require(first.query.structureKey() != prepare("SELECT a + 10, b + 20 FROM t ORDER BY 1 DESC LIMIT 3").query.structureKey(),
            "Sort direction is structural");
    rejects([] { prepare("SELECT a FROM t ORDER BY 0"); }, "ordinal");
    rejects([] { prepare("SELECT a FROM t ORDER BY 2"); }, "ordinal");
}

void test_parameter_types_and_nulls() {
    const auto integer = prepare("SELECT a + 9007199254740993 FROM t");
    require(integer.parameters.at(0).asInt64() == INT64_C(9007199254740993), "Do not round Int64 through double");
    const auto floating = prepare("SELECT a + 1.25 FROM t");
    require(integer.query.structureKey() == floating.query.structureKey(), "Literal types belong to schema, not structure");
    require(integer.query.schemaKey() != floating.query.schemaKey(), "Int64 and Float64 schemas differ");
    require(floating.parameters.at(0).asFloat64() == 1.25, "Float64 value must survive extraction");

    const auto typed_null = prepare("SELECT a FROM t WHERE a > NULL");
    const auto typed_int = prepare("SELECT a FROM t WHERE a > 10");
    require(typed_null.query.cacheKey() == typed_int.query.cacheKey(), "Column context must type the first NULL");
    require(typed_null.parameters.at(0).isNull(), "Typed NULL must retain invalidity");
    require(typed_null.query.parameter_schema.at(0) == ParameterType::Int64, "Column NULL is Int64");
    const auto only_null = prepare("SELECT NULL FROM t");
    require(only_null.query.parameter_schema.at(0) == ParameterType::NullOnly, "Ambiguous NULL remains NULL-only");
    const auto float_null = prepare("SELECT NULL + 1.5 FROM t");
    require(float_null.query.parameter_schema.at(0) == ParameterType::Float64, "Arithmetic gives NULL its numeric context");
    const auto case_null = prepare("SELECT CASE WHEN a > 1 THEN NULL ELSE 1.5 END FROM t");
    require(case_null.query.parameter_schema.at(1) == ParameterType::Float64, "CASE branch gives NULL its result context");
    const auto having_null = prepare("SELECT AVG(a) AS average FROM t GROUP BY b HAVING average > NULL");
    require(having_null.query.parameter_schema.at(0) == ParameterType::Float64, "HAVING sees aggregate output type");
}

void test_legacy_binding_is_pure_and_checked() {
    auto prepared = prepare("SELECT a + 10, b + 10 FROM t");
    const std::string key = prepared.query.cacheKey();
    ParameterValues changed = prepared.parameters;
    changed.at(0) = ParameterValue::integer(15);
    changed.at(1) = ParameterValue::integer(25);
    const auto bound = bindLegacyPlan(prepared.query, changed);
    const auto& selected = projection(*bound).select_exprs;
    require(static_cast<const LiteralIntExpr&>(*static_cast<const BinaryExpr&>(*selected[0]).right).value == 15,
            "Binding must use the new first value");
    require(static_cast<const LiteralIntExpr&>(*static_cast<const BinaryExpr&>(*selected[1]).right).value == 25,
            "Equal initial values must bind independently");
    require(prepared.query.cacheKey() == key && prepared.parameters.at(0).asInt64() == 10,
            "Binding cannot mutate a template or another execution's values");
    rejects([&] { bindLegacyPlan(prepared.query, {}); }, "count");
    changed.at(0) = ParameterValue::floating(1.5);
    rejects([&] { bindLegacyPlan(prepared.query, changed); }, "type");
    changed.at(0) = ParameterValue::null();
    const auto normalized = bindParameterValues(prepared.query.parameter_schema, changed);
    require(normalized.at(0).isNull() && normalized.at(0).type() == ParameterType::Int64,
            "Untyped NULL rebind must retain an existing typed slot");
    const auto null_bound = bindLegacyPlan(prepared.query, changed);
    require(static_cast<const BinaryExpr&>(*projection(*null_bound).select_exprs[0]).right->getType() == ExprType::LITERAL_NULL,
            "Legacy binding must preserve SQL NULL");
    rejects([] { ParameterValue::null().asInt64(); }, "type");
    rejects([] { ParameterValue::null(ParameterType::Int64).asInt64(); }, "NULL");
    rejects([] { ParameterValue::integer(1).asFloat64(); }, "type");
    rejects([] { ParameterValue::null(ParameterType::Float64).asFloat64(); }, "NULL");
    const auto float_query = prepare("SELECT 1.25 FROM t");
    const auto float_bound = bindLegacyPlan(float_query.query, float_query.parameters);
    require(static_cast<const LiteralFloatExpr&>(*projection(*float_bound).select_exprs[0]).value == 1.25,
            "Legacy binding preserves the Float64 alternative");
}

void test_limit_binding_checks() {
    auto prepared = prepare("SELECT a FROM t LIMIT 3");
    ParameterValues values{ParameterValue::integer(0)};
    const auto bound = bindLegacyPlan(prepared.query, values);
    require(static_cast<const SortLimitNode&>(*bound).literalLimit() == 0, "LIMIT 0 must remain valid");
    values.at(0) = ParameterValue::integer(-1);
    rejects([&] { bindLegacyPlan(prepared.query, values); }, "LIMIT");
    values.at(0) = ParameterValue::null();
    rejects([&] { bindLegacyPlan(prepared.query, values); }, "LIMIT");
    rejects([] { prepare("SELECT a FROM t LIMIT 3 OFFSET 1"); }, "OFFSET");
}

void test_column_resolution_checks() {
    const auto qualified = prepare("SELECT x.a FROM t x WHERE x.b > 10");
    const auto unqualified = prepare("SELECT a FROM t WHERE b > 20");
    require(qualified.query.cacheKey() == unqualified.query.cacheKey(), "Table aliases do not change resolved structure");
    rejects([] { prepare("SELECT missing FROM t"); }, "column");
    rejects([] { prepare("SELECT a FROM t, u"); }, "Ambiguous");
    rejects([] { prepare("SELECT t.a FROM t, u GROUP BY t.a, u.a HAVING a > 1"); }, "Ambiguous");
    rejects([] { prepare("SELECT b FROM t GROUP BY a"); }, "GROUP BY");
    rejects([] { prepare("SELECT z.a FROM t"); }, "qualifier");
    rejects([] { prepare("SELECT x.a FROM t x, u x"); }, "Duplicate table qualifier");
    rejects([] { prepare("SELECT a FROM absent"); }, "Table");
    rejects([] { prepare("SELECT x.a FROM t x, t y"); }, "Self");
}

void test_no_value_dependent_folding() {
    const auto zero = prepare("SELECT a * 0 FROM t WHERE 1 = 1");
    const auto other = prepare("SELECT a * 7 FROM t WHERE 1 = 2");
    require(zero.query.cacheKey() == other.query.cacheKey(), "Optimization must not specialize ordinary parameter values");
    require(zero.parameters.size() == 3, "Even equal literal-only predicates keep independent slots");
    require(projection(*zero.query.plan).select_exprs.at(0)->getType() == ExprType::OP_MUL,
            "Multiplication must survive a first zero parameter");
}

void test_key_boundaries_and_validation() {
    const auto base = prepare("SELECT SUM(a) FROM t");
    require(base.query.structureKey() != prepare("SELECT MAX(a) FROM t").query.structureKey(), "Aggregate function is structural");
    require(prepare("SELECT a AS x FROM t").query.structureKey() != prepare("SELECT a AS y FROM t").query.structureKey(),
            "Visible aliases must not collide in the plan key");
    require(prepare("SELECT a FROM t").query.structureKey() != prepare("SELECT b FROM t").query.structureKey(),
            "Selected column is structural");
    require(prepare("SELECT SUM(a), b FROM t GROUP BY b").query.structureKey() !=
            prepare("SELECT b, SUM(a) FROM t GROUP BY b").query.structureKey(),
            "SELECT output-reference order must remain structural even with legacy group-first storage");
    const auto typed_null = bindParameterValues({ParameterType::Float64}, {ParameterValue::null()});
    require(typed_null.at(0).type() == ParameterType::Float64 && typed_null.at(0).isNull(), "Typed Float64 NULL rebind");
    rejects([] { bindParameterValues({ParameterType::NullOnly}, {ParameterValue::integer(1)}); }, "type");
    QueryTemplate invalid;
    rejects([&] { invalid.structureKey(); }, "plan");
    rejects([&] { bindLegacyPlan(invalid, {}); }, "plan");
    rejects([] { parameterizeQuery(nullptr); }, "plan");
    invalid.plan = std::make_unique<ProjectionNode>();
    static_cast<ProjectionNode&>(*invalid.plan).select_exprs.push_back(std::make_unique<LiteralIntExpr>(10));
    rejects([&] { invalid.structureKey(); }, "literal");
    static_cast<ProjectionNode&>(*invalid.plan).select_exprs[0] = std::make_unique<ParameterRefExpr>(1);
    rejects([&] { invalid.structureKey(); }, "slot");
    rejects([&] { parameterizeQuery(std::move(invalid.plan)); }, "already parameterized");
}

/** @brief Join keys, bindings and shared NULL provenance are all exercised as public behaviours. */
void test_join_and_shared_null_boundaries() {
    const auto first = prepare("SELECT COUNT(*) FROM lineorder, customer "
        "WHERE lo_custkey = c_custkey AND c_region = 1");
    const auto changed = prepare("SELECT COUNT(*) FROM lineorder, customer "
        "WHERE lo_custkey = c_custkey AND c_region = 2");
    require(first.query.cacheKey() == changed.query.cacheKey(), "Join filtering values must not change structure");
    const auto bound = bindLegacyPlan(changed.query, changed.parameters);
    require(aggregate(*bound).aggregates[0].hasStarArgument(), "COUNT(*) remains a structural row count");
    const auto shared_null = prepare("SELECT CASE (NULL + 1.5) WHEN 1 THEN 10 WHEN 2 THEN 20 END FROM t");
    require(shared_null.parameters.size() == 7, "Simple CASE shares each source literal, including NULL");
    std::size_t nulls = 0;
    for (const auto& value : shared_null.parameters) {
        if (value.isNull() && value.type() == ParameterType::Float64) ++nulls;
    }
    require(nulls == 1, "The shared base NULL must keep its single Float64 schema slot");
    const auto having_case = prepare("SELECT SUM(a) AS total FROM t GROUP BY b "
        "HAVING CASE WHEN total > 1 THEN 1 ELSE NULL END = 1");
    require(having_case.parameters.size() == 4, "HAVING CASE ordinary values remain independent");
    (void)having_case.query.cacheKey();
    (void)bindLegacyPlan(having_case.query, having_case.parameters);
}

/** @brief Rebuilt expressions with different slot topology cannot share a cache entry. */
void test_reordered_parameter_identity_is_part_of_key() {
    auto make = [](std::size_t left, std::size_t right) {
        QueryTemplate result;
        result.parameter_schema = {ParameterType::Int64, ParameterType::Int64};
        auto selected = std::make_unique<ProjectionNode>();
        selected->select_exprs.push_back(std::make_unique<BinaryExpr>(ExprType::OP_SUB,
            std::make_unique<ParameterRefExpr>(left), std::make_unique<ParameterRefExpr>(right)));
        selected->addChild(std::make_unique<TableScanNode>("T"));
        result.plan = std::move(selected);
        return result;
    };
    const auto forward = make(0, 1);
    const auto reverse = make(1, 0);
    require(forward.cacheKey() != reverse.cacheKey(), "Parameter identity must be encoded in the template key");
    ParameterValues values{ParameterValue::integer(5), ParameterValue::integer(9)};
    const auto bound = bindLegacyPlan(reverse, values);
    const auto& expression = static_cast<const BinaryExpr&>(*projection(*bound).select_exprs[0]);
    require(static_cast<const LiteralIntExpr&>(*expression.left).value == 9 &&
            static_cast<const LiteralIntExpr&>(*expression.right).value == 5,
            "A rebuilt expression must retain its original slot-to-value bindings");
}

using NullableInt = std::optional<int64_t>;
using FixtureRow = std::map<std::string, NullableInt>;

/** @brief Independent small-input interpreter for testing frontend bindings, not GPU helpers. */
NullableInt evaluateFixtureExpr(const ExprNode& expr, const FixtureRow& row) {
    if (const auto* column = dynamic_cast<const ColumnRefExpr*>(&expr)) return row.at(column->column_name);
    if (const auto* literal = dynamic_cast<const LiteralIntExpr*>(&expr)) return literal->value;
    if (expr.getType() == ExprType::LITERAL_NULL) return {};
    if (const auto* branch = dynamic_cast<const CaseWhenExpr*>(&expr)) {
        const auto condition = evaluateFixtureExpr(*branch->condition, row);
        return evaluateFixtureExpr(condition && *condition ? *branch->then_expr : *branch->else_expr, row);
    }
    const auto& binary = dynamic_cast<const BinaryExpr&>(expr);
    const auto left = evaluateFixtureExpr(*binary.left, row);
    if (expr.getType() == ExprType::OP_IS_NULL) return !left;
    if (expr.getType() == ExprType::OP_IS_NOT_NULL) return left.has_value();
    if (expr.getType() == ExprType::OP_NOT) return left ? NullableInt(!*left) : NullableInt{};
    const auto right = evaluateFixtureExpr(*binary.right, row);
    if (expr.getType() == ExprType::OP_AND) {
        if ((left && !*left) || (right && !*right)) return 0;
        return left && right ? NullableInt(1) : NullableInt{};
    }
    if (expr.getType() == ExprType::OP_OR) {
        if ((left && *left) || (right && *right)) return 1;
        return left && right ? NullableInt(0) : NullableInt{};
    }
    if (!left || !right) return {};
    switch (expr.getType()) {
        case ExprType::OP_EQ: return *left == *right;
        case ExprType::OP_GT: return *left > *right;
        case ExprType::OP_ADD: return *left + *right;
        default: throw std::runtime_error("Expression is outside the nullable fixture interpreter");
    }
}

using FixtureResults = std::vector<std::vector<std::optional<long double>>>;

/** @brief Execute the five nullable fixture forms with an independent CPU algorithm. */
FixtureResults executeFixture(const OperatorNode& plan, const std::vector<FixtureRow>& input) {
    const auto& node = aggregate(plan);
    const FilterNode* filter = nullptr;
    const auto* child = node.getChildren()[0].get();
    if (child->getType() == OperatorType::FILTER) filter = static_cast<const FilterNode*>(child);
    std::map<std::vector<NullableInt>, std::vector<const FixtureRow*>> groups;
    if (node.group_by_exprs.empty()) groups[{}] = {};
    for (const auto& row : input) {
        if (filter) {
            const auto pass = evaluateFixtureExpr(*filter->predicate, row);
            if (!pass || !*pass) continue;
        }
        std::vector<NullableInt> key;
        for (const auto& expr : node.group_by_exprs) key.push_back(evaluateFixtureExpr(*expr, row));
        groups[key].push_back(&row);
    }
    FixtureResults result;
    for (const auto& group : groups) {
        std::vector<std::optional<long double>> output;
        for (const auto value : group.first) output.push_back(value ? std::optional<long double>(*value) : std::nullopt);
        for (const auto& definition : node.aggregates) {
            int64_t sum = 0;
            std::size_t count = 0;
            for (const auto* row : group.second) {
                const auto value = definition.hasStarArgument() ? NullableInt(1) : evaluateFixtureExpr(*definition.agg_expr, *row);
                if (value) { sum += *value; ++count; }
            }
            if (definition.isCount()) output.push_back(count);
            else if (count == 0) output.push_back(std::nullopt);
            else if (definition.isAvg()) output.push_back(static_cast<long double>(sum) / count);
            else if (definition.isSum()) output.push_back(sum);
            else throw std::runtime_error("Aggregate is outside the nullable fixture interpreter");
        }
        result.push_back(std::move(output));
    }
    return result;
}

std::vector<std::string> fields(const std::string& line, char delimiter) {
    std::vector<std::string> result;
    std::istringstream input(line + delimiter);
    std::string field;
    while (std::getline(input, field, delimiter)) result.push_back(field);
    return result;
}

/** @brief Compare frontend-bound nullable queries to the independently maintained TSV oracle. */
void test_nullable_dataset_frontend_bindings() {
    const std::string directory = std::string(CRYSTAL_TEST_SOURCE_DIR) + "/test/nullable_data/";
    std::ifstream csv(directory + "nullable_numbers.csv");
    require(csv.good(), "Cannot open nullable fixture");
    std::string line;
    std::getline(csv, line);
    const auto names = fields(line, ',');
    std::vector<FixtureRow> rows;
    while (std::getline(csv, line)) {
        if (line.empty()) continue;
        const auto values = fields(line, ',');
        require(values.size() == names.size(), "Nullable CSV width mismatch");
        FixtureRow row;
        for (std::size_t i = 0; i < names.size(); ++i) {
            row[names[i]] = values[i].empty() ? NullableInt{} : NullableInt(std::stoll(values[i]));
        }
        rows.push_back(std::move(row));
    }
    require(rows.size() == 12, "Nullable fixture row count");
    Catalog metadata;
    TableMetadata table("NULLABLE_NUMBERS", names, rows.size(), false);
    table.setColumnNullable("val");
    table.setColumnNullable("flag");
    metadata.pushTableMetadata(table);
    std::map<std::string, FixtureResults> expected;
    std::ifstream tsv(directory + "expected.tsv");
    require(tsv.good(), "Cannot open nullable expectations");
    while (std::getline(tsv, line)) {
        const auto cells = fields(line, '\t');
        if (cells.size() < 2 || cells[1].empty() || !std::isdigit(static_cast<unsigned char>(cells[1][0]))) continue;
        std::vector<std::optional<long double>> output;
        for (std::size_t i = 1; i < cells.size(); ++i) output.push_back(std::stold(cells[i]));
        expected[cells[0]].push_back(std::move(output));
    }
    std::ifstream sql(directory + "queries.sql");
    require(sql.good(), "Cannot open nullable SQL");
    std::string queries;
    while (std::getline(sql, line)) if (line.rfind("--", 0) != 0) queries += line + '\n';
    std::size_t index = 0;
    for (const auto& query : fields(queries, ';')) {
        if (query.find_first_not_of(" \t\n\r") == std::string::npos) continue;
        auto prepared = prepare(query, metadata);
        auto bound = bindLegacyPlan(prepared.query, prepared.parameters);
        const auto actual = executeFixture(*bound, rows);
        const auto& oracle = expected.at("q" + std::to_string(++index));
        require(actual.size() == oracle.size(), "Nullable fixture group count");
        for (std::size_t r = 0; r < actual.size(); ++r) {
            require(actual[r].size() == oracle[r].size(), "Nullable fixture result width");
            for (std::size_t c = 0; c < actual[r].size(); ++c) {
                require(actual[r][c].has_value() == oracle[r][c].has_value(), "Nullable fixture validity");
                if (actual[r][c]) require(std::abs(*actual[r][c] - *oracle[r][c]) < 1e-12L,
                    "Nullable fixture q" + std::to_string(index) + " row " + std::to_string(r) + " column " +
                    std::to_string(c) + ": expected " + std::to_string(*oracle[r][c]) + ", actual " + std::to_string(*actual[r][c]));
            }
        }
    }
    require(index == 5, "All five nullable queries must be checked");
}
} // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"template identity", test_template_identity},
        {"all value locations", test_all_value_locations},
        {"independent and cloned slots", test_independent_and_cloned_slots},
        {"resolved expression identity", test_reference_identity_before_values},
        {"structural ORDER BY", test_order_positions_are_structural},
        {"types and NULL", test_parameter_types_and_nulls},
        {"pure checked legacy binding", test_legacy_binding_is_pure_and_checked},
        {"LIMIT validation", test_limit_binding_checks},
        {"column resolution", test_column_resolution_checks},
        {"no value-dependent folding", test_no_value_dependent_folding},
        {"key boundaries", test_key_boundaries_and_validation},
        {"join and shared NULL boundaries", test_join_and_shared_null_boundaries},
        {"reordered parameter identity in key", test_reordered_parameter_identity_is_part_of_key},
        {"nullable CSV/TSV frontend bindings", test_nullable_dataset_frontend_bindings},
    };
    try {
        for (const auto& test : tests) {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << tests.size() << " frontend behavioural tests passed\n";
}
