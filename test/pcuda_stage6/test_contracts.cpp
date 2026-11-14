/** @file @brief Host contracts for numeric layouts, signed literal preparation and checked capacities. */
#include "core/aggregate_codegen.h"
#include "core/aggregate_numeric.h"
#include "core/translator.h"
#include "core/visitor.h"
#include "core/optimizer_rules.h"
#include "core/inline_math.h"
#include <cassert>
#include <iostream>
#include <limits>

using namespace db;

template<class Action> static void rejects(Action action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

static void capacityContracts() {
    Catalog catalog;
    catalog.emplaceTableMetadata("LINEORDER", std::vector<std::string>{"lo_quantity", "lo_revenue"}, UINT64_MAX, true);
    ProjectionNode projection;
    projection.select_exprs.push_back(std::make_unique<StarExpr>());
    projection.addChild(std::make_unique<TableScanNode>("LINEORDER"));
    rejects([&] { projection.calculateResultSize(catalog); });
    AggregateNode aggregate;
    aggregate.group_by_exprs.push_back(std::make_unique<ColumnRefExpr>("lo_quantity"));
    aggregate.aggregates.emplace_back("COUNT", std::make_unique<StarExpr>());
    rejects([&] { aggregate.calculateResultSize(catalog); });
}

static PreparedQuery prepare(const std::string& sql, const Catalog& catalog) {
    std::cout << sql << std::endl;
    hsql::SQLParserResult parsed;
    hsql::SQLParser::parse(sql, &parsed);
    assert(parsed.isValid());
    return QueryTranslator{}.prepare(static_cast<const hsql::SelectStatement*>(parsed.getStatement(0)), catalog);
}

static void negativeLiterals() {
    Catalog catalog;
    catalog.emplaceTableMetadata("LINEORDER", std::vector<std::string>{"lo_quantity"}, 9, true);
    const auto a = prepare("SELECT 0.5,10 FROM lineorder WHERE lo_quantity > 1", catalog);
    const auto b = prepare("SELECT -0.5,-10 FROM lineorder WHERE lo_quantity > -1", catalog);
    assert(a.query.structureKey() == b.query.structureKey());
    assert(a.query.schemaKey() == b.query.schemaKey());
    assert(b.parameters[0].asFloat64() == -0.5 && b.parameters[1].asInt64() == -10 && b.parameters[2].asInt64() == -1);
    const auto having = prepare("SELECT SUM(lo_quantity) FROM lineorder GROUP BY lo_quantity HAVING SUM(lo_quantity) > -0.5", catalog);
    assert(having.parameters[0].asFloat64() == -0.5);
    rejects([&] { prepare("SELECT -lo_quantity FROM lineorder", catalog); });
    rejects([&] { prepare("SELECT SUM(lo_quantity) FROM lineorder GROUP BY lo_quantity HAVING -SUM(lo_quantity)>1", catalog); });
}

template<class T, AggregateOp op> static void numericContract(T a, T b, T expected) {
    assert(aggregateValue<T>(aggregateBits(a)) == a);
    assert((AggregateMerge<T, op>{}(a, b) == expected));
    assert((AggregateMerge<T, op>{}(a, aggregateIdentity<T, op>()) == a));
}

static void numericStates() {
    assert(safe_gt(uint64_t{0}, int64_t{-1}) == 1);
    assert(safe_lt(safe_add(uint64_t{0}, int64_t{-1}), int64_t{0}) == 1);
    assert(safe_eq(safe_add(UINT64_MAX, INT64_MIN), INT64_MAX) == 1);
    assert(safe_gt(UINT64_MAX, INT64_MAX) == 1);
    assert(safe_eq(UINT64_MAX, INT64_MAX) == 0);
    assert(safe_neq(UINT64_MAX, INT64_MAX) == 1);
    assert(safe_lte(UINT64_MAX, INT64_MAX) == 0);
    assert(safe_gte(UINT64_MAX, INT64_MAX) == 1);
    assert(safe_lt(INT64_MIN, UINT64_MAX) == 1);
    assert(safe_eq(safe_mul(uint64_t{3}, int64_t{-2}), int64_t{-6}) == 1);
    assert(safe_eq(safe_sub(uint64_t{0}, int64_t{3}), int64_t{-3}) == 1);
    assert(safe_eq(safe_sub(uint64_t{3}, int64_t{0}), int64_t{3}) == 1);
    assert(safe_eq(safe_div(uint64_t{7}, int64_t{-2}), int64_t{-3}) == 1);
    assert(safe_eq(safe_mul(uint64_t{0}, int64_t{-2}), int64_t{0}) == 1);
    assert(safe_eq(safe_mul(uint64_t{1} << 63, int64_t{-1}), INT64_MIN) == 1);
    assert(safe_eq(safe_add(UINT64_C(9007199254740993), int64_t{-1}), UINT64_C(9007199254740992)) == 1);
    assert(safe_eq(safe_add(uint64_t{1}, -0.5), 0.5) == 1);
    assert(safe_eq(safe_add(safe_add(uint64_t{0}, int64_t{-1}), 0.5), -0.5) == 1);
    numericContract<int64_t, AggregateOp::Sum>(-4, 1, -3);
    numericContract<int64_t, AggregateOp::Minimum>(-4, 1, -4);
    numericContract<int64_t, AggregateOp::Maximum>(-4, -1, -1);
    numericContract<double, AggregateOp::Sum>(-0.5, 1.25, 0.75);
    numericContract<double, AggregateOp::Minimum>(-0.5, 1.25, -0.5);
    numericContract<double, AggregateOp::Maximum>(-0.5, -1.25, -0.5);
    numericContract<unsigned long long, AggregateOp::Sum>(1, 2, 3);
    numericContract<unsigned long long, AggregateOp::Minimum>(1, 2, 1);
    numericContract<unsigned long long, AggregateOp::Maximum>(1, 2, 2);
    assert(aggregateValue<int64_t>(aggregateBits(INT64_MIN)) == INT64_MIN);
    assert(aggregateValue<int64_t>(aggregateBits(INT64_MAX)) == INT64_MAX);
    assert(aggregateValue<int64_t>(aggregateBits(INT64_C(9007199254740993))) == INT64_C(9007199254740993));
}

/** @brief Independent small-number arithmetic exercises every tag pair; large integers are checked above without double. */
static void taggedPairs() {
    const TaggedNumeric inputs[] = {TaggedNumeric(int64_t{-3}), TaggedNumeric(uint64_t{3}), TaggedNumeric(0.5)};
    for (const auto a : inputs) for (const auto b : inputs) {
        const double x = static_cast<double>(a), y = static_cast<double>(b);
        assert((a == b) == (x == y));
        assert((a != b) == (x != y));
        assert((a < b) == (x < y));
        assert((a > b) == (x > y));
        assert((a <= b) == (x <= y));
        assert((a >= b) == (x >= y));
        const TaggedNumeric actual[] = {a + b, a - b, a * b, a / b};
        if (a.type == DB_TYPE_FLOAT64 || b.type == DB_TYPE_FLOAT64) {
            const double expected[] = {x + y, x - y, x * y, x / y};
            for (size_t i = 0; i < 4; ++i) assert(actual[i].type == DB_TYPE_FLOAT64 && static_cast<double>(actual[i]) == expected[i]);
        } else {
            const auto left = static_cast<int64_t>(x), right = static_cast<int64_t>(y);
            const int64_t expected[] = {left + right, left - right, left * right, left / right};
            for (size_t i = 0; i < 4; ++i) {
                assert(actual[i].type != DB_TYPE_FLOAT64 && static_cast<int64_t>(actual[i]) == expected[i]);
                if (expected[i] < 0) assert(actual[i].type == DB_TYPE_INT64);
            }
        }
    }
}

static void layoutContracts() {
    const ParameterSchema schema{ParameterType::Int64, ParameterType::Float64, ParameterType::NullOnly};
    for (const auto& name : {"COUNT", "SUM", "MIN", "MAX", "AVG"}) {
        AggregateNode node;
        for (size_t slot = 0; slot < schema.size(); ++slot) node.aggregates.emplace_back(name, std::make_unique<ParameterRefExpr>(slot));
        std::vector<ExpressionCode> inputs{{"db_parameter_0", "db_parameter_0_valid"}, {"db_parameter_1", "db_parameter_1_valid"}, {"db_parameter_2", "db_parameter_2_valid"}};
        assert(!emitScalarAggregate(node, schema, inputs).empty());
        for (size_t slot = 0; slot < schema.size(); ++slot) {
            const auto& aggregate = node.aggregates[slot];
            const auto state = name == std::string("COUNT") ? LogicalType::UInt64 : slot == 1 ? LogicalType::Float64 : LogicalType::Int64;
            const auto output = name == std::string("AVG") ? LogicalType::Float64 : state;
            assert(aggregateStateType(aggregate, schema) == state);
            assert(aggregateResultType(aggregate, schema) == output);
            assert(aggregateSlotType(node, slot, schema) == output);
            const AggregateCode code(aggregate, schema);
            assert(!code.identity().empty() && !code.merge("a", "b").empty() && !code.atomic("target", "value").empty());
            assert(!code.reduce("local").empty() && !code.read("raw").empty());
        }
        rejects([&] { emitScalarAggregate(node, schema, {}); });
    }
    AggregateNode groups;
    groups.group_by_exprs.push_back(std::make_unique<ColumnRefExpr>("lo_quantity"));
    groups.group_by_exprs.push_back(std::make_unique<CaseWhenExpr>(std::make_unique<LiteralIntExpr>(1), std::make_unique<ParameterRefExpr>(0), std::make_unique<ParameterRefExpr>(2)));
    assert(aggregateSlotType(groups, 0, schema) == LogicalType::Int64);
    assert(aggregateSlotType(groups, 1, schema) == LogicalType::Int64);
    assert(expressionResultType(nullptr) == LogicalType::UInt64);
    for (const auto op : {ExprType::OP_ADD, ExprType::OP_EQ, ExprType::OP_AND, ExprType::OP_IS_NULL, ExprType::OP_IS_NOT_NULL}) {
        BinaryExpr expr(op, std::make_unique<ParameterRefExpr>(1), std::make_unique<ParameterRefExpr>(0));
        assert(expressionResultType(&expr, schema) == (op == ExprType::OP_ADD ? LogicalType::Float64 : LogicalType::UInt64));
    }
    LiteralNullExpr null;
    assert(expressionResultType(&null) == LogicalType::UInt64);
}

static void expressionTypes() {
    const ParameterSchema schema{ParameterType::Int64, ParameterType::Float64, ParameterType::NullOnly};
    const ColumnExpression column = [](const ColumnRefExpr& ref) { return ExpressionCode{ref.column_name, "column_valid"}; };
    CaseWhenExpr branch(std::make_unique<LiteralIntExpr>(1), std::make_unique<ParameterRefExpr>(0), std::make_unique<ParameterRefExpr>(1));
    assert(emitSqlExpression(&branch, column, schema).value.find("static_cast<double>") != std::string::npos);
    branch.else_expr = std::make_unique<ParameterRefExpr>(2);
    assert(caseLeafCount(&branch) == 2);
    assert(!emitCaseGroupCode(&branch, column, schema).empty());
    assert(!emitCaseGroupCode(&branch, column).empty());
    rejects([&] { emitCaseGroupCode(&branch, column, {ParameterType::Float64, ParameterType::Float64, ParameterType::NullOnly}); });
    assert(emitSqlExpression(&branch, column, schema).value.find("static_cast<std::int64_t>") != std::string::npos);
    assert(!emitSqlExpression(&branch, column).value.empty());
    branch.then_expr = std::make_unique<ColumnRefExpr>("output");
    assert(sqlExpressionType(&branch, schema, [](const ColumnRefExpr&) { return ParameterType::Float64; }) == ParameterType::Float64);
    for (const auto op : {ExprType::OP_ADD, ExprType::OP_SUB, ExprType::OP_MUL, ExprType::OP_DIV, ExprType::OP_EQ, ExprType::OP_NEQ,
                          ExprType::OP_LT, ExprType::OP_LTE, ExprType::OP_GT, ExprType::OP_GTE, ExprType::OP_AND, ExprType::OP_OR}) {
        BinaryExpr expr(op, branch.clone(), std::make_unique<ParameterRefExpr>(0));
        assert(!emitSqlExpression(&expr, column, schema).value.empty());
        (void)sqlExpressionType(&expr, schema);
    }
    for (const auto op : {ExprType::OP_NOT, ExprType::OP_IS_NULL, ExprType::OP_IS_NOT_NULL}) {
        BinaryExpr expr(op, std::make_unique<LiteralNullExpr>(), nullptr);
        assert(!emitSqlExpression(&expr, column, schema).value.empty());
    }
    rejects([&] { emitCaseGroupCode(&branch, column, schema); });
}

/** @brief Exercise host lowering itself; hardware tests separately check the emitted kernels' results. */
static void physicalContracts() {
    Catalog catalog;
    TableMetadata fact("LINEORDER", {"lo_quantity", "lo_revenue"}, 9001, true);
    fact.setColumnStats("lo_quantity", {0, 256, 257});
    catalog.pushTableMetadata(fact);
    const auto lower = [&](const std::string& sql) {
        auto prepared = prepare(sql, catalog);
        prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
        JITContext context;
        context.parameter_schema = prepared.query.parameter_schema;
        JITOperatorVisitor visitor(context, catalog);
        prepared.query.plan->accept(visitor);
        return visitor.buildPhysicalPlan();
    };
    const auto sql = [](const std::string& a, const std::string& b, const std::string& limit) {
        const auto group = "CASE WHEN lo_quantity<4 THEN " + a + " ELSE " + b + " END";
        return "SELECT " + group + ",SUM(lo_revenue+0.5) FROM lineorder GROUP BY " + group + " HAVING SUM(lo_revenue+0.5)>0 ORDER BY 1 LIMIT " + limit;
    };
    const auto first = lower(sql("-9007199254740993", "9007199254740993", "0"));
    const auto changed = lower(sql("7", "7", "99999"));
    assert(first.bodies.size() == changed.bodies.size() && first.bodies.size() >= 2);
    for (size_t i = 0; i < first.bodies.size(); ++i) assert(first.bodies[i].code == changed.bodies[i].code);
    bool presence = false;
    for (const auto& buffer : first.plan.buffers) presence |= buffer.name == "d_aggregate_presence";
    assert(presence);
    const auto nullable = lower("SELECT CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END,COUNT(NULL) FROM lineorder GROUP BY CASE WHEN lo_quantity<4 THEN NULL ELSE NULL END HAVING COUNT(NULL)=0 ORDER BY 1");
    assert(!nullable.plan.kernels.empty());
    const auto dense = lower("SELECT lo_quantity,SUM(lo_revenue+0.5),MIN(lo_revenue+0.5),MAX(lo_revenue+0.5),AVG(lo_revenue+0.5) FROM lineorder GROUP BY lo_quantity HAVING lo_quantity<1 ORDER BY 1 LIMIT 1");
    size_t columns = 0;
    for (const auto& buffer : dense.plan.buffers) {
        if (buffer.role != BufferRole::ResultColumn) continue;
        assert(buffer.type == (buffer.column ? DB_TYPE_FLOAT64 : DB_TYPE_INT64));
        ++columns;
    }
    assert(columns == 5 && dense.bodies.size() > first.bodies.size());
    const auto count_only = lower("SELECT lo_quantity,COUNT(*) FROM lineorder GROUP BY lo_quantity ORDER BY 1 LIMIT 0");
    assert(!count_only.plan.kernels.empty());
}

int main() try {
    capacityContracts();
    negativeLiterals();
    numericStates();
    taggedPairs();
    layoutContracts();
    expressionTypes();
    physicalContracts();
    std::cout << "Runtime numeric/capacity host contracts passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
