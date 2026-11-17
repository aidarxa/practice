// test/test_optimizer.cpp
//
// Unit-тесты нового конвейера:
//   PredicatePushdownRule (optimizer_rules.h)
//   QueryTranslator       (translator.h)
//   Optimizer             (optimizer_rules.h)
//   JITOperatorVisitor    (visitor.h)
//   AggregateNode::calculateResultSize (operators.h)

#include "core/optimizer_rules.h"
#include "core/translator.h"
#include "core/visitor.h"
#include "core/query_template.h"
#include "core/module_codegen.h"
#include "core/text_plan.h"
#include "assertions_enabled.h"

#include <SQLParser.h>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace db;

// ============================================================================
// Вспомогательные функции
// ============================================================================

// Строит тестовый каталог SSB
static std::shared_ptr<Catalog> buildTestCatalog() {
    auto catalog = std::make_shared<Catalog>();

    // LINEORDER (fact)
    TableMetadata lo("LINEORDER",
                     {"lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey",
                      "lo_suppkey", "lo_orderdate", "lo_orderpriority",
                      "lo_shippriority", "lo_quantity", "lo_extendedprice",
                      "lo_ordtotalprice", "lo_discount", "lo_revenue",
                      "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"},
                     59986214, true);
    catalog->pushTableMetadata(lo);

    // SUPPLIER (dimension)
    TableMetadata s("SUPPLIER",
                    {"s_suppkey", "s_name", "s_address", "s_city", "s_nation",
                     "s_region", "s_phone"},
                    20000, false);
    s.setColumnStats("s_suppkey", {1, 20000, 20000});
    s.setColumnPrimaryKey("s_suppkey");
    s.setColumnStats("s_city",   {0, 249,   250});
    s.setColumnStats("s_region", {0, 4,     5});
    s.setColumnStats("s_nation", {0, 24,    25});
    catalog->pushTableMetadata(s);

    // CUSTOMER (dimension)
    TableMetadata c("CUSTOMER",
                    {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                     "c_region", "c_phone", "c_mktsegment"},
                    300000, false);
    c.setColumnStats("c_custkey", {1, 300000, 300000});
    c.setColumnPrimaryKey("c_custkey");
    c.setColumnStats("c_city",   {0, 249,    250});
    c.setColumnStats("c_region", {0, 4,      5});
    c.setColumnStats("c_nation", {0, 24,     25});
    catalog->pushTableMetadata(c);

    // PART (dimension)
    TableMetadata p("PART",
                    {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                     "p_color", "p_type", "p_size", "p_container"},
                    800000, false);
    p.setColumnStats("p_partkey",  {1, 800000, 800000});
    p.setColumnPrimaryKey("p_partkey");
    p.setColumnStats("p_category", {1, 25,     25});
    p.setColumnStats("p_brand1",   {1, 1000,   1000});
    catalog->pushTableMetadata(p);

    // DDATE (dimension)
    TableMetadata d("DDATE",
                    {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year",
                     "d_yearmonthnum", "d_yearmonth", "d_daynuminweek",
                     "d_daynuminmonth", "d_daynuminyear", "d_sellingseason",
                     "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl",
                     "d_weekdayfl"},
                    2556, false);
    d.setColumnStats("d_datekey", {19920101, 19981230, 2556});
    d.setColumnPrimaryKey("d_datekey");
    d.setColumnStats("d_year",    {1992, 1998, 7});
    catalog->pushTableMetadata(d);

    return catalog;
}

// Парсит SQL и возвращает SelectStatement. Бросает исключение при ошибке.
static hsql::SelectStatement* parseSQL(const std::string& sql,
                                        hsql::SQLParserResult& result) {
    hsql::SQLParser::parse(sql, &result);
    if (!result.isValid() || result.size() == 0) return nullptr;
    return const_cast<hsql::SelectStatement*>(
        static_cast<const hsql::SelectStatement*>(result.getStatement(0)));
}

static PlannedQuery physicalPlanForSql(const std::string& sql,
                                         const std::shared_ptr<Catalog>& catalog) {
    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto prepared = translator.prepare(ast, *catalog);
    db::Optimizer opt;
    prepared.query.plan = opt.optimize(std::move(prepared.query.plan));
    auto tree = bindLegacyPlan(prepared.query, prepared.parameters);

    db::JITContext jit_ctx;
    db::JITOperatorVisitor visitor(jit_ctx, *catalog);
    tree->accept(visitor);
    return visitor.buildPhysicalPlan();
}

static std::string generateJitCodeForSql(const std::string& sql, const std::shared_ptr<Catalog>& catalog) {
    return describePcudamodules(physicalPlanForSql(sql, catalog));
}

/** @brief Only specialized bodies become SQL modules; universal prefix work is a prebuilt service. */
static void test_physical_plan_boundaries_and_control_flow() {
    auto base = buildTestCatalog();
    auto metadata = std::make_shared<Catalog>();
    for (auto table : base->getTablesMetadata()) {
        if (table.getName() == "SUPPLIER") {
            table.setColumnUnique("s_suppkey", false);
            table.setColumnStats("s_suppkey", {1, 10, 10});
        }
        metadata->pushTableMetadata(std::move(table));
    }
    const auto query = physicalPlanForSql("SELECT lo_orderkey,s_nation FROM lineorder,supplier WHERE lo_suppkey=s_suppkey ORDER BY 1,2 LIMIT 3", metadata);
    std::set<std::string> kernels;
    for (const auto& kernel : query.plan.kernels) {
        assert(kernels.insert(kernel.name).second);
        std::set<std::string> arguments;
        for (const auto& argument : kernel.arguments) assert(arguments.insert(argument.name).second);
    }
    for (const auto& name : {"MHTBlockScan_s", "MHTBlockSumsScan_s", "MHTSuperSumsScan_s", "MHTAddSuperSums_s", "MHTAddSums_s",
                             "ProjectionTileBlockScan", "ProjectionBlockSumsScan", "ProjectionAddBlockSums"}) assert(!kernels.count(name));
    assert(kernels.count("MHT_PfxSum_s") == 0);
    assert(kernels.count("Scan_LINEORDER_Count") && kernels.count("Scan_LINEORDER_Write"));
    const auto description = describePlan(query.plan);
    size_t prefixes = 0;
    for (const auto& step : query.plan.steps) prefixes += std::holds_alternative<DevicePrefix>(step.operation);
    assert(prefixes == 2); // MHT offsets and exact projection offsets, without host count materialization
    // Native module lowering preserves fusion and the fixed ABI without host-runtime headers.
    const auto legacy = describePcudamodules(query);
    assert(legacy.find("core/execution.h") == std::string::npos);
    assert(legacy.find("pcudaParallelFor") != std::string::npos);
    assert(legacy.find("sycl::") == std::string::npos);
    assert(legacy.find("malloc") == std::string::npos);
    for (const auto* operation : {"ensure", "fill", "copy", "device-prefix", "branch", "loop", "publish"})
        assert(description.find(operation) != std::string::npos);
    const auto old_loop_boundary = physicalPlanForSql("SELECT s_nation,s_region FROM supplier WHERE s_nation>0 ORDER BY s_nation,s_region LIMIT 4096", base);
    bool full_sort = false;
    for (const auto& kernel : old_loop_boundary.plan.kernels) full_sort |= kernel.name == "SortLimitBitonicSort";
    assert(full_sort); // K must not leave the local selector in a non-progressing host loop
    std::cout << "Physical launch boundaries and control flow... PASSED\n";
}

// ============================================================================
// Test 1: PredicatePushdownRule — ручное построение дерева
// ============================================================================
static void test_pushdown_rule_basic() {
    std::cout << "Test 1: PredicatePushdownRule basic... ";

    // Вручную строим:
    //   Filter(p_category=1 AND s_region=1)
    //     HashJoin(lo_partkey=p_partkey)
    //       left:  HashJoin(lo_suppkey=s_suppkey)
    //                left:  TableScan(SUPPLIER)
    //                right: TableScan(LINEORDER)
    //       right: TableScan(PART)

    auto lo_scan = std::make_unique<db::TableScanNode>("LINEORDER");
    auto s_scan  = std::make_unique<db::TableScanNode>("SUPPLIER");
    auto p_scan  = std::make_unique<db::TableScanNode>("PART");

    auto join1_cond = std::make_unique<db::BinaryExpr>(
        db::ExprType::OP_EQ,
        std::make_unique<db::ColumnRefExpr>("lo_suppkey", "LINEORDER"),
        std::make_unique<db::ColumnRefExpr>("s_suppkey",  "SUPPLIER"));
    auto join1 = std::make_unique<db::HashJoinNode>(std::move(join1_cond));
    join1->addChild(std::move(s_scan));
    join1->addChild(std::move(lo_scan));

    auto join2_cond = std::make_unique<db::BinaryExpr>(
        db::ExprType::OP_EQ,
        std::make_unique<db::ColumnRefExpr>("lo_partkey", "LINEORDER"),
        std::make_unique<db::ColumnRefExpr>("p_partkey",  "PART"));
    auto join2 = std::make_unique<db::HashJoinNode>(std::move(join2_cond));
    join2->addChild(std::move(p_scan));
    join2->addChild(std::move(join1));

    // Предикат: p_category = 1 AND s_region = 1
    auto pred = std::make_unique<db::BinaryExpr>(
        db::ExprType::OP_AND,
        std::make_unique<db::BinaryExpr>(
            db::ExprType::OP_EQ,
            std::make_unique<db::ColumnRefExpr>("p_category", "PART"),
            std::make_unique<db::LiteralIntExpr>(1)),
        std::make_unique<db::BinaryExpr>(
            db::ExprType::OP_EQ,
            std::make_unique<db::ColumnRefExpr>("s_region", "SUPPLIER"),
            std::make_unique<db::LiteralIntExpr>(1)));

    auto filter = std::make_unique<db::FilterNode>(std::move(pred));
    filter->addChild(std::move(join2));

    std::unique_ptr<db::OperatorNode> root = std::move(filter);

    // Применяем правило
    db::PredicatePushdownRule rule;
    rule.apply(root);

    // После pushdown FilterNode удалён (все предикаты спущены)
    assert(root->getType() == db::OperatorType::HASH_JOIN);

    // Левый дочерний → Filter(p_category=1) → TableScan(PART)
    auto* left = root->getChildren()[0].get();
    assert(left->getType() == db::OperatorType::FILTER);

    // Правый → HashJoin, его левый → Filter(s_region=1) → TableScan(SUPPLIER)
    auto* right = root->getChildren()[1].get();
    assert(right->getType() == db::OperatorType::HASH_JOIN);
    auto* right_left = right->getChildren()[0].get();
    assert(right_left->getType() == db::OperatorType::FILTER);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 2: PredicatePushdownRule — через Translator + Optimizer
// ============================================================================
static void test_pushdown_rule_via_translator() {
    std::cout << "Test 2: PredicatePushdownRule via Translator + Optimizer... ";

    std::string sql =
        "SELECT SUM(lo_revenue), d_year, p_brand1 "
        "FROM lineorder, ddate, part, supplier "
        "WHERE lo_orderdate = d_datekey "
        "AND lo_partkey = p_partkey "
        "AND lo_suppkey = s_suppkey "
        "AND p_category = 1 "
        "AND s_region = 1 "
        "GROUP BY d_year, p_brand1";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    assert(tree != nullptr);

    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    // Корень — AggregateNode
    assert(tree->getType() == db::OperatorType::AGGREGATE);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 3: JIT Visitor — Q1.1 (scalar aggregation, no GROUP BY)
// ============================================================================
static void test_jit_visitor_q11() {
    std::cout << "Test 3: JIT Visitor Q1.1 scalar agg... ";

    std::string sql =
        "SELECT SUM(lo_extendedprice * lo_discount) "
        "FROM lineorder "
        "WHERE lo_orderdate >= 19930101 "
        "AND lo_orderdate < 19940101 "
        "AND lo_discount >= 1 "
        "AND lo_discount <= 3 "
        "AND lo_quantity < 25";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    auto catalog = buildTestCatalog();
    db::JITContext ctx;
    db::JITOperatorVisitor visitor(ctx, *catalog);
    tree->accept(visitor);
    std::string code = visitor.generateCode();

    assert(code.find("db_kernel_launch_v2") != std::string::npos);
    assert(code.find("ExecutionContext") == std::string::npos);
    const auto physical = visitor.buildPhysicalPlan();
    assert(std::count_if(physical.plan.kernels.begin(), physical.plan.kernels.end(), [](const auto& kernel) { return kernel.name == "Scan_LINEORDER"; }) == 1);
    assert(code.find(": Scan_LINEORDER") != std::string::npos);
    // Скалярная агрегация → reduce_over_group + atomic_result
    assert(code.find("crystal::blockReduce") != std::string::npos);
    assert(code.find("db::atomicAggregate<std::int64_t,db::AggregateOp::Sum>(d_result[0]") != std::string::npos);
    // Нет build-ядер (только LINEORDER, без измерений)
    assert(code.find("build_hashtable_") == std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 4: AggregateNode::calculateResultSize
// ============================================================================
static void test_calculate_result_size() {
    std::cout << "Test 4: AggregateNode::calculateResultSize... ";

    auto catalog = buildTestCatalog();

    // --- Q2.1: GROUP BY d_year, p_brand1 + SUM(lo_revenue) ---
    // d_year(7) * p_brand1(1000) * tuple_size(3) = 21000
    {
        std::string sql =
            "SELECT SUM(lo_revenue), d_year, p_brand1 "
            "FROM lineorder, ddate, part, supplier "
            "WHERE lo_orderdate = d_datekey "
            "AND lo_partkey = p_partkey "
            "AND lo_suppkey = s_suppkey "
            "AND p_category = 1 "
            "AND s_region = 1 "
            "GROUP BY d_year, p_brand1";

        hsql::SQLParserResult result;
        auto* ast = parseSQL(sql, result);
        assert(ast != nullptr);

        db::QueryTranslator translator;
        auto tree = translator.translate(ast);
        db::Optimizer opt;
        tree = opt.optimize(std::move(tree));

        auto* agg = static_cast<const db::AggregateNode*>(tree.get());
        assert(agg->getType() == db::OperatorType::AGGREGATE);

        uint64_t size = agg->calculateResultSize(*catalog);
        assert(size == 21000 && "Q2.1: d_year(7)*p_brand1(1000)*tuple(3) must equal 21000");
    }

    // --- Q1.1: scalar aggregation → завсегда 1 ---
    {
        std::string sql =
            "SELECT SUM(lo_extendedprice * lo_discount) "
            "FROM lineorder "
            "WHERE lo_orderdate >= 19930101 "
            "AND lo_orderdate < 19940101 "
            "AND lo_discount >= 1 "
            "AND lo_discount <= 3 "
            "AND lo_quantity < 25";

        hsql::SQLParserResult result;
        auto* ast = parseSQL(sql, result);
        assert(ast != nullptr);

        db::QueryTranslator translator;
        auto tree = translator.translate(ast);

        auto* agg = static_cast<const db::AggregateNode*>(tree.get());
        assert(agg->getType() == db::OperatorType::AGGREGATE);

        uint64_t size = agg->calculateResultSize(*catalog);
        assert(size == 1 && "Scalar agg must always return size 1");
    }

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 5: JIT Visitor — Q2.1 (grouped aggregation, 4-table join)
// ============================================================================
static void test_jit_visitor_q21() {
    std::cout << "Test 5: JIT Visitor Q2.1 grouped agg... ";

    std::string sql =
        "SELECT SUM(lo_revenue), d_year, p_brand1 "
        "FROM lineorder, ddate, part, supplier "
        "WHERE lo_orderdate = d_datekey "
        "AND lo_partkey = p_partkey "
        "AND lo_suppkey = s_suppkey "
        "AND p_category = 1 "
        "AND s_region = 1 "
        "GROUP BY d_year, p_brand1";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    auto catalog = buildTestCatalog();
    db::JITContext ctx;
    db::JITOperatorVisitor visitor(ctx, *catalog);
    tree->accept(visitor);
    std::string code = visitor.generateCode();

    assert(code.find("db_kernel_launch_v2") != std::string::npos);
    assert(code.find(": Scan_LINEORDER") != std::string::npos);
    // Build-ядра присутствуют (DDATE, PART, SUPPLIER)
    assert(code.find(": build_hashtable_") != std::string::npos);
    const auto physical = visitor.buildPhysicalPlan();
    assert(physical.plan.kernels.size() == 4);
    size_t scratch = 0;
    for (const auto& buffer : physical.plan.buffers) scratch += buffer.role == BufferRole::Scratch;
    assert(scratch == 6); // three tables and three explicit PHT occupancy maps
    assert(code.find("malloc_device") == std::string::npos);
    // Групповая агрегация → atomic_ref, НЕ reduce_over_group
    assert(code.find("db::atomicAggregate") != std::string::npos);
    assert(code.find("reduce_over_group") == std::string::npos);
    assert(code.find("q.wait") == std::string::npos); // synchronization belongs to runtime

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 6: JIT Visitor — Q3.1 (4-way join, GROUP BY c_nation, s_nation, d_year)
// ============================================================================
static void test_jit_visitor_q31() {
    std::cout << "Test 6: JIT Visitor Q3.1 4-way join... ";

    std::string sql =
        "SELECT c_nation, s_nation, d_year, SUM(lo_revenue) as revenue "
        "FROM customer, lineorder, supplier, ddate "
        "WHERE lo_custkey = c_custkey "
        "AND lo_suppkey = s_suppkey "
        "AND lo_orderdate = d_datekey "
        "AND c_region = 1 "
        "AND s_region = 1 "
        "AND d_year >= 1992 "
        "AND d_year <= 1997 "
        "GROUP BY c_nation, s_nation, d_year";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    // Корень — AggregateNode
    assert(tree->getType() == db::OperatorType::AGGREGATE);
    auto* agg = static_cast<const db::AggregateNode*>(tree.get());

    // c_nation(25) * s_nation(25) * d_year(7) = 4375 groups, tuple_size=4 → 17500
    auto catalog = buildTestCatalog();
    uint64_t size = agg->calculateResultSize(*catalog);
    assert(size == 17500 && "Q3.1: c_nation(25)*s_nation(25)*d_year(7)*4 must equal 17500");

    // JIT-код
    db::JITContext jit_ctx;
    db::JITOperatorVisitor visitor(jit_ctx, *catalog);
    tree->accept(visitor);
    std::string code = visitor.generateCode();

    // Минимум 3 build-ядра (CUSTOMER, SUPPLIER, DDATE)
    size_t builds = 0, pos = 0;
    while ((pos = code.find(": build_hashtable_", pos)) != std::string::npos) { ++builds; pos += 18; }
    assert(builds >= 3 && "Q3.1 must emit at least 3 build kernels");

    assert(code.find("db::atomicAggregate") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 7: mixed predicates should use universal path without debug comments
// ============================================================================
static void test_jit_visitor_mixed_predicate_universal_path() {
    std::cout << "Test 7: JIT Visitor mixed predicate fallback to universal path... ";

    std::string sql =
        "SELECT SUM(lo_revenue) "
        "FROM lineorder "
        "WHERE (lo_discount + 1) < 5 OR lo_quantity = 10";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    auto catalog = buildTestCatalog();
    db::JITContext ctx;
    db::JITOperatorVisitor visitor(ctx, *catalog);
    tree->accept(visitor);
    std::string code = visitor.generateCode();

    assert(code.find("Fallback for complex expression") == std::string::npos);
    assert(code.find("safe_add") != std::string::npos);
    assert(code.find("||") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 8: unsupported predicate should fail before AdaptiveCPP compilation
// ============================================================================
static void test_jit_visitor_unsupported_predicate_throws() {
    std::cout << "Test 8: JIT Visitor unsupported predicate throws early... ";

    std::string sql =
        "SELECT SUM(lo_revenue) "
        "FROM lineorder "
        "WHERE lo_discount = 1.5";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr);

    db::QueryTranslator translator;
    auto tree = translator.translate(ast);
    db::Optimizer opt;
    tree = opt.optimize(std::move(tree));

    auto catalog = buildTestCatalog();
    db::JITContext ctx;
    db::JITOperatorVisitor visitor(ctx, *catalog);

    bool thrown = false;
    try {
        tree->accept(visitor);
    } catch (const std::runtime_error& e) {
        thrown = true;
        const std::string msg = e.what();
        assert(msg.find("not translatable") != std::string::npos);
        assert(msg.find("LITERAL_FLOAT") != std::string::npos);
    }
    assert(thrown && "Expected unsupported predicate to throw before JIT C++ compilation");

    std::cout << "PASSED\n";
}


// ============================================================================
// Test 9: Catalog uniqueness metadata is explicit and available to codegen
// ============================================================================
static void test_catalog_uniqueness_metadata() {
    std::cout << "Test 9: Catalog uniqueness metadata... ";

    auto catalog = buildTestCatalog();

    const auto& d = catalog->getTableMetadata("DDATE");
    const auto& p = catalog->getTableMetadata("PART");
    const auto& s = catalog->getTableMetadata("SUPPLIER");
    const auto& c = catalog->getTableMetadata("CUSTOMER");

    assert(d.isColumnPrimaryKey("d_datekey"));
    assert(p.isColumnPrimaryKey("p_partkey"));
    assert(s.isColumnPrimaryKey("s_suppkey"));
    assert(c.isColumnPrimaryKey("c_custkey"));

    assert(d.isColumnUnique("d_datekey"));
    assert(p.isColumnUnique("p_partkey"));
    assert(s.isColumnUnique("s_suppkey"));
    assert(c.isColumnUnique("c_custkey"));

    const auto& lo = catalog->getTableMetadata("LINEORDER");
    assert(!lo.isColumnUnique("lo_custkey"));
    assert(!lo.isColumnPrimaryKey("lo_custkey"));

    std::cout << "PASSED\n";
}



// ============================================================================
// Test 10: nullable metadata survives stats updates
// ============================================================================
static void test_catalog_nullable_metadata() {
    std::cout << "Test 10: Catalog nullable metadata... ";

    TableMetadata t("T", {"a", "b"}, 10, false);
    t.setColumnNullable("a", true);
    t.setColumnStats("a", {1, 9, 9});
    assert(t.isColumnNullable("a"));

    t.setColumnStats("b", {0, 1, 2, false, false, true});
    assert(t.isColumnNullable("b"));

    t.setColumnPrimaryKey("a");
    assert(t.isColumnPrimaryKey("a"));
    assert(t.isColumnUnique("a"));
    assert(t.isColumnNullable("a"));

    std::cout << "PASSED\n";
}



// ============================================================================
// Test 11: nullable expression codegen and typed columnar result ABI
// ============================================================================
static void test_nullable_expression_codegen_and_typed_result_abi() {
    std::cout << "Test 11: nullable expression codegen + typed columnar ABI... ";

    auto catalog = buildTestCatalog();
    // Mark fact aggregate/projection inputs nullable. This synthetic catalog is
    // enough to validate generated NULL semantics without modifying SSB data.
    TableMetadata lo = catalog->getTableMetadata("LINEORDER");
    lo.setColumnNullable("lo_revenue", true);
    auto nullable_catalog = std::make_shared<Catalog>();
    nullable_catalog->pushTableMetadata(lo);
    for (const auto& table : catalog->getTablesMetadata()) {
        if (table.getName() != "LINEORDER") nullable_catalog->pushTableMetadata(table);
    }

    auto generate = [&](const std::string& sql) {
        hsql::SQLParserResult result;
        auto* ast = parseSQL(sql, result);
        assert(ast != nullptr);
        db::QueryTranslator translator;
        auto tree = translator.translate(ast);
        db::Optimizer opt;
        tree = opt.optimize(std::move(tree));
        db::JITContext jit_ctx;
        db::JITOperatorVisitor visitor(jit_ctx, *nullable_catalog);
        tree->accept(visitor);
        return visitor.buildPhysicalPlan();
    };

    const auto aggregate = generate(
        "SELECT COUNT(*), COUNT(lo_revenue), SUM(lo_revenue), AVG(lo_revenue) "
        "FROM lineorder");
    const auto agg_code = describePcudamodules(aggregate);
    assert(agg_code.find("BlockLoadValidity") != std::string::npos);
    assert(agg_code.find("items_valid") != std::string::npos);
    size_t typed_columns = 0;
    for (const auto& buffer : aggregate.plan.buffers) {
        if (buffer.role != BufferRole::ResultColumn) continue;
        assert(buffer.type == (buffer.column == 3 ? DB_TYPE_FLOAT64 : buffer.column == 2 ? DB_TYPE_INT64 : DB_TYPE_UINT64));
        ++typed_columns;
    }
    assert(typed_columns == 4);

    const std::string is_null_projection = describePcudamodules(generate(
        "SELECT lo_revenue IS NULL FROM lineorder"));
    assert(is_null_projection.find("std::uint64_t* d_result_col_0") != std::string::npos);
    assert(is_null_projection.find("(!(db::bitmap_valid_at(n_lo_revenue,") != std::string::npos);
    assert(is_null_projection.find("d_result[out_row") == std::string::npos);

    const std::string where_is_null = describePcudamodules(generate(
        "SELECT COUNT(*) FROM lineorder WHERE lo_revenue IS NULL"));
    assert(where_is_null.find("flags[i] = flags[i] && 0") == std::string::npos);
    assert(where_is_null.find("!(lo_revenue_valid[i])") != std::string::npos ||
           where_is_null.find("!(items_valid[i])") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 12: ORDER BY/LIMIT projection uses direct radix Top-K codegen
// ============================================================================
static void test_topn_projection_direct_radix_codegen() {
    std::cout << "Test 12: projection ORDER BY/LIMIT direct radix Top-K codegen... ";

    const std::string code = generateJitCodeForSql(
        "SELECT lo_orderkey, lo_revenue "
        "FROM lineorder "
        "ORDER BY lo_revenue DESC "
        "LIMIT 100",
        buildTestCatalog());

    assert(code.find("ProjectionDirectTopKRadixHistogram") != std::string::npos);
    assert(code.find("ProjectionDirectTopKRadixReduce") != std::string::npos);
    assert(code.find("ProjectionDirectTopKCollect") != std::string::npos);
    assert(code.find("ProjectionDirectTopKLocalSelect") == std::string::npos);
    assert(code.find("projection_topk_threshold_key") != std::string::npos);
    assert(code.find("ProjectionDirectTopKCountGE") == std::string::npos);
    assert(code.find("ProjectionDirectTopKMinMax") == std::string::npos);

    const std::string large_limit_code = generateJitCodeForSql(
        "SELECT lo_orderkey, lo_revenue "
        "FROM lineorder "
        "ORDER BY lo_revenue DESC "
        "LIMIT 4100",
        buildTestCatalog());

    assert(large_limit_code.find("ProjectionDirectTopKRadixHistogram") != std::string::npos);
    assert(large_limit_code.find("ProjectionDirectTopKCollect") != std::string::npos);
    assert(large_limit_code.find("ProjectionDirectTopKSort") != std::string::npos);
    assert(large_limit_code.find("SortLimitBitonicSort") == std::string::npos);

    const std::string filtered_large_limit_code = generateJitCodeForSql(
        "SELECT lo_orderkey, lo_revenue "
        "FROM lineorder "
        "WHERE lo_revenue > 0 "
        "ORDER BY lo_revenue DESC "
        "LIMIT 4100",
        buildTestCatalog());

    assert(filtered_large_limit_code.find("SortLimitTopKThresholdCollect") != std::string::npos);
    const auto physical = physicalPlanForSql("SELECT lo_orderkey, lo_revenue FROM lineorder WHERE lo_revenue > 0 ORDER BY lo_revenue DESC LIMIT 4100", buildTestCatalog());
    assert(describePlan(physical.plan).find("branch") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 13: SELECT * after PK/FK join avoids wide unconditional projection loads
// ============================================================================
static void test_projection_join_star_selective_direct_load_codegen() {
    std::cout << "Test 13: join SELECT * selective projection codegen... ";

    const std::string code = generateJitCodeForSql(
        "SELECT * "
        "FROM lineorder, customer "
        "WHERE lo_custkey = c_custkey "
        "AND c_region = 1",
        buildTestCatalog());

    assert(code.find("d_lo_orderkey[tile_offset + tid + BLOCK_THREADS * i]") != std::string::npos);
    assert(code.find("d_c_name[c_row_id[i]]") != std::string::npos);
    assert(code.find("BlockProbeAndPHT_2KeyOnly") != std::string::npos);
    assert(code.find("d_projection_write_counts") == std::string::npos);
    assert(code.find("BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_orderkey + tile_offset") == std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 14: grouped aggregate local-reduce cost model keeps wide domains atomic
// ============================================================================
static void test_grouped_aggregate_local_reduce_cost_model_codegen() {
    std::cout << "Test 14: grouped aggregate local-reduce cost model codegen... ";

    const auto catalog = buildTestCatalog();
    const std::string wide_group_code = generateJitCodeForSql(
        "SELECT c_nation, sum(lo_revenue) "
        "FROM lineorder, customer, supplier "
        "WHERE lo_custkey = c_custkey "
        "AND lo_suppkey = s_suppkey "
        "AND c_nation = s_nation "
        "AND c_region = 1 "
        "GROUP BY c_nation",
        catalog);
    assert(wide_group_code.find("local_agg_0[25]") == std::string::npos);
    assert(wide_group_code.find("db::atomicAggregate<std::int64_t,db::AggregateOp::Sum>(d_result[out + 1]") != std::string::npos);

    const std::string small_group_code = generateJitCodeForSql(
        "SELECT c_region, sum(lo_extendedprice * lo_discount) "
        "FROM lineorder, customer "
        "WHERE lo_custkey = c_custkey "
        "GROUP BY c_region",
        catalog);
    assert(small_group_code.find("local_agg_0[5]") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 15: OR-expanded grouped aggregates scope local-reduce temporaries
// ============================================================================
static void test_or_join_grouped_aggregate_local_reduce_scopes_codegen() {
    std::cout << "Test 15: OR join grouped aggregate local-reduce scopes codegen... ";

    const std::string code = generateJitCodeForSql(
        "SELECT d_year, sum(lo_revenue) "
        "FROM lineorder, ddate "
        "WHERE (lo_orderdate = d_datekey OR lo_commitdate = d_datekey) "
        "AND d_year >= 1992 AND d_year <= 1997 "
        "GROUP BY d_year",
        buildTestCatalog());

    const std::string scoped_decl = "            {\n                std::int64_t local_agg_0[7];";
    const std::size_t first = code.find(scoped_decl);
    assert(first != std::string::npos);
    const std::size_t second = code.find(scoped_decl, first + scoped_decl.size());
    assert(second != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 16: query modules no longer own the host algorithm or timing
// ============================================================================
static void test_generated_execute_timing_codegen() {
    std::cout << "Test 16: generated execute timing codegen... ";

    const std::string code = generateJitCodeForSql(
        "SELECT SUM(lo_revenue) "
        "FROM lineorder",
        buildTestCatalog());

    assert(code.find("ctx->") == std::string::npos);
    assert(code.find("malloc_device") == std::string::npos);
    assert(code.find("q.memcpy") == std::string::npos);
    assert(code.find("execute_query") == std::string::npos);
    assert(code.find("db::abi2::validateInvocation") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
/** @brief Pushdown preserves independent slot bindings and a usable parent chain. */
static void test_parameterized_pushdown_bindings_and_parents() {
    std::cout << "Test 17: parameterized pushdown bindings and parents... ";
    hsql::SQLParserResult parsed;
    auto* ast = parseSQL(
        "SELECT SUM(lo_revenue) FROM lineorder, part, supplier "
        "WHERE lo_partkey = p_partkey AND lo_suppkey = s_suppkey "
        "AND p_category = 1 AND s_region = 1 AND lo_quantity > 1", parsed);
    assert(ast);
    auto metadata = buildTestCatalog();
    QueryTranslator translator;
    auto prepared = translator.prepare(ast, *metadata);
    assert(prepared.parameters.size() == 3);
    Optimizer optimizer;
    prepared.query.plan = optimizer.optimize(std::move(prepared.query.plan));
    std::set<std::size_t> slots;
    std::function<void(const OperatorNode*, const OperatorNode*)> check =
        [&](const OperatorNode* node, const OperatorNode* parent) {
            assert(node->parent_ == parent && "Optimizer must preserve parent ownership links");
            if (node->getType() == OperatorType::FILTER) {
                const auto& expr = static_cast<const BinaryExpr&>(*static_cast<const FilterNode*>(node)->predicate);
                const auto& column = static_cast<const ColumnRefExpr&>(*expr.left);
                const auto& parameter = static_cast<const ParameterRefExpr&>(*expr.right);
                const std::size_t expected = column.column_name == "p_category" ? 0 :
                    column.column_name == "s_region" ? 1 : 2;
                assert(parameter.slot == expected);
                assert(collectTableNames(node->getChildren()[0].get()).count(column.table_name) == 1);
                slots.insert(parameter.slot);
            }
            for (const auto& child : node->getChildren()) check(child.get(), node);
        };
    check(prepared.query.plan.get(), nullptr);
    assert(slots == (std::set<std::size_t>{0, 1, 2}));
    std::cout << "PASSED\n";
}

/** @brief PHT occupancy is an explicit cleared buffer and a build/probe contract argument. */
static void test_pht_occupancy_contract() {
    auto catalog=std::make_shared<Catalog>();
    const auto original=buildTestCatalog();
    for (auto table:original->getTablesMetadata()) {
        if (table.getName()=="LINEORDER") table.setColumnNullable("lo_suppkey");
        catalog->pushTableMetadata(table);
    }
    const auto query = physicalPlanForSql(
        "SELECT SUM(lo_revenue) FROM lineorder,supplier WHERE lo_suppkey=s_suppkey", catalog);
    const auto buffer = std::find_if(query.plan.buffers.begin(), query.plan.buffers.end(),
        [](const auto& b) { return b.name == "d_s_hash_table_occupied"; });
    assert(buffer != query.plan.buffers.end());
    assert(buffer->type == DB_TYPE_INT32 && buffer->legacy_cpp_type == "unsigned int");
    size_t users = 0;
    for (const auto& kernel : query.plan.kernels)
        for (const auto& argument : kernel.arguments) users += argument.name == buffer->name;
    assert(users == 2);
    const auto code = describePcudamodules(query);
    assert(code.find("num_tile_items, d_s_hash_table_occupied") != std::string::npos);
    assert(code.find("BlockLoadValidity<BLOCK_THREADS, ITEMS_PER_THREAD>(n_lo_suppkey") != std::string::npos);
    assert(code.find("int items_valid[ITEMS_PER_THREAD]") != std::string::npos);
    size_t checked_bounds=0;
    for (const auto& step:query.plan.steps) {
        const auto* guard=std::get_if<Require>(&step.operation);
        if (!guard || (guard->message.find("row IDs")==std::string::npos && guard->message.find("domain exceeds")==std::string::npos)) continue;
        assert(guard->condition.evaluate({{"rows_LINEORDER",INT32_MAX},{"rows_SUPPLIER",INT32_MAX}})==1);
        assert(guard->condition.evaluate({{"rows_LINEORDER",uint64_t(INT32_MAX)+1},{"rows_SUPPLIER",uint64_t(INT32_MAX)+1}})==0);
        ++checked_bounds;
    }
    assert(checked_bounds>=3); // scan, PHT domain and build-row representability
}

/** @brief Physical flag columns use decoded Int32 buffers and keep value-free parameter slots. */
static void test_decoded_flag_parameter_plan() {
    auto catalog = std::make_shared<Catalog>();
    TableMetadata date("DDATE", db::ssbColumnNames("DDATE"), 2557, false);
    for (const auto& column : db::ssbColumns)
        if (std::string(column.table) == "DDATE") date.setColumnStorage(column.name, column.storage);
    date.setColumnStats("d_weekdayfl", {0, 1, 2});
    catalog->pushTableMetadata(date);
    const auto query = physicalPlanForSql("SELECT SUM(d_year) FROM ddate WHERE d_weekdayfl=1", catalog);
    bool decoded_input = false;
    for (const auto& kernel : query.plan.kernels)
        for (const auto& argument : kernel.arguments)
            if (argument.name == "d_d_weekdayfl") { assert(argument.schema.type == DB_TYPE_INT32); decoded_input = true; }
    assert(decoded_input);
    const auto source = describePcudamodules(query);
    assert(source.find("sycl::") == std::string::npos);
    assert(source.find("execute_query") == std::string::npos);
}

/** @brief Text parameters change values/NULL, not optimized structure or native module identity. */
static void test_text_parameterized_physical_plan() {
    Catalog catalog;
    TableMetadata customer("CUSTOMER", {"c_custkey", "c_name"}, 8, false);
    customer.setColumnStorage("c_name", {PhysicalEncoding::FixedText, 25});
    customer.setColumnNullable("c_name");
    catalog.pushTableMetadata(customer);
    auto prepare = [&](const std::string& value) {
        hsql::SQLParserResult ast;
        auto* statement = parseSQL("SELECT COUNT(c_name),c_name,MIN(c_name) FROM customer WHERE c_name>" + value +
                                   " GROUP BY c_name HAVING MIN(c_name)<>'' ORDER BY 2 LIMIT 3", ast);
        assert(statement);
        auto prepared = QueryTranslator{}.prepare(statement, catalog);
        prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
        return prepared;
    };
    auto a = prepare("'Alice'"), b = prepare("'Я🙂'"), null = prepare("NULL");
    assert(a.query.cacheKey() == b.query.cacheKey() && a.query.cacheKey() == null.query.cacheKey());
    assert(a.parameters.at(2).asText() == "Alice");
    assert(b.parameters.at(2).asText() == u8"Я🙂");
    assert(null.parameters.at(2).isNull() && null.parameters.at(2).type() == ParameterType::Text);
    assert(usesTextPlan(*a.query.plan, catalog, a.query.parameter_schema));
    const auto first = lowerTextPlan(*a.query.plan, catalog, a.query.parameter_schema);
    const auto second = lowerTextPlan(*b.query.plan, catalog, b.query.parameter_schema);
    assert(first.bodies.size() == second.bodies.size() && !first.bodies.empty());
    for (size_t i=0;i<first.bodies.size();++i)
        assert(generatePcudacode(first.plan.kernels[i], first.bodies[i], "fixture").key ==
               generatePcudacode(second.plan.kernels[i], second.bodies[i], "fixture").key);
    std::vector<uint32_t> types;
    for (const auto& buffer : first.plan.buffers)
        if (buffer.role == BufferRole::ResultColumn) types.push_back(buffer.type);
    assert(types == (std::vector<uint32_t>{DB_TYPE_UINT64,DB_TYPE_TEXT,DB_TYPE_TEXT}));
}

/** @brief A changed aggregate keeps the independent build module, without literal-bound lowering. */
static void test_value_free_cross_plan_modules() {
    const auto catalog = buildTestCatalog();
    const auto sources = [&](const std::string& aggregate) {
        hsql::SQLParserResult parsed;
        const auto* ast = parseSQL("SELECT " + aggregate + " FROM lineorder,supplier WHERE lo_suppkey=s_suppkey AND s_region>1", parsed);
        assert(ast);
        auto prepared = QueryTranslator{}.prepare(ast, *catalog);
        prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
        JITContext jit; jit.parameter_schema = prepared.query.parameter_schema;
        JITOperatorVisitor visitor(jit, *catalog);
        prepared.query.plan->accept(visitor);
        const auto physical = visitor.buildPhysicalPlan();
        std::set<std::string> keys;
        for (size_t i=0;i<physical.bodies.size();++i)
            keys.insert(generatePcudacode(physical.plan.kernels[i], physical.bodies[i], "acceptance").key);
        return keys;
    };
    const auto count = sources("COUNT(*)"), sum = sources("SUM(lo_revenue)");
    std::vector<std::string> shared;
    std::set_intersection(count.begin(), count.end(), sum.begin(), sum.end(), std::back_inserter(shared));
    assert(!shared.empty() && count != sum);
}

/** @brief The current SSB SQL resolves against unencoded native types and arbitrary table sizes. */
static void test_native_ssb_plan_schema() {
    Catalog catalog;
    for (const auto& definition : ssbTables) {
        TableMetadata table(definition.name, ssbColumnNames(definition.name), 12345, !*definition.primary_key);
        for (const auto& column : ssbColumns)
            if (std::string_view(column.table) == definition.name) table.setColumnStorage(column.name, column.storage);
        if (*definition.primary_key) table.setColumnPrimaryKey(definition.primary_key);
        catalog.pushTableMetadata(table);
    }
    for (const auto* id : {"11", "12", "13", "21", "22", "23", "31", "32", "33", "34", "41", "42", "43"}) {
        std::ifstream file(std::string(CRYSTAL_TEST_SOURCE_DIR) + "/test/queries/sql/q" + id + ".sql");
        assert(file);
        const std::string sql{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        hsql::SQLParserResult parsed;
        const auto* ast = parseSQL(sql, parsed);
        assert(ast);
        auto prepared = QueryTranslator{}.prepare(ast, catalog);
        prepared.query.plan = Optimizer{}.optimize(std::move(prepared.query.plan));
        if (id[0] == '1') continue; // These three use only numeric fact columns.
        assert(usesTextPlan(*prepared.query.plan, catalog, prepared.query.parameter_schema));
        const auto native = lowerTextPlan(*prepared.query.plan, catalog, prepared.query.parameter_schema);
        assert(!native.bodies.empty());
        for (std::size_t i = 0; i < native.bodies.size(); ++i) {
            const auto source = generatePcudacode(native.plan.kernels[i], native.bodies[i], "native-ssb").source;
            assert(source.find("sycl::") == std::string::npos && source.find("MFGR#") == std::string::npos);
        }
    }
}

int main() {
    test_native_ssb_plan_schema();
    test_value_free_cross_plan_modules();
    test_text_parameterized_physical_plan();
    test_decoded_flag_parameter_plan();
    test_pht_occupancy_contract();
    test_physical_plan_boundaries_and_control_flow();
    std::cout << "=== test_optimizer (new pipeline) ===\n\n";

    test_parameterized_pushdown_bindings_and_parents();
    test_pushdown_rule_basic();
    test_pushdown_rule_via_translator();
    test_jit_visitor_q11();
    test_calculate_result_size();
    test_jit_visitor_q21();
    test_jit_visitor_q31();
    test_jit_visitor_mixed_predicate_universal_path();
    test_jit_visitor_unsupported_predicate_throws();
    test_catalog_uniqueness_metadata();
    test_catalog_nullable_metadata();
    test_nullable_expression_codegen_and_typed_result_abi();
    test_topn_projection_direct_radix_codegen();
    test_projection_join_star_selective_direct_load_codegen();
    test_grouped_aggregate_local_reduce_cost_model_codegen();
    test_or_join_grouped_aggregate_local_reduce_scopes_codegen();
    test_generated_execute_timing_codegen();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
