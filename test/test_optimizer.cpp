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

#include <SQLParser.h>
#include <cassert>
#include <iostream>
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

    assert(code.find("extern \"C\" void execute_query(db::ExecutionContext* ctx)") != std::string::npos);
    assert(code.find("sycl::queue& q = *(ctx->q_);") != std::string::npos);
    assert(code.find("class select_kernel;") != std::string::npos);
    // Скалярная агрегация → reduce_over_group + atomic_result
    assert(code.find("reduce_over_group") != std::string::npos);
    assert(code.find("atomic_result") != std::string::npos);
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

    assert(code.find("extern \"C\" void execute_query(db::ExecutionContext* ctx)") != std::string::npos);
    assert(code.find("class select_kernel;") != std::string::npos);
    // Build-ядра присутствуют (DDATE, PART, SUPPLIER)
    assert(code.find("build_hashtable_") != std::string::npos);
    // Хеш-таблицы аллоцируются
    assert(code.find("sycl::malloc_device<int>") != std::string::npos);
    // Групповая агрегация → atomic_ref, НЕ reduce_over_group
    assert(code.find("sycl::atomic_ref") != std::string::npos);
    assert(code.find("fetch_add") != std::string::npos);
    assert(code.find("reduce_over_group") == std::string::npos);
    // Ровно один q.wait()
    size_t waits = 0, pos = 0;
    while ((pos = code.find("q.wait();", pos)) != std::string::npos) { ++waits; pos += 9; }
    assert(waits == 1 && "Must have exactly one q.wait()");

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
    while ((pos = code.find("build_hashtable_", pos)) != std::string::npos) { ++builds; pos += 16; }
    assert(builds >= 3 && "Q3.1 must emit at least 3 build kernels");

    assert(code.find("sycl::atomic_ref") != std::string::npos);

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
        return visitor.generateCode();
    };

    const std::string agg_code = generate(
        "SELECT COUNT(*), COUNT(lo_revenue), SUM(lo_revenue), AVG(lo_revenue) "
        "FROM lineorder");
    assert(agg_code.find("BlockLoadValidity") != std::string::npos);
    assert(agg_code.find("items_valid") != std::string::npos);
    assert(agg_code.find("getResultColumnUInt64Pointer(0)") != std::string::npos);
    assert(agg_code.find("getResultColumnUInt64Pointer(1)") != std::string::npos);
    assert(agg_code.find("getResultColumnUInt64Pointer(2)") != std::string::npos);
    assert(agg_code.find("getResultColumnFloat64Pointer(3)") != std::string::npos);

    const std::string is_null_projection = generate(
        "SELECT lo_revenue IS NULL FROM lineorder");
    assert(is_null_projection.find("getResultColumnUInt64Pointer(0)") != std::string::npos);
    assert(is_null_projection.find("(!(items_valid[i]))") != std::string::npos ||
           is_null_projection.find("!((items_valid[i]))") != std::string::npos);
    assert(is_null_projection.find("d_result[out_row") == std::string::npos);

    const std::string where_is_null = generate(
        "SELECT COUNT(*) FROM lineorder WHERE lo_revenue IS NULL");
    assert(where_is_null.find("flags[i] = flags[i] && 0") == std::string::npos);
    assert(where_is_null.find("!(lo_revenue_valid[i])") != std::string::npos ||
           where_is_null.find("!(items_valid[i])") != std::string::npos);

    std::cout << "PASSED\n";
}

// ============================================================================
int main() {
    std::cout << "=== test_optimizer (new pipeline) ===\n\n";

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

    std::cout << "\nAll tests passed!\n";
    return 0;
}
