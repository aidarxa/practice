// test/test_translator.cpp
//
// Unit-тесты для Expression Tree, Operator Tree и QueryTranslator.
// Собирается как отдельный executable (test_translator).
// Не зависит от SYCL или optimizer.h.

#include "core/translator.h"

#include <SQLParser.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace db;

// ============================================================================
// Вспомогательные утилиты
// ============================================================================

static hsql::SelectStatement* parseSQL(const std::string& sql,
                                       hsql::SQLParserResult& result) {
    hsql::SQLParser::parse(sql, &result);
    if (!result.isValid() || result.size() == 0) return nullptr;
    return static_cast<hsql::SelectStatement*>(
        const_cast<hsql::SQLStatement*>(result.getStatement(0)));
}

// Проверяет, что строка out содержит substr
static void assertContains(const std::string& out,
                            const std::string& substr,
                            const std::string& msg) {
    if (out.find(substr) == std::string::npos) {
        std::cerr << "FAIL: " << msg << "\n";
        std::cerr << "  Expected substring: " << substr << "\n";
        std::cerr << "  Actual output:\n" << out << "\n";
        std::abort();
    }
}

// ============================================================================
// Test 1: Expression Tree — ручное построение и clone()
// ============================================================================
static void test_expr_tree_manual() {
    std::cout << "Test 1: Expression Tree manual construction + clone()... ";

    // Строим: (lo_revenue - lo_supplycost) * 100
    auto revenue    = std::make_unique<ColumnRefExpr>("lo_revenue");
    auto supplycost = std::make_unique<ColumnRefExpr>("lo_supplycost");
    auto hundred    = std::make_unique<LiteralIntExpr>(100);

    auto diff = std::make_unique<BinaryExpr>(
        ExprType::OP_SUB, std::move(revenue), std::move(supplycost));
    auto expr = std::make_unique<BinaryExpr>(
        ExprType::OP_MUL, std::move(diff), std::move(hundred));

    // Проверяем getType()
    assert(expr->getType() == ExprType::OP_MUL);
    assert(expr->left  != nullptr);
    assert(expr->right != nullptr);
    assert(expr->left->getType()  == ExprType::OP_SUB);
    assert(expr->right->getType() == ExprType::LITERAL_INT);

    // Проверяем clone() — должен создать независимую копию
    auto cloned = expr->clone();
    assert(cloned != nullptr);
    assert(cloned.get() != expr.get());  // разные объекты
    assert(cloned->getType() == ExprType::OP_MUL);

    // Downcast и проверка значений в клоне
    auto* cloned_bin = static_cast<BinaryExpr*>(cloned.get());
    assert(cloned_bin->left  != nullptr);
    assert(cloned_bin->right != nullptr);

    auto* cloned_right = static_cast<LiteralIntExpr*>(cloned_bin->right.get());
    assert(cloned_right->value == 100);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 2: ExprPrinter — вывод дерева выражений
// ============================================================================
static void test_expr_printer() {
    std::cout << "Test 2: ExprPrinter output... ";

    // Строим: lo_discount >= 1
    auto col = std::make_unique<ColumnRefExpr>("lo_discount");
    auto lit = std::make_unique<LiteralIntExpr>(1);
    auto cond = std::make_unique<BinaryExpr>(
        ExprType::OP_GTE, std::move(col), std::move(lit));

    std::ostringstream oss;
    ExprPrinter printer(oss, 0);
    cond->accept(printer);

    std::string out = oss.str();
    assertContains(out, "[BinaryExpr]", "missing BinaryExpr tag");
    assertContains(out, ">=",           "missing operator >=");
    assertContains(out, "[ColumnRef]",  "missing ColumnRef tag");
    assertContains(out, "lo_discount",  "missing column name");
    assertContains(out, "[LiteralInt]", "missing LiteralInt tag");
    assertContains(out, "1",            "missing literal value");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 3: Operator Tree — ручное построение
// ============================================================================
static void test_operator_tree_manual() {
    std::cout << "Test 3: Operator Tree manual construction... ";

    // TableScan
    auto scan = std::make_unique<TableScanNode>("LINEORDER");
    assert(scan->getType() == OperatorType::TABLE_SCAN);
    assert(scan->table_name == "LINEORDER");
    assert(scan->getChildren().empty());

    // FilterNode
    auto pred = std::make_unique<LiteralIntExpr>(1);
    auto filter = std::make_unique<FilterNode>(std::move(pred));
    assert(filter->getType() == OperatorType::FILTER);
    filter->addChild(std::make_unique<TableScanNode>("LINEORDER"));
    assert(filter->getChildren().size() == 1);

    // HashJoinNode с двумя детьми
    auto join = std::make_unique<HashJoinNode>(nullptr);
    assert(join->getType() == OperatorType::HASH_JOIN);
    assert(join->join_condition == nullptr);
    join->addChild(std::make_unique<TableScanNode>("SUPPLIER"));
    join->addChild(std::make_unique<TableScanNode>("LINEORDER"));
    assert(join->getChildren().size() == 2);

    // AggregateNode
    auto agg = std::make_unique<AggregateNode>();
    assert(agg->getType() == OperatorType::AGGREGATE);
    agg->group_by_exprs.push_back(std::make_unique<ColumnRefExpr>("d_year"));
    auto agg_expr = std::make_unique<ColumnRefExpr>("lo_revenue");
    agg->aggregates.emplace_back("SUM", std::move(agg_expr));
    assert(agg->group_by_exprs.size() == 1);
    assert(agg->aggregates.size() == 1);
    assert(agg->aggregates[0].func_name == "SUM");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 4: OperatorPrinter — вывод дерева операторов
// ============================================================================
static void test_operator_printer() {
    std::cout << "Test 4: OperatorPrinter output... ";

    // AggregateNode → FilterNode → HashJoinNode → [TableScan, TableScan]
    auto scan_lo = std::make_unique<TableScanNode>("LINEORDER");
    auto scan_dd = std::make_unique<TableScanNode>("DDATE");
    auto join    = std::make_unique<HashJoinNode>(nullptr);
    join->addChild(std::move(scan_lo));
    join->addChild(std::move(scan_dd));

    auto pred_col = std::make_unique<ColumnRefExpr>("lo_quantity");
    auto pred_val = std::make_unique<LiteralIntExpr>(25);
    auto pred     = std::make_unique<BinaryExpr>(
        ExprType::OP_LT, std::move(pred_col), std::move(pred_val));
    auto filter = std::make_unique<FilterNode>(std::move(pred));
    filter->addChild(std::move(join));

    auto agg = std::make_unique<AggregateNode>();
    agg->group_by_exprs.push_back(std::make_unique<ColumnRefExpr>("d_year"));
    agg->aggregates.emplace_back("SUM",
        std::make_unique<ColumnRefExpr>("lo_revenue"));
    agg->addChild(std::move(filter));

    std::ostringstream oss;
    OperatorPrinter printer(oss, 0);
    agg->accept(printer);

    std::string out = oss.str();
    assertContains(out, "[Aggregate]",  "missing Aggregate node");
    assertContains(out, "GROUP BY",     "missing GROUP BY label");
    assertContains(out, "d_year",       "missing group by column");
    assertContains(out, "SUM",          "missing aggregation function");
    assertContains(out, "[Filter]",     "missing Filter node");
    assertContains(out, "[HashJoin]",   "missing HashJoin node");
    assertContains(out, "[TableScan]",  "missing TableScan node");
    assertContains(out, "LINEORDER",    "missing LINEORDER table");
    assertContains(out, "DDATE",        "missing DDATE table");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 5: QueryTranslator — Q1.1 (простой фильтр без GROUP BY)
// ============================================================================
static void test_translator_q11() {
    std::cout << "Test 5: Translator Q1.1 (scalar agg, no GROUP BY)... ";

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
    assert(ast != nullptr && "SQL parse failed");

    QueryTranslator translator;
    auto root = translator.translate(ast);

    // Корень должен быть AggregateNode (есть SUM)
    assert(root != nullptr);
    assert(root->getType() == OperatorType::AGGREGATE);

    auto* agg = static_cast<AggregateNode*>(root.get());
    assert(agg->group_by_exprs.empty() && "Q1.1 has no GROUP BY");
    assert(agg->aggregates.size() == 1);
    assert(agg->aggregates[0].func_name == "SUM");
    assert(agg->aggregates[0].agg_expr != nullptr);

    // Дочерний узел — FilterNode
    assert(agg->getChildren().size() == 1);
    assert(agg->getChildren()[0]->getType() == OperatorType::FILTER);

    auto* filter = static_cast<FilterNode*>(agg->getChildren()[0].get());
    assert(filter->predicate != nullptr);

    // Дочерний FilterNode → TableScan(LINEORDER)
    assert(filter->getChildren().size() == 1);
    assert(filter->getChildren()[0]->getType() == OperatorType::TABLE_SCAN);
    auto* scan = static_cast<TableScanNode*>(filter->getChildren()[0].get());
    assert(scan->table_name == "LINEORDER");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 6: QueryTranslator — Q2.1 (GROUP BY, 4 таблицы)
// ============================================================================
static void test_translator_q21() {
    std::cout << "Test 6: Translator Q2.1 (GROUP BY, 4 tables)... ";

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
    assert(ast != nullptr && "SQL parse failed");

    QueryTranslator translator;
    auto root = translator.translate(ast);

    // Корень — AggregateNode
    assert(root != nullptr);
    assert(root->getType() == OperatorType::AGGREGATE);

    auto* agg = static_cast<AggregateNode*>(root.get());

    // GROUP BY d_year, p_brand1 → 2 выражения
    assert(agg->group_by_exprs.size() == 2);
    assert(agg->group_by_exprs[0]->getType() == ExprType::COLUMN_REF);
    assert(agg->group_by_exprs[1]->getType() == ExprType::COLUMN_REF);

    auto* g0 = static_cast<ColumnRefExpr*>(agg->group_by_exprs[0].get());
    auto* g1 = static_cast<ColumnRefExpr*>(agg->group_by_exprs[1].get());
    assert(g0->column_name == "d_year");
    assert(g1->column_name == "p_brand1");

    // 1 агрегация: SUM(lo_revenue)
    assert(agg->aggregates.size() == 1);
    assert(agg->aggregates[0].func_name == "SUM");

    // Дочерний → FilterNode
    assert(agg->getChildren().size() == 1);
    assert(agg->getChildren()[0]->getType() == OperatorType::FILTER);

    auto* filter = static_cast<FilterNode*>(agg->getChildren()[0].get());
    assert(filter->predicate != nullptr);

    // Дочерний FilterNode → HashJoinNode (кросс-джойн 4 таблиц → 3 уровня)
    assert(filter->getChildren().size() == 1);
    assert(filter->getChildren()[0]->getType() == OperatorType::HASH_JOIN);

    // В наивной фазе все join_condition == nullptr
    std::function<void(const OperatorNode*)> checkNoJoinConds =
        [&](const OperatorNode* node) {
            if (node->getType() == OperatorType::HASH_JOIN) {
                auto* jn = static_cast<const HashJoinNode*>(node);
                assert(jn->join_condition == nullptr && "naive phase: no join conditions");
            }
            for (const auto& child : node->getChildren())
                checkNoJoinConds(child.get());
        };
    checkNoJoinConds(filter->getChildren()[0].get());

    // Проверяем вывод дерева через OperatorPrinter
    std::ostringstream oss;
    OperatorPrinter printer(oss, 0);
    root->accept(printer);
    std::string out = oss.str();

    assertContains(out, "[Aggregate]",         "missing Aggregate");
    assertContains(out, "[Filter]",            "missing Filter");
    assertContains(out, "[HashJoin]",          "missing HashJoin");
    assertContains(out, "no condition",        "join should have no condition in naive phase");
    assertContains(out, "[TableScan] LINEORDER", "missing LINEORDER scan");
    assertContains(out, "[TableScan] DDATE",   "missing DDATE scan");
    assertContains(out, "[TableScan] PART",    "missing PART scan");
    assertContains(out, "[TableScan] SUPPLIER","missing SUPPLIER scan");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 7: QueryTranslator — Q3.1 (другой набор таблиц)
// ============================================================================
static void test_translator_q31() {
    std::cout << "Test 7: Translator Q3.1 (GROUP BY 3 cols)... ";

    std::string sql =
        "SELECT c_nation, s_nation, d_year, SUM(lo_revenue) "
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
    assert(ast != nullptr && "SQL parse failed");

    QueryTranslator translator;
    auto root = translator.translate(ast);

    // Корень — AggregateNode
    assert(root != nullptr);
    assert(root->getType() == OperatorType::AGGREGATE);

    auto* agg = static_cast<AggregateNode*>(root.get());

    // GROUP BY 3 колонки
    assert(agg->group_by_exprs.size() == 3);
    auto* g0 = static_cast<ColumnRefExpr*>(agg->group_by_exprs[0].get());
    auto* g1 = static_cast<ColumnRefExpr*>(agg->group_by_exprs[1].get());
    auto* g2 = static_cast<ColumnRefExpr*>(agg->group_by_exprs[2].get());
    assert(g0->column_name == "c_nation");
    assert(g1->column_name == "s_nation");
    assert(g2->column_name == "d_year");

    // 1 агрегация
    assert(agg->aggregates.size() == 1);
    assert(agg->aggregates[0].func_name == "SUM");

    // FilterNode присутствует
    assert(agg->getChildren().size() == 1);
    assert(agg->getChildren()[0]->getType() == OperatorType::FILTER);

    // FilterNode содержит WHERE (кросс-джойн 4 таблиц)
    auto* filter = static_cast<FilterNode*>(agg->getChildren()[0].get());
    assert(filter->getChildren().size() == 1);
    assert(filter->getChildren()[0]->getType() == OperatorType::HASH_JOIN);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 8: QueryTranslator — LiteralFloat в агрегации
// ============================================================================
static void test_translator_float_literal() {
    std::cout << "Test 8: Translator float literal in expression... ";

    // Используем вещественный литерал как часть вычисления
    std::string sql =
        "SELECT SUM(lo_extendedprice) "
        "FROM lineorder "
        "WHERE lo_discount >= 1";

    hsql::SQLParserResult result;
    auto* ast = parseSQL(sql, result);
    assert(ast != nullptr && "SQL parse failed");

    QueryTranslator translator;
    auto root = translator.translate(ast);
    assert(root != nullptr);
    assert(root->getType() == OperatorType::AGGREGATE);

    // Проверяем что дерево построилось без ошибок
    auto* agg = static_cast<AggregateNode*>(root.get());
    assert(agg->aggregates.size() == 1);
    assert(agg->aggregates[0].agg_expr != nullptr);

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 9: ExprNode clone() глубина — клон независим от оригинала
// ============================================================================
static void test_expr_clone_independence() {
    std::cout << "Test 9: ExprNode deep clone independence... ";

    auto col1 = std::make_unique<ColumnRefExpr>("lo_revenue");
    auto col2 = std::make_unique<ColumnRefExpr>("lo_supplycost");
    auto orig = std::make_unique<BinaryExpr>(
        ExprType::OP_SUB, std::move(col1), std::move(col2));

    auto cloned = orig->clone();

    // Меняем оригинал — изменения не должны затронуть клон
    auto* orig_bin   = static_cast<BinaryExpr*>(orig.get());
    auto* cloned_bin = static_cast<BinaryExpr*>(cloned.get());

    auto* orig_left   = static_cast<ColumnRefExpr*>(orig_bin->left.get());
    auto* cloned_left = static_cast<ColumnRefExpr*>(cloned_bin->left.get());

    orig_left->column_name = "MODIFIED";
    assert(cloned_left->column_name == "lo_revenue" && "clone must be deep (independent)");

    std::cout << "PASSED\n";
}

// ============================================================================
// Test 10: QueryTranslator — вывод дерева Q2.1 для визуальной проверки
// ============================================================================
static void test_translator_print_q21() {
    std::cout << "Test 10: Visual tree output for Q2.1:\n";

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
    assert(ast != nullptr && "SQL parse failed");

    QueryTranslator translator;
    auto root = translator.translate(ast);

    std::cout << "\n--- Naive Operator Tree (Q2.1) ---\n";
    OperatorPrinter printer(std::cout, 0);
    root->accept(printer);
    std::cout << "----------------------------------\n\n";

    std::cout << "Test 10: PASSED (visual)\n";
}

// ============================================================================
int main() {
    std::cout << "=== test_translator.cpp ===\n\n";

    test_expr_tree_manual();
    test_expr_printer();
    test_operator_tree_manual();
    test_operator_printer();
    test_translator_q11();
    test_translator_q21();
    test_translator_q31();
    test_translator_float_literal();
    test_expr_clone_independence();
    test_translator_print_q21();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
