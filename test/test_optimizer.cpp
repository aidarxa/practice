#include "../../include/core/optimizer.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

using namespace db;

// ============================================================================
// Helper: parse SQL and return SelectStatement
// ============================================================================
static hsql::SelectStatement *parseSQL(const std::string &sql,
                                       hsql::SQLParserResult &result) {
  hsql::SQLParser::parse(sql, &result);
  if (!result.isValid() || result.size() == 0) return nullptr;
  return (hsql::SelectStatement *)result.getStatement(0);
}

// Build a test catalog matching SSB schema
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
  s.setColumnStats("s_region", {0, 4, 5});
  s.setColumnStats("s_nation", {0, 24, 25});
  catalog->pushTableMetadata(s);

  // CUSTOMER (dimension)
  TableMetadata c("CUSTOMER",
                  {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                   "c_region", "c_phone", "c_mktsegment"},
                  300000, false);
  c.setColumnStats("c_custkey", {1, 300000, 300000});
  c.setColumnStats("c_region", {0, 4, 5});
  c.setColumnStats("c_nation", {0, 24, 25});
  catalog->pushTableMetadata(c);

  // PART (dimension)
  TableMetadata p("PART",
                  {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                   "p_color", "p_type", "p_size", "p_container"},
                  800000, false);
  p.setColumnStats("p_partkey", {1, 800000, 800000});
  p.setColumnStats("p_category", {1, 25, 25});
  p.setColumnStats("p_brand1", {1, 1000, 1000});
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
  d.setColumnStats("d_year", {1992, 1998, 7});
  catalog->pushTableMetadata(d);

  return catalog;
}

// ============================================================================
// Test 1: buildLogicalPlan — Q2.1
// ============================================================================
static void test_buildLogicalPlan_q21() {
  std::cout << "Test 1: buildLogicalPlan Q2.1... ";
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
  auto *ast = parseSQL(sql, result);
  assert(ast != nullptr);

  auto catalog = buildTestCatalog();
  Planner planner(catalog);
  auto lp = planner.buildLogicalPlan(ast);

  // Should have 2 columns (d_year, p_brand1)
  // + 1 aggregation (SUM(lo_revenue))
  assert(!lp->aggregations_.empty());
  assert(lp->aggregations_[0].func_name_ == "SUM");

  // 4 tables
  assert(lp->tables_.size() == 4);

  // 5 conditions: 3 JOIN + 2 FILTER
  int joins = 0, filters = 0;
  for (auto &c : lp->conditions_) {
    if (c.type == ExpressionType::JOIN) joins++;
    if (c.type == ExpressionType::FILTER) filters++;
  }
  assert(joins == 3);
  assert(filters == 2);

  // 2 group-by columns
  assert(lp->group_by_.size() == 2);

  std::cout << "PASSED\n";
}

// ============================================================================
// Test 2: QueryOptimizer — predicate pushdown
// ============================================================================
static void test_optimizer_pushdown() {
  std::cout << "Test 2: QueryOptimizer pushdown... ";
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
  auto *ast = parseSQL(sql, result);
  assert(ast != nullptr);

  auto catalog = buildTestCatalog();
  Planner planner(catalog);
  auto lp = planner.buildLogicalPlan(ast);

  QueryOptimizer opt(catalog);
  opt.optimize(lp);

  // After pushdown, FILTERs should come before JOINs for their tables
  // Verify no condition was lost
  int joins = 0, filters = 0;
  for (auto &c : lp->conditions_) {
    if (c.type == ExpressionType::JOIN) joins++;
    if (c.type == ExpressionType::FILTER) filters++;
  }
  assert(joins == 3);
  assert(filters == 2);
  assert(lp->conditions_.size() == 5);

  std::cout << "PASSED\n";
}

// ============================================================================
// Test 3: buildPhysicalPlan — kernel structure
// ============================================================================
static void test_buildPhysicalPlan_q21() {
  std::cout << "Test 3: buildPhysicalPlan Q2.1... ";
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
  auto *ast = parseSQL(sql, result);
  assert(ast != nullptr);

  auto catalog = buildTestCatalog();
  Planner planner(catalog);
  auto lp = planner.buildLogicalPlan(ast);
  QueryOptimizer opt(catalog);
  opt.optimize(lp);
  auto pp = planner.buildPhysicalPlan(lp);

  // Should have 3 build kernels + 1 select kernel = 4
  assert(pp->kernels.size() == 4);

  // Last kernel should be select_kernel
  assert(pp->kernels.back().name_ == "select_kernel");

  // Should have hash tables (INTERNAL_TEMP)
  assert(!pp->hash_tables_.empty());
  for (auto &ht : pp->hash_tables_) {
    assert(ht.scope_ == BufferScope::INTERNAL_TEMP);
    assert(ht.needs_zeroing_ == true);
  }

  // Result buffer should be EXTERNAL_OUTPUT
  assert(pp->device_result_buffer_.scope_ == BufferScope::EXTERNAL_OUTPUT);
  assert(pp->device_result_buffer_.needs_zeroing_ == true);

  // Data columns should be EXTERNAL_INPUT
  for (auto &dc : pp->data_columns_) {
    assert(dc.scope_ == BufferScope::EXTERNAL_INPUT);
  }

  std::cout << "PASSED\n";
}

// ============================================================================
// Test 4: CodeGenerator — JIT output structure
// ============================================================================
static void test_codegen_q21() {
  std::cout << "Test 4: CodeGenerator Q2.1... ";
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
  auto *ast = parseSQL(sql, result);
  assert(ast != nullptr);

  auto catalog = buildTestCatalog();
  Planner planner(catalog);
  auto lp = planner.buildLogicalPlan(ast);
  QueryOptimizer opt(catalog);
  opt.optimize(lp);
  auto pp = planner.buildPhysicalPlan(lp);

    // Проверяем генератор
    db::CodeGenerator cg;
    std::string code = cg.generate(*pp);

    // 1. Проверяем сигнатуру с ExecutionContext
    assert(code.find("extern \"C\" void execute_query(db::ExecutionContext* ctx) {") != std::string::npos && "Missing execution context signature");
    
    // 2. Проверяем извлечение очереди и буферов
    assert(code.find("sycl::queue& q = *(ctx->q_);") != std::string::npos && "Missing queue extraction");
    assert(code.find("int* d_lo_orderdate = ctx->getBuffer<int>(\"d_lo_orderdate\");") != std::string::npos && "Missing data column pointer extraction");
    assert(code.find("unsigned long long* d_result = ctx->getResultPointer();") != std::string::npos && "Missing d_result pointer extraction");

    // 3. Проверяем, что INTERNAL_TEMP аллоцируются в самом ядре
    assert(code.find("int* d_d_hash_table = sycl::malloc_device<int>") != std::string::npos && "Missing hash table malloc");

    // 4. Проверяем наличие ядер в строке
    assert(code.find("class ") != std::string::npos && "Missing forward declaration");
    assert(code.find("q.submit([&](sycl::handler& h) {") != std::string::npos && "Missing queue submit");

    // 5. Проверяем отсутствие множественных q.wait()
    size_t count_waits = 0;
    size_t pos = 0;
    while ((pos = code.find("q.wait();", pos)) != std::string::npos) {
        count_waits++;
        pos += 9;
    }
    assert(count_waits == 1 && "There should be exactly one q.wait() before sycl::free");

    std::cout << "[ OK ] test_codegen_signature passed.\n";
}

// ============================================================================
// Test 5: Scalar aggregation (Q1x-style, no GROUP BY)
// ============================================================================
static void test_scalar_aggregation() {
  std::cout << "Test 5: Scalar aggregation... ";
  std::string sql =
      "SELECT SUM(lo_extendedprice * lo_discount) "
      "FROM lineorder "
      "WHERE lo_orderdate >= 19930101 "
      "AND lo_orderdate < 19940101 "
      "AND lo_discount >= 1 "
      "AND lo_discount <= 3 "
      "AND lo_quantity < 25";

  hsql::SQLParserResult result;
  auto *ast = parseSQL(sql, result);
  assert(ast != nullptr);

  auto catalog = buildTestCatalog();
  Planner planner(catalog);
  auto lp = planner.buildLogicalPlan(ast);

  // No GROUP BY
  assert(lp->group_by_.empty());
  // 1 aggregation
  assert(lp->aggregations_.size() == 1);
  // 5 FILTERs, 0 JOINs
  int filters = 0;
  for (auto &c : lp->conditions_) {
    if (c.type == ExpressionType::FILTER) filters++;
  }
  assert(filters == 5);

  auto pp = planner.buildPhysicalPlan(lp);

  // Only 1 kernel (no dimensions)
  assert(pp->kernels.size() == 1);
  assert(pp->kernels[0].name_ == "select_kernel");

  // No hash tables
  assert(pp->hash_tables_.empty());

  CodeGenerator cg;
  std::string code = cg.generate(*pp);

  // Should have reduce_over_group for scalar agg
  assert(code.find("reduce_over_group") != std::string::npos);
  assert(code.find("atomic_result") != std::string::npos);

  // Check fact table filters
  assert(code.find("BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 1, num_tile_items);") != std::string::npos);
  assert(code.find("BlockPredALTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 3, num_tile_items);") != std::string::npos);
  assert(code.find("BlockPredAGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 19930101, num_tile_items);") != std::string::npos);
  assert(code.find("BlockPredALT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 19940101, num_tile_items);") != std::string::npos);
  assert(code.find("BlockPredALT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 25, num_tile_items);") != std::string::npos);

  // Check named registers for aggregation
  assert(code.find("int lo_extendedprice[ITEMS_PER_THREAD];") != std::string::npos);
  assert(code.find("int lo_discount[ITEMS_PER_THREAD];") != std::string::npos);
  
  // Check that aggregation uses the named registers and cast
  assert(code.find("sum += ((unsigned long long)lo_extendedprice[i] * lo_discount[i]);") != std::string::npos);

  std::cout << "PASSED\n";
}

// ============================================================================
int main() {
  std::cout << "=== optimizer.cpp tests ===\n\n";
  test_buildLogicalPlan_q21();
  test_optimizer_pushdown();
  test_buildPhysicalPlan_q21();
  test_codegen_q21();
  test_scalar_aggregation();
  std::cout << "\nAll tests passed!\n";
  return 0;
}
