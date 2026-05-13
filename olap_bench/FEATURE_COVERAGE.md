# OLAP feature coverage checklist

This file separates the benchmark feature surface into current functionality and future functionality.

## Current functionality to keep fast and correct

- Single-table scan.
- Predicate filters with `=`, `<`, `<=`, `>`, `>=`, arithmetic expressions, `AND`, `OR`, `NOT`.
- `IS NULL` and `IS NOT NULL` for the current nullable model.
- Projection of columns and scalar expressions.
- `SELECT *` for single table and PK/FK join materialization.
- Inner equi-joins on SSB PK/FK keys.
- Limited OR-join forms already used by custom tests.
- Scalar aggregates: `COUNT(*)`, `COUNT(expr)`, `SUM(expr)`, `MIN(expr)`, `MAX(expr)`, `AVG(expr)`.
- Grouped aggregates over common SSB dimension columns.
- Typed columnar result fetch for projection and dense result paths.
- Sparse fast path for SSB-style grouped aggregates when no post-aggregate filter/order requires dense output.
- Runtime memory guard and dynamic materialization for projection.
- SQL-level `ORDER BY` and `LIMIT` on final GPU materialized results using device-side Bitonic Sort for ordering.
- `HAVING` after grouped aggregation when referenced aggregate expressions are present in the SELECT output.
- Table aliases for ordinary non-self-join queries.
- Column aliases in SELECT output and references from `ORDER BY` / `HAVING`.
- Native Hyrise AST `CASE` support: searched CASE, simple CASE, multiple `WHEN` branches, optional `ELSE` as SQL NULL; internally lowered to nested `CaseWhenExpr` nodes.
- Nullable mini dataset with normalized expected outputs under `test/nullable_data`.

## Required missing OLAP functionality

These are required for a general-purpose OLAP SQL engine, even if not required by current SSB smoke tests.

1. Top-N optimization beyond full Bitonic Sort. `ORDER BY`/`LIMIT` semantics are supported; a heap/selection Top-N kernel is still future work.
2. Hidden aggregate slots for `HAVING` expressions not projected by SELECT.
3. `DISTINCT` and `COUNT(DISTINCT ...)`.
4. Self-join aliases. Ordinary table aliases are supported; two independent logical aliases over the same physical table are still unsupported.
5. Full MHT coverage on real non-unique joins, including OR-join dedup on `(probe_row_id, build_row_id)`.
6. Outer joins, at minimum `LEFT JOIN`.
7. Derived tables, scalar subqueries, `IN`/`EXISTS`, and CTEs.
8. General type coercion for CASE branches beyond the current numeric execution path.
9. SQL type system beyond encoded `int`: Decimal, Float, Date, String/string-id semantics, casts and type coercion.
10. Device-side result materialization for every sparse/dense pipeline breaker without host computation.
11. Deterministic correctness oracle workflow, ideally ClickHouse/DuckDB comparison on the same generated data.
12. A performance regression gate: per-query generated-code checks and timing thresholds.

## Performance policy

The generator must specialize per query. Heavy abstractions are allowed only when the query semantics require them. Examples:

- SSB PK/FK grouped aggregates should use the sparse fast path and minimal registers.
- Projection with unknown output cardinality should use two-pass materialization.
- Non-unique joins should use MHT only when metadata requires row expansion.
- Null bitmap reads should be omitted entirely for non-nullable columns.
- Dense/columnar materialization should not be inserted into the hot path of a query that can return a compact sparse result safely.
