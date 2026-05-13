# OLAP feature coverage checklist

This file separates the benchmark feature surface into current functionality and required future functionality.

## Current functionality to keep fast and correct

- Single-table scan.
- Predicate filters with `=`, `<`, `<=`, `>`, `>=`, arithmetic expressions, `AND`, `OR`, `NOT`.
- `IS NULL` and `IS NOT NULL` for current nullable model.
- Projection of columns and scalar expressions.
- `SELECT *` for single table and PK/FK join materialization.
- Inner equi-joins on SSB PK/FK keys.
- Limited OR-join forms already used by custom tests.
- Scalar aggregates: `COUNT(*)`, `COUNT(expr)`, `SUM(expr)`, `MIN(expr)`, `MAX(expr)`, `AVG(expr)`.
- Grouped aggregates over common SSB dimension columns.
- Typed columnar result fetch for projection and dense result paths.
- Sparse fast path for SSB-style grouped aggregates.
- Runtime memory guard and dynamic materialization for projection.

## Required missing OLAP functionality

These are required for a general-purpose OLAP SQL engine, even if not required by current SSB smoke tests.

1. SQL-level `ORDER BY`, `LIMIT`, and Top-N. Output limiting in `db_cli` is not SQL semantics.
2. `HAVING` after grouped aggregation.
3. `DISTINCT` and `COUNT(DISTINCT ...)`.
4. Table aliases and self-joins. These are necessary for real SQL workloads and for synthetic MHT tests.
5. Full MHT coverage on real non-unique joins, including OR-join dedup on `(probe_row_id, build_row_id)`.
6. Outer joins, at minimum `LEFT JOIN`.
7. Derived tables, scalar subqueries, `IN`/`EXISTS`, and CTEs.
8. `CASE WHEN` expression support.
9. SQL type system beyond encoded `int`: Decimal, Float, Date, String/string-id semantics, casts and type coercion.
10. SQL three-valued logic regression suite with real nullable test data, not only standard non-null SSB.
11. Device-side result materialization for every sparse/dense pipeline breaker without host computation.
12. Deterministic correctness oracle workflow, ideally ClickHouse/DuckDB comparison on the same generated data.
13. A performance regression gate: per-query generated-code checks and timing thresholds.

## Performance policy

The generator must specialize per query. Heavy abstractions are allowed only when the query semantics require them. Examples:

- SSB PK/FK grouped aggregates should use the sparse fast path and minimal registers.
- Projection with unknown output cardinality should use two-pass materialization.
- Non-unique joins should use MHT only when metadata requires row expansion.
- Null bitmap reads should be omitted entirely for non-nullable columns.
- Dense/columnar materialization should not be inserted into the hot path of a query that can return a compact sparse result safely.

