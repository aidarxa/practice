# Crystal-SYCL OLAP benchmark suite

This benchmark suite is organized by OLAP functionality class. It contains two categories of SQL:

- `current`: expected to run on the current engine.
- `future`: intentionally included as targets for future implementation. These queries may fail to parse or execute today.
- `current_heavy`: expected to run, but may allocate or materialize large results. Run explicitly.

The suite is intended for correctness regression, generated-code inspection, and performance tracking. It is not limited to the current parser subset; it also documents the minimum OLAP feature surface expected from the engine.

## Classes

| Class | Purpose |
|---|---|
| `00_health` | Base table availability and scalar count checks. |
| `01_scan_filter` | Predicate evaluation, boolean logic, arithmetic filters, NULL predicates. |
| `02_projection` | Column projection, expression projection, boolean projection, small `SELECT *`. |
| `03_aggregate_scalar` | `COUNT`, `SUM`, `MIN`, `MAX`, `AVG`, aggregate expressions. |
| `04_group_by` | Grouped aggregation, multi-key grouping, group-by expression target. |
| `05_join_pk_fk` | Star-schema unique PK/FK joins. |
| `06_projection_join` | Projection and late materialization after joins. |
| `07_ssb_standard` | Standard SSB query set already used as baseline. |
| `08_custom` | Custom regression queries. |
| `09_nullable` | NULL, `IS NULL`, `IS NOT NULL`, 3-valued logic smoke tests. |
| `10_order_limit_topn_future` | `ORDER BY`, `LIMIT`, Top-N targets. |
| `11_having_future` | `HAVING` targets. |
| `12_distinct_future` | `DISTINCT` and `COUNT(DISTINCT)` targets. |
| `13_outer_join_future` | Outer join targets. |
| `14_subquery_cte_future` | Derived tables and CTE targets. |
| `15_window_future` | Window function targets. |
| `16_case_future` | `CASE WHEN` targets. |
| `17_alias_selfjoin_future` | Aliases and self-join targets. |
| `18_mht_nonunique_future` | Non-unique join / MHT / OR-dedup targets. |
| `19_type_system_future` | Decimal, date extraction, string predicates. |
| `20_memory_stress` | Wide materialization and memory-guard stress tests. |

## Isolated benchmark run

Run each query in a fresh `db_cli` process:

```bash
python3 bench/olap_bench/tools/run_olap_bench.py \
  --db-cli ./build/db_cli \
  --manifest bench/olap_bench/manifest.json \
  --runs 3 \
  --warmups 1 \
  --limit 1000 \
  --out bench/olap_bench/results/latest
```

By default, the runner executes only `current` queries. To include heavy or future queries:

```bash
python3 bench/olap_bench/tools/run_olap_bench.py --include-heavy --include-future ...
```

## Sequential session smoke test

This detects state leaks across queries, including scratch-buffer reuse bugs:

```bash
python3 bench/olap_bench/tools/run_session_smoke.py \
  --db-cli ./build/db_cli \
  --manifest bench/olap_bench/manifest.json \
  --limit 20 \
  --timeout 600 \
  --out bench/olap_bench/results/session_smoke.txt
```

## Output files

`run_olap_bench.py` writes:

- `summary.csv`: one row per measured query run.
- `summary.json`: structured results.
- `logs/<query_id>__run_<n>.out`: raw `db_cli` transcript.
- `logs/<query_id>__run_<n>.err`: stderr.

The runner extracts these metrics when present:

- `Rows returned`
- `Rows shown`
- `GPU execution time`
- `Result materialization/fetch time`
- `Code generation time`
- `ACPP compilation time`
- `Library load time`
- `Engine processing time`

