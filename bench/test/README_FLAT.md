# Flat-table benchmark variant

This package adds a flat-table SQL variant for `lineorder_flat` and updates benchmark scripts.

## Files

- `bench_clickhouse.py`
- `bench_duckdb_cli.py`
- `bench_common.py`
- `olap_bench/manifest_flat.json`
- `olap_bench/sql_flat/...`
- `olap_bench/flat_unsupported.json`

Copy/extract the package into the repository root, so that `olap_bench/manifest_flat.json` is next to the existing `olap_bench/manifest.json`.

## Star schema run

```bash
python3 bench_clickhouse.py \
  --schema star \
  --database ssb \
  --user default \
  --ask-password \
  --mode both \
  --out-dir results/clickhouse_star
```

```bash
python3 bench_duckdb_cli.py \
  --schema star \
  --database /path/to/ssb_sf20.duckdb \
  --read-only \
  --threads 64 \
  --mode both \
  --out-dir results/duckdb_star
```

`--schema star` uses `olap_bench/manifest.json` by default.

## Flat table run

```bash
python3 bench_clickhouse.py \
  --schema flat \
  --database ssb \
  --user default \
  --ask-password \
  --mode both \
  --out-dir results/clickhouse_flat
```

```bash
python3 bench_duckdb_cli.py \
  --schema flat \
  --database /path/to/ssb_sf20.duckdb \
  --read-only \
  --threads 64 \
  --mode both \
  --out-dir results/duckdb_flat
```

`--schema flat` uses `olap_bench/manifest_flat.json` by default.

## Heavy queries

`current_heavy` queries are included by default together with `current`.
Use `--no-heavy` to exclude them:

```bash
python3 bench_clickhouse.py --schema flat --database ssb --no-heavy
```

`future` queries are excluded.

## Mode

`--mode cold` runs measured cold executions only.

`--mode hot` runs warmups and measured hot executions only.

`--mode both` runs both phases for every query:

1. measured cold run(s), controlled by `--cold-runs`;
2. warmup run(s), controlled by `--warmups`;
3. measured hot run(s), controlled by `--runs`.

The default is `--mode both`.

## Unsupported flat queries

The flat table is fact-rooted. Some star-schema queries do not have a strict flat equivalent with your current `lineorder_flat` definition:

- dimension-only health checks: `count(*) from customer/part/supplier/ddate`;
- `select * from ddate`;
- `custom1`, because it joins `ddate` by `lo_orderdate` or `lo_commitdate`, while `lineorder_flat` contains date attributes only for `lo_orderdate`.

These queries are omitted from `manifest_flat.json` and listed in `olap_bench/flat_unsupported.json`.
