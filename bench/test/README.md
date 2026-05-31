# SQL benchmark scripts v2

This version does not require the Python `duckdb` package.

Files:
- `bench_common.py`
- `bench_clickhouse.py`
- `bench_duckdb_cli.py`

Manifest filtering:
- `status = current` is included.
- `status = future` is always skipped.
- `status = current_heavy` is skipped unless `--include-heavy` is passed.

## Install DuckDB CLI

Recommended:

```bash
curl https://install.duckdb.org | sh
echo 'export PATH="$HOME/.duckdb/cli/latest:$PATH"' >> ~/.bashrc
source ~/.bashrc
duckdb --version
```

Alternative: download `duckdb_cli-linux-amd64.zip` from DuckDB's installation page and put the `duckdb` binary into a directory from `PATH`.

## DuckDB benchmark

```bash
python3 bench_duckdb_cli.py \
  --manifest olap_bench/manifest.json \
  --database /path/to/ssb_sf20.duckdb \
  --read-only \
  --threads 64 \
  --mode both \
  --cold-runs 1 \
  --warmups 1 \
  --runs 5 \
  --out-dir results/duckdb_current
```

If `duckdb` is not in `PATH`:

```bash
python3 bench_duckdb_cli.py \
  --duckdb-bin /absolute/path/to/duckdb \
  --manifest olap_bench/manifest.json \
  --database /path/to/ssb_sf20.duckdb \
  --mode both
```

## ClickHouse benchmark

```bash
python3 bench_clickhouse.py \
  --manifest olap_bench/manifest.json \
  --database ssb \
  --host localhost \
  --http-port 8123 \
  --mode both \
  --cold-runs 1 \
  --warmups 1 \
  --runs 5 \
  --out-dir results/clickhouse_current
```

## Outputs

Each run directory contains:
- `runs.csv`: every cold/warmup/hot run.
- `summary.csv`: grouped measured results with mean/median/min/max.

## Cold/hot interpretation

- DuckDB cold: a new DuckDB CLI process for a measured run.
- DuckDB hot: one persistent DuckDB CLI process reused for warmup and hot measurements.
- ClickHouse cold: a new HTTP connection for a measured run.
- ClickHouse hot: one HTTP connection reused for warmup and hot measurements.

For stricter cold measurements, pass a hook:

```bash
--cold-before-command 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
```


## ClickHouse password options

Hidden prompt:

```bash
python3 bench_clickhouse.py \
  --manifest olap_bench/manifest.json \
  --database ssb \
  --user default \
  --ask-password \
  --mode both \
  --out-dir results/clickhouse_current
```

Environment variable:

```bash
export CLICKHOUSE_PASSWORD='your_password'

python3 bench_clickhouse.py \
  --manifest olap_bench/manifest.json \
  --database ssb \
  --user default \
  --password-env CLICKHOUSE_PASSWORD \
  --mode both \
  --out-dir results/clickhouse_current
```

Direct argument:

```bash
python3 bench_clickhouse.py \
  --manifest olap_bench/manifest.json \
  --database ssb \
  --user default \
  --password 'your_password' \
  --mode both \
  --out-dir results/clickhouse_current
```

Prefer `--ask-password` or `--password-env`; `--password` can be exposed in shell history and process lists.


## ClickHouse keep-alive note

`bench_clickhouse.py` v4 keeps a persistent HTTP connection only within a single query's warmup + hot runs.
This avoids `Broken pipe` / `Request-sent` failures after long cold queries, when ClickHouse closes an idle keep-alive connection.
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
