#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
import sys
import time

from bench_common import (
    add_common_args,
    load_manifest_queries,
    resolve_manifest_path,
    selected_phases,
    timestamped_out_dir,
    write_rows_csv,
    write_summary_csv,
)


ERROR_MARKERS = (
    "Parser Error:",
    "Binder Error:",
    "Catalog Error:",
    "Conversion Error:",
    "Invalid Input Error:",
    "IO Error:",
    "Out of Memory Error:",
    "TransactionContext Error:",
)


def wrap_duckdb_sql(sql: str, output_mode: str) -> str:
    sql = sql.strip().rstrip(";")
    if output_mode == "null":
        return f"COPY ({sql}) TO '/dev/null' (FORMAT CSV, HEADER false);"
    return sql + ";"


def run_hook(command: str | None) -> None:
    if not command:
        return
    subprocess.run(command, shell=True, check=True)


def make_duckdb_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [args.duckdb_bin]
    if args.read_only:
        cmd.append("-readonly")
    cmd.append(args.database)
    return cmd


class DuckDBSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.proc = subprocess.Popen(
            make_duckdb_cmd(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self._marker_seq = 0
        self._send_init()

    def _send_init(self) -> None:
        init_commands = [
            ".bail off",
            ".headers off",
            ".mode list",
            ".timer off",
        ]
        for pragma in self.args.pragma:
            init_commands.append(f"PRAGMA {pragma};")
        if self.args.threads:
            init_commands.append(f"PRAGMA threads={int(self.args.threads)};")
        if self.args.memory_limit:
            init_commands.append(f"PRAGMA memory_limit='{self.args.memory_limit}';")

        self.proc.stdin.write("\n".join(init_commands) + "\n")
        self.proc.stdin.flush()

    def run_sql(self, sql: str, timeout_sec: int) -> tuple[bool, str]:
        self._marker_seq += 1
        marker = f"__DUCKDB_BENCH_DONE__{self._marker_seq}__"
        command = wrap_duckdb_sql(sql, self.args.output_mode) + "\n"
        command += f"SELECT '{marker}';\n"

        self.proc.stdin.write(command)
        self.proc.stdin.flush()

        output_lines = []
        deadline = time.monotonic() + timeout_sec
        while True:
            if time.monotonic() > deadline:
                self.close(kill=True)
                return False, "timeout"

            line = self.proc.stdout.readline()
            if line == "":
                rc = self.proc.poll()
                if rc is not None:
                    return False, f"duckdb process exited with code {rc}"
                time.sleep(0.01)
                continue

            output_lines.append(line)
            if marker in line:
                break

        output = "".join(output_lines)
        success = not any(marker_text in output for marker_text in ERROR_MARKERS)
        return success, output

    def close(self, kill: bool = False) -> None:
        if self.proc.poll() is None:
            if kill:
                self.proc.kill()
            else:
                try:
                    self.proc.stdin.write(".quit\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


def measured_run(args, query: dict, phase: str, run_index: int, session: DuckDBSession | None = None) -> dict:
    if args.print_sql:
        print(f"\n-- {query['query_id']} [{phase} {run_index}]\n{query['sql']};", flush=True)

    if phase == "cold":
        run_hook(args.cold_before_command)

    own_session = session is None
    if own_session:
        session = DuckDBSession(args)

    started = time.perf_counter()
    success = True
    error = ""

    try:
        ok, output = session.run_sql(query["sql"], query["timeout_sec"])
        success = ok
        if not ok:
            error = output.replace("\n", " ")[:2000]
    except Exception as exc:
        success = False
        error = str(exc).replace("\n", " ")[:2000]

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if own_session:
        session.close()

    return {
        "engine": "duckdb-cli",
        "schema": args.schema,
        "mode": args.mode,
        "class_id": query["class_id"],
        "query_id": query["query_id"],
        "source_query_id": query["source_query_id"],
        "path": query["path"],
        "statement_index": query["statement_index"],
        "status": query["status"],
        "phase": phase,
        "run_index": run_index,
        "measured": phase in ("cold", "hot"),
        "elapsed_ms": elapsed_ms,
        "success": success,
        "error": error,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark current OLAP manifest queries on DuckDB CLI. No Python duckdb package required.")
    add_common_args(ap)
    ap.add_argument("--database", required=True, help="Path to DuckDB database file.")
    ap.add_argument("--duckdb-bin", default="duckdb", help="Path to duckdb executable.")
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--memory-limit", default=None, help="DuckDB memory limit, e.g. 32GB.")
    ap.add_argument("--pragma", action="append", default=[], help="Extra PRAGMA body, e.g. preserve_insertion_order=false.")
    ap.add_argument("--output-mode", choices=["null", "client"], default="null")
    ap.add_argument(
        "--cold-before-command",
        default=None,
        help="Optional shell command executed before every measured cold run.",
    )
    args = ap.parse_args()

    manifest = resolve_manifest_path(args)
    classes = set(args.classes) if args.classes else None
    queries = load_manifest_queries(manifest, include_heavy=not args.no_heavy, classes=classes)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else timestamped_out_dir("olap_bench/results", "duckdb_cli")
    rows = []
    do_cold, do_hot = selected_phases(args.mode)

    print(f"Schema: {args.schema}")
    print(f"Manifest: {manifest}")
    print(f"Selected queries/statements: {len(queries)}")
    print(f"Output directory: {out_dir}")

    failures = 0
    hot_session = DuckDBSession(args) if do_hot else None

    try:
        for q in queries:
            if do_cold:
                for i in range(args.cold_runs):
                    row = measured_run(args, q, "cold", i, session=None)
                    rows.append(row)
                    print(f"[cold] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                    failures += 0 if row["success"] else 1
                    if args.fail_on_error and not row["success"]:
                        raise SystemExit(1)

            if do_hot:
                for i in range(args.warmups):
                    row = measured_run(args, q, "warmup", i, session=hot_session)
                    row["measured"] = False
                    rows.append(row)
                    print(f"[warmup] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                    failures += 0 if row["success"] else 1
                    if args.fail_on_error and not row["success"]:
                        raise SystemExit(1)

                for i in range(args.runs):
                    row = measured_run(args, q, "hot", i, session=hot_session)
                    rows.append(row)
                    print(f"[hot] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                    failures += 0 if row["success"] else 1
                    if args.fail_on_error and not row["success"]:
                        raise SystemExit(1)
    finally:
        if hot_session is not None:
            hot_session.close()

    write_rows_csv(out_dir / "runs.csv", rows)
    write_summary_csv(out_dir / "summary.csv", rows)
    print(f"Wrote: {out_dir / 'runs.csv'}")
    print(f"Wrote: {out_dir / 'summary.csv'}")
    return 1 if failures and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
