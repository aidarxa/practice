#!/usr/bin/env python3
import argparse
import getpass
import http.client
import os
import pathlib
import subprocess
import time
import urllib.parse

from bench_common import (
    add_common_args,
    load_manifest_queries,
    resolve_manifest_path,
    selected_phases,
    timestamped_out_dir,
    write_rows_csv,
    write_summary_csv,
)


def wrap_clickhouse_sql(sql: str, output_mode: str) -> str:
    sql = sql.strip().rstrip(";")
    if output_mode == "null":
        return f"{sql} FORMAT Null"
    return sql


def run_hook(command: str | None) -> None:
    if not command:
        return
    subprocess.run(command, shell=True, check=True)


def resolve_password(args: argparse.Namespace) -> str:
    if args.ask_password:
        return getpass.getpass("ClickHouse password: ")

    if args.password_env:
        return os.environ.get(args.password_env, "")

    return args.password or ""


def clickhouse_request(
    args: argparse.Namespace,
    sql: str,
    password: str,
    reuse_conn: http.client.HTTPConnection | None = None,
) -> None:
    params = {
        "database": args.database,
        "default_format": "Null" if args.output_mode == "null" else args.format,
    }
    for setting in args.setting:
        if "=" not in setting:
            raise ValueError(f"--setting must have key=value form, got: {setting}")
        key, value = setting.split("=", 1)
        params[key] = value

    path = "/?" + urllib.parse.urlencode(params)
    body = wrap_clickhouse_sql(sql, args.output_mode).encode("utf-8")
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "X-ClickHouse-User": args.user,
    }
    if password:
        headers["X-ClickHouse-Key"] = password

    conn = reuse_conn or http.client.HTTPConnection(args.host, args.http_port, timeout=args.timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        payload = resp.read()
        if resp.status >= 400:
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
    finally:
        if reuse_conn is None:
            conn.close()


def measured_run(args, query: dict, phase: str, run_index: int, password: str, conn=None) -> dict:
    if args.print_sql:
        print(f"\n-- {query['query_id']} [{phase} {run_index}]\n{query['sql']};", flush=True)

    if phase == "cold":
        run_hook(args.cold_before_command)

    started = time.perf_counter()
    success = True
    error = ""
    try:
        clickhouse_request(args, query["sql"], password=password, reuse_conn=conn)
    except Exception as exc:
        success = False
        error = str(exc).replace("\n", " ")[:2000]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "engine": "clickhouse",
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
    ap = argparse.ArgumentParser(description="Benchmark current OLAP manifest queries on ClickHouse over HTTP.")
    add_common_args(ap)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--http-port", type=int, default=8123)
    ap.add_argument("--database", default="default")
    ap.add_argument("--user", default="default")
    ap.add_argument("--password", default="", help="ClickHouse password. Avoid this on shared machines because it is visible in shell history/process lists.")
    ap.add_argument("--ask-password", action="store_true", help="Prompt for ClickHouse password with hidden input.")
    ap.add_argument("--password-env", default=None, help="Read ClickHouse password from this environment variable, e.g. CLICKHOUSE_PASSWORD.")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output-mode", choices=["null", "client"], default="null")
    ap.add_argument("--format", default="TabSeparated")
    ap.add_argument("--setting", action="append", default=["use_query_cache=0"], help="ClickHouse HTTP setting key=value. Can be repeated.")
    ap.add_argument("--cold-before-command", default=None, help="Optional shell command executed before every measured cold run.")
    args = ap.parse_args()

    if args.ask_password and args.password_env:
        raise SystemExit("Use either --ask-password or --password-env, not both.")
    if args.ask_password and args.password:
        raise SystemExit("Use either --ask-password or --password, not both.")
    if args.password_env and args.password:
        raise SystemExit("Use either --password-env or --password, not both.")

    password = resolve_password(args)

    manifest = resolve_manifest_path(args)
    classes = set(args.classes) if args.classes else None
    queries = load_manifest_queries(manifest, include_heavy=not args.no_heavy, classes=classes)

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else timestamped_out_dir("olap_bench/results", "clickhouse")
    rows = []
    do_cold, do_hot = selected_phases(args.mode)

    print(f"Schema: {args.schema}")
    print(f"Manifest: {manifest}")
    print(f"Selected queries/statements: {len(queries)}")
    print(f"Output directory: {out_dir}")

    failures = 0

    for q in queries:
        if do_cold:
            for i in range(args.cold_runs):
                row = measured_run(args, q, "cold", i, password=password, conn=None)
                rows.append(row)
                print(f"[cold] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                failures += 0 if row["success"] else 1
                if args.fail_on_error and not row["success"]:
                    raise SystemExit(1)

        if do_hot:
            # Keep the connection only within one query's warmup + measured hot runs.
            # A single global keep-alive connection can be closed by ClickHouse while long cold
            # queries are executing on separate connections, causing BrokenPipe/Request-sent
            # errors for every later hot run.
            hot_conn = http.client.HTTPConnection(args.host, args.http_port, timeout=args.timeout)
            try:
                for i in range(args.warmups):
                    row = measured_run(args, q, "warmup", i, password=password, conn=hot_conn)
                    row["measured"] = False
                    rows.append(row)
                    print(f"[warmup] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                    failures += 0 if row["success"] else 1
                    if args.fail_on_error and not row["success"]:
                        raise SystemExit(1)

                for i in range(args.runs):
                    row = measured_run(args, q, "hot", i, password=password, conn=hot_conn)
                    rows.append(row)
                    print(f"[hot] {q['query_id']} run={i} ok={row['success']} ms={row['elapsed_ms']:.3f}")
                    failures += 0 if row["success"] else 1
                    if args.fail_on_error and not row["success"]:
                        raise SystemExit(1)
            finally:
                hot_conn.close()

    write_rows_csv(out_dir / "runs.csv", rows)
    write_summary_csv(out_dir / "summary.csv", rows)
    print(f"Wrote: {out_dir / 'runs.csv'}")
    print(f"Wrote: {out_dir / 'summary.csv'}")
    return 1 if failures and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
