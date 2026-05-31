#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import pathlib
import statistics


def split_sql_statements(sql: str) -> list[str]:
    statements = []
    buf = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                buf.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                in_line_comment = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

        if ch == "'" and not in_double:
            buf.append(ch)
            if in_single and nxt == "'":
                buf.append(nxt)
                i += 2
                continue
            in_single = not in_single
            i += 1
            continue

        if ch == '"' and not in_single:
            buf.append(ch)
            in_double = not in_double
            i += 1
            continue

        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    stmt = "".join(buf).strip()
    if stmt:
        statements.append(stmt)
    return statements


def load_manifest_queries(
    manifest_path: pathlib.Path,
    include_heavy: bool = True,
    classes: set[str] | None = None,
) -> list[dict]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())

    allowed_statuses = {"current"}
    if include_heavy:
        allowed_statuses.add("current_heavy")
    # Future queries remain excluded by default and are not enabled by these scripts.

    out = []
    for cls in manifest["classes"]:
        class_id = cls["id"]
        class_name = cls.get("name", class_id)
        if classes and class_id not in classes and class_name not in classes:
            continue

        for q in cls["queries"]:
            status = q["status"]
            if status not in allowed_statuses:
                continue

            sql_path = root / q["path"]
            sql_text = sql_path.read_text()
            statements = split_sql_statements(sql_text)
            if not statements:
                continue

            for idx, statement in enumerate(statements):
                query_id = q["id"] if len(statements) == 1 else f"{q['id']}__stmt{idx + 1}"
                out.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "query_id": query_id,
                    "source_query_id": q["id"],
                    "path": q["path"],
                    "statement_index": idx,
                    "status": status,
                    "timeout_sec": int(q.get("timeout_sec", 120)),
                    "sql": statement.strip(),
                })

    return out


def write_rows_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "engine", "schema", "mode", "class_id", "query_id", "source_query_id", "path",
        "statement_index", "status", "phase", "run_index", "measured",
        "elapsed_ms", "success", "error",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    groups = {}
    for row in rows:
        if not row.get("measured"):
            continue
        key = (
            row["engine"], row.get("schema", "star"), row["mode"], row["class_id"], row["query_id"],
            row["source_query_id"], row["path"], row["statement_index"], row["status"],
            row["phase"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for key, vals in sorted(groups.items()):
        engine, schema, mode, class_id, query_id, source_query_id, path_text, stmt_idx, status, phase = key
        ok_vals = [v for v in vals if v.get("success") and isinstance(v.get("elapsed_ms"), (int, float))]
        elapsed = [v["elapsed_ms"] for v in ok_vals]
        summary.append({
            "engine": engine,
            "schema": schema,
            "mode": mode,
            "class_id": class_id,
            "query_id": query_id,
            "source_query_id": source_query_id,
            "path": path_text,
            "statement_index": stmt_idx,
            "status": status,
            "phase": phase,
            "runs": len(vals),
            "successful_runs": len(ok_vals),
            "failed_runs": len(vals) - len(ok_vals),
            "elapsed_ms_mean": statistics.mean(elapsed) if elapsed else None,
            "elapsed_ms_median": statistics.median(elapsed) if elapsed else None,
            "elapsed_ms_min": min(elapsed) if elapsed else None,
            "elapsed_ms_max": max(elapsed) if elapsed else None,
        })

    fields = [
        "engine", "schema", "mode", "class_id", "query_id", "source_query_id", "path",
        "statement_index", "status", "phase", "runs", "successful_runs", "failed_runs",
        "elapsed_ms_mean", "elapsed_ms_median", "elapsed_ms_min", "elapsed_ms_max",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow(row)


def timestamped_out_dir(base: str, engine: str) -> pathlib.Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return pathlib.Path(base) / f"{engine}_{stamp}"


def default_manifest_for_schema(schema: str) -> pathlib.Path:
    if schema == "flat":
        return pathlib.Path("olap_bench/manifest_flat.json")
    return pathlib.Path("olap_bench/manifest.json")


def resolve_manifest_path(args: argparse.Namespace) -> pathlib.Path:
    if args.manifest:
        return pathlib.Path(args.manifest)
    return default_manifest_for_schema(args.schema)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--schema", choices=["star", "flat"], default="star", help="Benchmark schema variant. star uses olap_bench/manifest.json; flat uses olap_bench/manifest_flat.json unless --manifest is set.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs", type=int, default=3, help="Number of hot measured runs per query.")
    ap.add_argument("--warmups", type=int, default=1, help="Warmup runs before hot measurements.")
    ap.add_argument("--cold-runs", type=int, default=1, help="Measured cold runs before warmup.")
    ap.add_argument("--mode", choices=["cold", "hot", "both"], default="both")
    ap.add_argument("--no-heavy", action="store_true", help="Exclude current_heavy queries. By default, current and current_heavy are included; future queries are excluded.")
    ap.add_argument("--include-heavy", action="store_true", help="Compatibility no-op: current_heavy is included by default.")
    ap.add_argument("--class", dest="classes", action="append", default=[], help="Manifest class id/name. Can be repeated.")
    ap.add_argument("--fail-on-error", action="store_true")
    ap.add_argument("--print-sql", action="store_true")


def selected_phases(mode: str) -> tuple[bool, bool]:
    return mode in ("cold", "both"), mode in ("hot", "both")
