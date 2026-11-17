#!/usr/bin/env python3
## @file
## @brief Stage-1 CLI regression with offline, independent NumPy expectations.
"""@details This is a correctness runner, not a performance comparison. Native
Bash time observes the process/compiler; Python never times the measured chain.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from sql_baseline import isolate_sql, run_session  # noqa: E402
from native_timing import collect_events, read_time, timed_command  # noqa: E402


def load_timing():
    """@brief Reuse existing SQL timing definitions without renaming their fields."""
    spec = importlib.util.spec_from_file_location("olap_timing", ROOT / "olap_bench/tools/run_olap_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def targeted_queries():
    """@brief Cover changed values and output-reference identity on a small SSB dimension."""
    queries = []
    for value in (1, 2, 99):
        queries.append((f"filter-{value}", f"SELECT COUNT(*) FROM supplier WHERE s_region > {value}", "filter", value))
    queries += [
        ("null-filter", "SELECT COUNT(*) FROM supplier WHERE s_region > NULL", "legacy-rejection", None),
        ("null-or", "SELECT COUNT(*) FROM supplier WHERE s_region = NULL OR s_region > 2", "filter", 2),
        ("null-case", "SELECT SUM(CASE WHEN s_region = NULL THEN 1 ELSE 0 END) FROM supplier", "null-case", None),
        ("large-integer", "SELECT 9007199254740993 AS exact FROM supplier LIMIT 1", "large-integer", None),
        ("float", "SELECT 1.25 AS fraction FROM supplier LIMIT 1", "float", None),
    ]
    for value in (3, 7):
        queries.append((f"arithmetic-{value}",
            f"SELECT SUM(CASE WHEN s_region > 1 THEN s_nation + {value} ELSE 0 END) FROM supplier", "arithmetic", value))
    for then, otherwise in ((10, 10), (11, 13)):
        queries.append((f"independent-{then}-{otherwise}",
            f"SELECT SUM(CASE WHEN s_region > 1 THEN {then} ELSE {otherwise} END) FROM supplier", "independent", [then, otherwise]))
    for addition, threshold, limit in ((3, 0, 3), (7, 40000, 5), (3, 0, 0)):
        queries.append((f"having-{addition}-{threshold}-{limit}",
            f"SELECT s_region, SUM(s_nation + {addition}) AS total FROM supplier GROUP BY s_region "
            f"HAVING total > {threshold} ORDER BY total DESC, s_region LIMIT {limit}", "having", [addition, threshold, limit]))
    queries.append(("ordinal", "SELECT SUM(s_nation) AS total, s_region FROM supplier "
                    "GROUP BY s_region ORDER BY 1 DESC, 2 LIMIT 2", "having", [0, -1, 2]))
    queries += [(name + "-repeat", sql, oracle, value) for name, sql, oracle, value in queries[:3]]
    return queries


def workload(include_manifest, include_heavy):
    """@brief Split current manifest files into marked statements; exclude future SQL."""
    items = []
    if include_manifest:
        manifest = json.loads((ROOT / "olap_bench/manifest.json").read_text())
        allowed = {"current", "current_heavy"} if include_heavy else {"current"}
        for category in manifest["classes"]:
            for query in category["queries"]:
                if query["status"] not in allowed:
                    continue
                text = (ROOT / "olap_bench" / query["path"]).read_text()
                statements = [s.strip() for s in re.sub(r"--[^\n]*", "", text).split(";") if s.strip()]
                for index, statement in enumerate(statements):
                    items.append((query["id"] + f"-{index}", statement, None, None))
    return items + targeted_queries()


def parse_cell_row(line):
    """Decode numeric/NULL cells and lossless quoted UTF-8 Text, including embedded pipes."""
    decoder = json.JSONDecoder()
    cells = []
    remaining = line.lstrip()
    while remaining:
        if remaining.startswith('"'):
            value, end = decoder.raw_decode(remaining)
        else:
            end = remaining.find("|")
            if end < 0:
                end = len(remaining)
            raw = remaining[:end].strip()
            value = None if raw == "NULL" else (float(raw) if "." in raw or "e" in raw.lower() else int(raw))
        cells.append(value)
        remaining = remaining[end:].lstrip()
        if remaining:
            if not remaining.startswith("|"):
                raise ValueError("Invalid result cell separator")
            remaining = remaining[1:].lstrip()
            if not remaining:
                raise ValueError("Missing result cell")
    return tuple(cells)


def parse_rows(text):
    """@brief Read visible typed cells, preserving exact integers, Text and NULL."""
    match = re.search(r"--- Result ---\s*\n(.*?)Rows returned:\s*(\d+)", text, re.S)
    if not match:
        raise ValueError("Missing CLI result")
    lines = match[1].strip().splitlines()
    if not lines or not lines[0].startswith("Columns:"):
        raise ValueError("Missing column names")
    columns = [c.strip() for c in lines.pop(0).split(":", 1)[1].split("|")]
    if int(match[2]) == 0:
        return columns, [], 0
    rows = []
    for line in lines:
        if not line.startswith("Row ") and not re.match(r'(?:"|NULL\b|[-+0-9])', line.strip()):
            continue
        row = parse_cell_row(re.sub(r"^Row \d+:\s*", "", line))
        if len(row) != len(columns):
            raise ValueError("Row width mismatch")
        rows.append(row)
    if not rows:
        raise ValueError("Nonempty result contains no visible rows")
    return columns, rows, int(match[2])


def expected_rows(oracle, argument, data):
    """@brief Offline oracle from physical int32 columns, not generated/device helpers."""
    import numpy as np
    region = np.memmap(data / "SUPPLIER5", dtype="<i4", mode="r")
    nation = np.memmap(data / "SUPPLIER4", dtype="<i4", mode="r")
    if oracle == "filter":
        return [(int(np.count_nonzero(region > argument)),)]
    if oracle == "null-case":
        return [(0,)]
    if oracle == "large-integer":
        return [(9007199254740993,)]
    if oracle == "float":
        return [(1.25,)]
    if oracle == "arithmetic":
        return [(int(np.sum(nation[region > 1].astype(np.int64) + argument)),)]
    if oracle == "independent":
        return [(int(np.where(region > 1, argument[0], argument[1]).sum()),)]
    if oracle == "having":
        addition, threshold, limit = argument
        rows = [(int(group), int(np.sum(nation[region == group].astype(np.int64) + addition)))
                for group in np.unique(region)]
        rows = sorted((row for row in rows if row[1] > threshold), key=lambda row: (-row[1], row[0]))
        return rows[:limit]
    raise ValueError("Unknown oracle: " + oracle)


def main(additional_queries=(), oracle_function=expected_rows):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/10")
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output-limit", type=int, default=32, help="CLI row window; increase for independent Top-K boundary checks")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cli-traced-" if args.traces else "cli-plain-", dir=args.output_root.resolve()))
    legacy_pch = Path("/dev/shm/jit_headers.h.gch")
    descriptor = os.open(legacy_pch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    report = {"success": False, "purpose": "correctness-only", "records": [], "traces": args.traces,
              "device_cache": "new directory per process; retained inside session; OS/HIP caches not cleared"}
    try:
        binary = args.db_cli.resolve()
        events = output / "compiler-events"
        events.mkdir()
        toolchain = output / "toolchain"
        toolchain.mkdir()
        wrapper = toolchain / "acpp"
        shutil.copyfile(Path(__file__).with_name("compiler_observer.sh"), wrapper)
        wrapper.chmod(0o700)
        config = output / "crystal.conf"
        config.write_text(f"dump_generated_code=off\nextended_timing=on\noutput_limit={args.output_limit}\n"
                          "memory_guard_enabled=on\nreuse_scratch_buffers=true\n")
        env = os.environ.copy()
        env.update({"LC_ALL": "C", "PATH": str(toolchain) + os.pathsep + env.get("PATH", ""),
                    "CRYSTAL_STAGE0_REAL_ACPP": shutil.which("acpp"), "CRYSTAL_STAGE0_EVENT_DIR": str(events),
                    "CRYSTAL_STAGE1_TRACES": "1" if args.traces else "0", "CRYSTAL_CONFIG": str(config),
                    "CRYSTAL_INCLUDE_DIR": str(ROOT / "include"), "CRYSTAL_DEPS_INCLUDE_DIR": str(ROOT / "deps/include"),
                    "ACPP_INCLUDE_DIR": str(args.acpp_include.resolve()), "ACPP_APPDB_DIR": str(output / "device-cache")})
        report.update({"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                       "environment": {key: value for key, value in env.items() if key.startswith(("CRYSTAL_", "ACPP_", "HIP_", "HSA_"))}})
        items = workload(not args.skip_manifest, args.include_heavy) + list(additional_queries)
        nonce = uuid.uuid4().hex
        commands = ["\\dump off", "\\timing on", f"\\limit {args.output_limit}"]
        for index, (_, sql, _, _) in enumerate(items):
            commands += [f"\\echo __STAGE1_{index}__", isolate_sql(sql.rstrip(";"), nonce) + ";"]
        commands += [f"\\echo __STAGE1_{len(items)}__", "\\q"]
        stdin = "\n".join(commands) + "\n"
        (output / "stdin.sql").write_text(stdin)
        command = timed_command([str(binary)], output / "process.time.json", output / "process.stderr")
        report["command"] = command
        rc, stdout, timer_stderr, timed_out = run_session(command, stdin, ROOT, env, args.timeout)
        stderr = (output / "process.stderr").read_text() + timer_stderr
        (output / "stdout.txt").write_text(stdout)
        (output / "stderr.txt").write_text(stderr)
        report.update({"returncode": rc, "timed_out": timed_out})
        if not timed_out:
            report["process_time"] = read_time(output / "process.time.json")
        timing = load_timing()
        for index, (name, sql, oracle, argument) in enumerate(items):
            text = stdout.split(f"__STAGE1_{index}__\n", 1)[-1].split(f"__STAGE1_{index + 1}__\n", 1)[0]
            record = {"id": name, "sql": sql, "oracle": oracle, "success": False}
            record.update(timing.parse_metrics(text))
            try:
                if stdout.count(f"__STAGE1_{index}__\n") != 1 or stdout.count(f"__STAGE1_{index + 1}__\n") != 1:
                    raise ValueError("Missing or duplicate CLI marker")
                if oracle == "legacy-rejection":
                    diagnostic = "Unsupported comparison shape in fast-path predicate, op=OP_GT, left=COLUMN_REF, right=LITERAL_NULL"
                    if text.strip() or diagnostic not in stderr:
                        raise ValueError("Missing documented legacy rejection")
                    record.update({"success": True, "expected_legacy_rejection": diagnostic})
                    report["records"].append(record)
                    continue
                if timing.has_sql_error(text) or timing.has_process_crash_marker(text):
                    raise ValueError("SQL/GPU error in transcript")
                columns, rows, count = parse_rows(text)
                record.update({"columns": columns, "rows": rows, "row_count": count})
                if oracle:
                    expected = oracle_function(oracle, argument, args.data.resolve())
                    record["expected"] = expected
                    if rows != expected or count != len(expected):
                        raise ValueError("Independent CPU oracle mismatch")
                record["success"] = True
            except (ValueError, IndexError) as error:
                record["error"] = str(error)
            report["records"].append(record)
        report["compiler_events"] = collect_events(events)
        report["stderr_errors"] = [line for line in stderr.splitlines() if "[ERROR]" in line or "Fatal error:" in line]
        expected_errors = ["[ERROR] Filter predicate is not translatable before JIT C++ compilation: "
                           "Unsupported comparison shape in fast-path predicate, op=OP_GT, left=COLUMN_REF, right=LITERAL_NULL"]
        report["success"] = (rc == 0 and not timed_out and all(row["success"] for row in report["records"])
                             and report["stderr_errors"] == expected_errors and not timing.has_process_crash_marker(stderr))
        print(f"Artifacts: {output}")
        print(f"Passed {sum(row['success'] for row in report['records'])}/{len(items)} CLI checks; "
              f"independent oracle: {sum(row['oracle'] not in (None, 'legacy-rejection') for row in report['records'])}")
    finally:
        if legacy_pch.exists():
            shutil.move(str(legacy_pch), output / "legacy-jit_headers.pch")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
