#!/usr/bin/env python3
## @file
## @brief Actual db_cli numeric/CASE parameter sequences with independent offline NumPy oracles.
"""Correctness runner, not a latency/SLA gate. Native timers and traces are retained."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, read_time
from sql_baseline import run_session


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def workload():
    """Define value changes separately from permitted schema/structural transitions."""
    items = []
    for index, (value, threshold) in enumerate(((0.5, 0), (-0.5, 0), (-0.5, 99), (None, 0), (2, 0), (-2, 0), (0.5, 0))):
        text = "NULL" if value is None else str(value)
        expr = f"s_nation+{text}"
        sql = f"SELECT SUM({expr}),MIN({expr}),MAX({expr}),AVG({expr}),COUNT({expr}) FROM supplier WHERE s_region>={threshold} ORDER BY 1"
        items.append((f"scalar-{index}", sql, "scalar", [value, threshold], index not in (0, 4)))
    for index, (a, b, addition, having, limit) in enumerate(((1, 1000000000, 0.5, 0, 10), (7, 7, -0.5, 0, 10), (-9007199254740993, 9007199254740993, 0.5, 0, 1), (None, None, 0.5, 0, 10), (2, 3, 0.5, 0, 0), (2, 3, 0.5, 9999999, 10))):
        leaf = lambda v: "NULL" if v is None else str(v)
        group = f"CASE WHEN s_region<2 THEN {leaf(a)} ELSE {leaf(b)} END"
        total = f"SUM(s_nation+{addition})"
        sql = f"SELECT {group},{total},COUNT(*) FROM supplier GROUP BY {group} HAVING {total}>{having} ORDER BY 1 LIMIT {limit}"
        items.append((f"case-{index}", sql, "case", [a, b, addition, having, limit], index > 0))
    for index, value in enumerate((None, 9007199254740993, None)):
        sql = f"SELECT {'NULL' if value is None else value} AS exact FROM supplier ORDER BY 1 LIMIT 1"
        items.append((f"null-first-{index}", sql, "literal", value, index == 2))
    for index, (minimum, maximum) in enumerate(((-(1 << 63), (1 << 63) - 1), (-9007199254740993, 9007199254740993), (None, None))):
        literal = lambda v: "NULL" if v is None else str(v)
        sql = f"SELECT MIN({literal(minimum)}),MAX({literal(maximum)}) FROM supplier ORDER BY 1"
        items.append((f"exact-range-{index}", sql, "exact-range", [minimum, maximum], index > 0))
    for index, threshold in enumerate((0, -1, 99999)):
        sql = f"SELECT s_region,COUNT(*) FROM supplier GROUP BY s_region HAVING COUNT(*)>{threshold} ORDER BY 1"
        items.append((f"count-threshold-{index}", sql, "count", threshold, index > 0))
    for index, addition in enumerate((-10000, 10000, None, -10000.5, -10000)):
        literal = "NULL" if addition is None else str(addition)
        sql = f"SELECT s_region,COUNT(*) FROM supplier GROUP BY s_region HAVING COUNT(*)+{literal}<0 ORDER BY 1"
        items.append((f"count-arithmetic-{index}", sql, "count-arithmetic", addition, index in (1, 2, 4)))
    return items


def expected(oracle, argument, data):
    """Run only after the measured CLI has exited. Never import a device/engine helper."""
    import numpy as np
    if oracle == "literal":
        return [[argument]]
    if oracle == "exact-range":
        return [argument]
    region = np.memmap(data / "SUPPLIER5", dtype="<i4", mode="r")
    nation = np.memmap(data / "SUPPLIER4", dtype="<i4", mode="r")
    if oracle in ("count", "count-arithmetic"):
        groups, counts = np.unique(region, return_counts=True)
        if oracle == "count-arithmetic" and argument is None:
            return []
        return [[int(group), int(count)] for group, count in zip(groups, counts)
                if (int(count) > argument if oracle == "count" else int(count) + argument < 0)]
    if oracle == "scalar":
        value, threshold = argument
        selected = nation[region >= threshold]
        if value is None or not selected.size:
            return [[None, None, None, None, 0]]
        values = selected.astype(np.int64 if isinstance(value, int) else np.float64) + value
        return [[values.sum().item(), values.min().item(), values.max().item(), values.mean().item(), int(values.size)]]
    a, b, addition, having, limit = argument
    groups = {}
    for condition, key in ((region < 2, a), (region >= 2, b)):
        values = nation[condition].astype(np.float64) + addition
        total, count = groups.get(key, (0.0, 0))
        groups[key] = (total + values.sum().item(), count + int(values.size))
    rows = [[key, total, count] for key, (total, count) in groups.items() if count and total > having]
    return sorted(rows, key=lambda row: (row[0] is None, row[0] or 0))[:limit]


def equal(actual, wanted):
    if len(actual) != len(wanted):
        return False
    for row, reference in zip(actual, wanted):
        if len(row) != len(reference):
            return False
        for a, b in zip(row, reference):
            if a is None or b is None:
                if a is not b:
                    return False
            elif isinstance(b, float):
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                    return False
            elif a != b:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/10")
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="parameters-traced-" if args.traces else "parameters-plain-", dir=args.output_root.resolve()))
    config = output / "crystal.conf"
    config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\n"
                      "memory_guard_enabled=on\nreuse_scratch_buffers=true\njit_keep_artifacts=on\n"
                      f"jit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    items = workload()
    commands = ["\\dump off", "\\timing on", "\\limit 100"]
    for index, (_, sql, _, _, _) in enumerate(items):
        commands += [f"\\echo __STAGE6_{index}__", sql + ";"]
    commands += [f"\\echo __STAGE6_{len(items)}__", "\\q"]
    stdin = "\n".join(commands) + "\n"
    (output / "stdin.sql").write_text(stdin)
    env = dict(os.environ, LC_ALL="C", TMPDIR="/tmp/opencode", CRYSTAL_CONFIG=str(config), ACPP_APPDB_DIR=str(output / "device-cache"))
    command = timed_command([args.db_cli.resolve()], output / "process.time.json", output / "process.stderr")
    (output / "command.json").write_text(json.dumps({"argv": command, "environment": {k: v for k, v in env.items() if k.startswith(("CRYSTAL_", "ACPP_", "HIP_", "HSA_"))}}, indent=2))
    rc, stdout, timer_stderr, timeout = run_session(command, stdin, ROOT, env, 2400)
    stderr = (output / "process.stderr").read_text() + timer_stderr
    (output / "stdout.txt").write_text(stdout)
    (output / "stderr.txt").write_text(stderr)
    rows = module("row_parser", "test/pcuda_stage1/validate_cli.py")
    timing = module("timing", "olap_bench/tools/run_olap_bench.py")
    records = []
    for index, (name, sql, oracle, argument, ready) in enumerate(items):
        text = stdout.split(f"__STAGE6_{index}__\n", 1)[-1].split(f"__STAGE6_{index + 1}__\n", 1)[0]
        record = {"id": name, "sql": sql, "ready": ready, "success": False}
        record.update(timing.parse_metrics(text))
        try:
            if stdout.count(f"__STAGE6_{index}__\n") != 1 or stdout.count(f"__STAGE6_{index + 1}__\n") != 1:
                raise ValueError("Missing/duplicate marker")
            if record.get("reported_timing_model") != "pcuda-modules-v1":
                raise ValueError("Wrong timing model")
            _, actual, count = rows.parse_rows(text)
            reference = expected(oracle, argument, args.data)
            record.update(rows=actual, expected=reference)
            if timing.has_sql_error(text) or count != len(reference) or not equal(actual, reference):
                raise ValueError("Independent numeric oracle mismatch")
            if ready and any(record.get("module_" + key + "_calls") != 0 for key in ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")):
                raise ValueError("Compatible values triggered new SQL module work")
            record["success"] = True
        except (ValueError, IndexError) as error:
            record["error"] = str(error)
        records.append(record)
    report = {"success": rc == 0 and not timeout and all(r["success"] for r in records), "records": records,
              "returncode": rc, "timeout": timeout, "traces": args.traces, "binary_sha256": hashlib.sha256(args.db_cli.read_bytes()).hexdigest(),
              "process_time": None if timeout else read_time(output / "process.time.json"),
              "timing_model": "pcuda-modules-v1", "summary": "direct total_query_ms; oracle after process exit; no SLA/speedup claim",
              "caches": "new process/module/device directory; retained within session; OS/HIP/driver caches not cleared"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts: {output}\nPassed {sum(r['success'] for r in records)}/{len(records)}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
