#!/usr/bin/env python3
## @file
## @brief Independent SF10 oracles for manual PCUDA examples and decoded numeric storage through db_cli.
"""Correctness runner. Native Bash time is retained; Python analyzes completed processes only."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, read_time
from sql_baseline import run_session, reference_rows


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def unexpected_diagnostics(text):
    """Accept only the documented cold-device-cache advisory, not arbitrary stderr."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    advisory = "[AdaptiveCpp Warning] kernel_cache: This application run has resulted in new binaries being JIT-compiled."
    return [line for line in text.splitlines() if line.strip() and not line.startswith(advisory)]


def manual_trials(text):
    blocks = re.findall(r"__SSB_RESULT_BEGIN__\n(.*?)__SSB_RESULT_END__", text, re.S)
    return [Counter(tuple(map(int, line.split())) for line in block.splitlines() if line.strip()) for block in blocks]


def process(binary, arguments, stdin, directory, config=None, merge_diagnostics=False):
    """Keep commands, outputs, native timers and a private device cache for every process."""
    directory.mkdir()
    environment = dict(os.environ, LC_ALL="C", TMPDIR="/tmp/opencode", ACPP_APPDB_DIR=str(directory / "device-cache"))
    if config:
        environment["CRYSTAL_CONFIG"] = str(config)
    application = [binary, *arguments]
    if merge_diagnostics:
        # Native exec, not a Python timer/proxy: keep a SQL error between its session markers.
        application = ["bash", "-c", 'exec "$@" 2>&1', "stage7-cli", *application]
    command = timed_command(application, directory / "process.time.json", directory / "process.stderr")
    (directory / "command.json").write_text(json.dumps({"argv": command, "environment": {k: v for k, v in environment.items()
        if k.startswith(("ACPP_", "CRYSTAL_", "HIP_", "HSA_"))}}, indent=2))
    (directory / "stdin.sql").write_text(stdin)
    rc, stdout, timer_stderr, timeout = run_session(command, stdin, ROOT, environment, 2400)
    stderr = (directory / "process.stderr").read_text() + timer_stderr
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    return {"returncode": rc, "timeout": timeout, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "native_process_ms": None if timeout else read_time(directory / "process.time.json")}, stdout, stderr


def manual(args, output):
    records = []
    for query in (11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43):
        record, stdout, stderr = process(args.build / f"ssb_q{query}", [str(args.data), str(args.trials)], "", output / f"q{query}")
        text = (ROOT / f"test/queries/result/scale10/{query}.txt").read_text()
        expected = reference_rows(text)
        actual = manual_trials(stdout)
        diagnostics = unexpected_diagnostics(stderr)
        record.update(id=f"q{query}", trials=len(actual), expected_rows=sum(expected.values()), unexpected_diagnostics=diagnostics,
                      matches_reference=[rows == expected for rows in actual],
                      rows=[sorted(rows.elements()) for rows in actual], reference=str(ROOT / f"test/queries/result/scale10/{query}.txt"))
        record["success"] = record["returncode"] == 0 and not record["timeout"] and not diagnostics and len(actual) == args.trials and all(rows == expected for rows in actual)
        records.append(record)
    return {"success": all(record["success"] for record in records), "records": records,
            "checks": len(records) * args.trials, "oracle": "existing independent SF10 result files; compare multisets, preserve duplicate rows",
            "q11_boundary": "manual/reference use 1993 only; SQL file additionally includes 1994-01-01, checked separately"}


def storage_workload():
    items = []
    for column in ("d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"):
        items.append((column, f"SELECT COUNT(*),SUM({column}),MIN({column}),MAX({column}),AVG({column}) FROM ddate ORDER BY 1", "flag", column, False))
    items += [(f"flag-filter-{index}", f"SELECT COUNT(*) FROM ddate WHERE d_weekdayfl={value} ORDER BY 1", "filter", value, index > 0)
              for index, value in enumerate(("1", "0", "NULL", "1"))]
    items += [
        ("flag-groups", "SELECT d_weekdayfl,COUNT(*),SUM(d_year) FROM ddate GROUP BY d_weekdayfl ORDER BY 1", "groups", None, False),
        ("date-numbers", "SELECT d_monthnuminyear,d_weeknuminyear,COUNT(*) FROM ddate GROUP BY d_monthnuminyear,d_weeknuminyear ORDER BY 1,2", "calendar", None, False),
        ("byte-fact", "SELECT COUNT(*),SUM(lo_shippriority),MIN(lo_shippriority),MAX(lo_shippriority) FROM lineorder ORDER BY 1", "fact", None, False),
        ("projection", "SELECT d_datekey,d_weekdayfl,d_holidayfl FROM ddate ORDER BY d_datekey LIMIT 7", "projection", None, False),
    ]
    deferred_paths = ("03_projection/proj_05_select_star_small_dimension.sql", "03_projection/proj_join_02_star_customer.sql",
                      "08_custom/custom2.sql", "10_stress/stress_01_wide_star_customer.sql", "10_stress/stress_02_wide_star_customer_supplier.sql")
    # Locate manifest-owned filenames instead of changing their current/current_heavy status.
    deferred_sql = []
    for relative in deferred_paths:
        filename = Path(relative).name
        matches = list((ROOT / "olap_bench/sql").rglob(filename))
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous deferred manifest SQL: " + filename)
        deferred_sql.append(matches[0].read_text())
    for index, sql in enumerate(("SELECT s_name FROM supplier", "SELECT COUNT(c_mktsegment) FROM customer", "SELECT * FROM ddate", *deferred_sql)):
        items.append((f"text-rejected-{index}", re.sub(r"--[^\n]*", "", sql).strip().rstrip(";"), "unsupported", None, False))
    items.append(("recovery-repeat", "SELECT COUNT(*) FROM ddate WHERE d_weekdayfl=1 ORDER BY 1", "filter", "1", True))
    return items


def storage_expected(kind, argument, directory):
    """Decode ASCII bytes and physical indexes directly, without importing the engine schema."""
    import numpy as np
    physical = {"d_lastdayinweekfl": "DDATE13", "d_lastdayinmonthfl": "DDATE14", "d_holidayfl": "DDATE15", "d_weekdayfl": "DDATE16"}
    def flags(filename):
        values = np.fromfile(directory / filename, dtype=np.uint8)
        if not np.all((values == ord("0")) | (values == ord("1"))):
            raise ValueError("Invalid independent flag source")
        return values.astype(np.int64) - ord("0")
    weekdays = flags("DDATE16")
    years = np.fromfile(directory / "DDATE4", dtype="<i4").astype(np.int64)
    if kind == "flag":
        values = flags(physical[argument])
        return [[len(values), int(values.sum()), int(values.min()), int(values.max()), float(values.mean())]]
    if kind == "filter":
        return [[0 if argument == "NULL" else int(np.count_nonzero(weekdays == int(argument)))]]
    if kind == "groups":
        return [[value, int(np.count_nonzero(weekdays == value)), int(years[weekdays == value].sum())] for value in (0, 1)]
    if kind == "calendar":
        month = np.fromfile(directory / "DDATE10", dtype="<i4")
        week = np.fromfile(directory / "DDATE11", dtype="<i4")
        counts = Counter(zip(map(int, month), map(int, week)))
        return [[m, w, count] for (m, w), count in sorted(counts.items())]
    if kind == "fact":
        values = flags("LINEORDER7")
        return [[len(values), int(values.sum()), int(values.min()), int(values.max())]]
    if kind == "projection":
        keys = np.fromfile(directory / "DDATE0", dtype="<i4")
        holiday = flags("DDATE15")
        return [[int(keys[i]), int(weekdays[i]), int(holiday[i])] for i in np.argsort(keys)[:7]]
    raise ValueError("Unknown storage oracle")


def storage(args, output):
    config = output / "crystal.conf"
    config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\nmemory_guard_enabled=on\nreuse_scratch_buffers=true\n"
                      f"jit_keep_artifacts=on\njit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    items = storage_workload()
    commands = ["\\dump off", "\\timing on", "\\limit 5000"]
    for index, (_, sql, _, _, _) in enumerate(items):
        commands += [f"\\echo __STAGE7_{index}__", sql + ";"]
    commands += [f"\\echo __STAGE7_{len(items)}__", "\\q"]
    session, stdout, stderr = process(args.build / "db_cli", [], "\n".join(commands) + "\n", output / "session", config, True)
    parser = module("stage7_rows", "test/pcuda_stage1/validate_cli.py")
    timing = module("stage7_timing", "olap_bench/tools/run_olap_bench.py")
    compare = module("stage7_equal", "test/pcuda_stage6/validate_cli.py").equal
    records = []
    for index, (name, sql, kind, argument, ready) in enumerate(items):
        text = stdout.split(f"__STAGE7_{index}__\n", 1)[-1].split(f"__STAGE7_{index + 1}__\n", 1)[0]
        record = dict(id=name, sql=sql, ready=ready, success=False, status="unverified")
        record.update(timing.parse_metrics(text))
        try:
            if stdout.count(f"__STAGE7_{index}__\n") != 1 or stdout.count(f"__STAGE7_{index + 1}__\n") != 1:
                raise ValueError("Missing/duplicate session marker")
            if kind == "unsupported":
                if "7-TEXT" not in text or "--- Result ---" in text:
                    raise ValueError("Text was not explicitly rejected before numeric execution")
                record.update(status="expected-unsupported", success=True)
            else:
                if record.get("reported_timing_model") != args.timing_model or timing.has_sql_error(text):
                    raise ValueError("SQL/timing diagnostic")
                _, actual, count = parser.parse_rows(text)
                expected = storage_expected(kind, argument, args.data)
                record.update(rows=actual, expected=expected)
                if count != len(expected) or not compare(actual, expected):
                    raise ValueError("Independent numeric storage oracle mismatch")
                if ready and any(record.get("module_" + key + "_calls") != 0 for key in ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")):
                    raise ValueError("Compatible values triggered codegen/build/load")
                record.update(status="executed", success=True)
        except (ValueError, IndexError) as error:
            record["error"] = str(error)
        records.append(record)
    return {**session, "success": session["returncode"] == 0 and not session["timeout"] and all(record["success"] for record in records),
            "records": records, "timing_model": args.timing_model, "traces": args.traces,
            "executed": sum(record["status"] == "executed" for record in records),
            "expected_unsupported": sum(record["status"] == "expected-unsupported" for record in records),
            "stderr": stderr, "oracle": "offline NumPy ASCII-byte/LE Int32 physical files; never numeric raw-prefix text"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manual", "storage"))
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/10")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--timing-model", choices=("pcuda-modules-v1",), required=True)
    args = parser.parse_args()
    args.build, args.data = args.build.resolve(), args.data.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.mode + ("-traced-" if args.traces else "-plain-"), dir=args.output_root.resolve()))
    report = manual(args, output) if args.mode == "manual" else storage(args, output)
    report.update(caches="new process/device directory; retained within process; OS/HIP/driver caches not cleared",
                  measurement="correctness only; native process wall is not summary or time to first result; no SQL SLA claim")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts: {output}\nSuccess: {report['success']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
