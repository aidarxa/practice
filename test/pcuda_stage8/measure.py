#!/usr/bin/env python3
## @file
## @brief Sequential independent-process SSB acceptance; all timers live in native Bash/C++ clients.
"""Measure first; analyze only after the entire rotated plain/traced series has finished."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, read_time, trace_summary
from sql_baseline import REFERENCE_GROUPS, reference_rows, cli_rows, run_session


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def workload(query, runs):
    sql = (ROOT / f"test/queries/sql/{query}.sql").read_text().strip()
    items = [dict(phase="cold" if i == 0 else "warm", sql=sql) for i in range(runs)]
    if query == "q33":
        changed = sql.replace("231", "235").replace(">=1992", ">=1993").replace("<= 1997", "<= 1996")
        items += [dict(phase="changed", sql=changed) for _ in range(runs - 1)]
    return items


def measure(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="ssb-series-", dir=args.output_root.resolve()))
    queries = args.query or list(REFERENCE_GROUPS)
    env = dict(os.environ, LC_ALL="C", TMPDIR="/tmp/opencode", ACPP_ADAPTIVITY_LEVEL="1", ACPP_ALLOCATION_TRACKING="1")
    records = []
    print(output, flush=True)
    for session in range(args.sessions):
        rotated = queries[session % len(queries):] + queries[:session % len(queries)]
        for query in rotated:
            for traced in ((False, True) if session % 2 == 0 else (True, False)):
                directory = output / f"{session}-{query}-{'traced' if traced else 'plain'}"
                directory.mkdir()
                config = directory / "crystal.conf"
                config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\n"
                                  "memory_guard_enabled=on\nreuse_scratch_buffers=true\njit_keep_artifacts=on\njit_jobs=4\n"
                                  f"jit_artifact_dir={directory}\njit_traces={'on' if traced else 'off'}\n")
                items = workload(query, args.runs)
                commands = ["\\dump off", "\\timing on", "\\limit all"]
                for i, item in enumerate(items):
                    commands += [f"\\echo __ACCEPTANCE_{i}__", item["sql"]]
                commands += [f"\\echo __ACCEPTANCE_{len(items)}__", "\\q"]
                stdin = "\n".join(commands) + "\n"
                (directory / "stdin.sql").write_text(stdin)
                child_env = dict(env, CRYSTAL_CONFIG=str(config), ACPP_APPDB_DIR=str(directory / "device-cache"))
                argv = timed_command([args.build.resolve() / "pcuda_acceptance_first_result", directory / "first-result.json",
                                      args.build.resolve() / "db_cli"], directory / "process.time.json", directory / "process.stderr")
                record = dict(query=query, session=session, traces=traced, directory=directory.name, argv=argv,
                              environment={k:v for k,v in child_env.items() if k.startswith(("CRYSTAL_","ACPP_","HIP_","HSA_")) or k in ("LC_ALL","TMPDIR")})
                (directory / "command.json").write_text(json.dumps(record, indent=2) + "\n")
                rc, stdout, stderr, timeout = run_session(argv, stdin, ROOT, child_env, 600)
                (directory / "stdout.txt").write_text(stdout)
                (directory / "timer.stderr").write_text(stderr)
                record.update(returncode=rc, timeout=timeout, items=items)
                records.append(record)
                (output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
                print(directory.name, rc, flush=True)
    # Raw metadata only: no CPU oracle, trace scan, statistical filtering or clock in this organizer.
    report = dict(sessions=args.sessions, runs=args.runs, queries=queries, order=records,
                  binary_sha256=hashlib.sha256((args.build / "db_cli").read_bytes()).hexdigest(),
                  timing_model=args.timing_model, caches="New process/module/device cache per query/session/mode; retained inside warm/changed; OS/HIP/driver caches not cleared")
    (output / "series.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(r["returncode"] == 0 and not r["timeout"] for r in records) else 1


EVENTS = ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")


def check_record(record, first, expected_model="pcuda-modules-v1"):
    """Fail closed on missing/new intervals, even when a short-looking total is present."""
    for name in ("sql_to_pcuda_ms", "total_query_ms", "generated_execute_ms", "host_fetch_ms", "build_elapsed_ms"):
        value = record.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/invalid direct interval: " + name)
    if record.get("reported_timing_model") != expected_model:
        raise ValueError("Incorrect timing model")
    if first:
        if not all(record.get("module_" + name + "_calls", 0) > 0 for name in EVENTS):
            raise ValueError("An independent cold process did not prepare its SQL modules")
    elif any(record.get("module_" + name + "_calls") != 0 for name in EVENTS):
        raise ValueError("Ready/changed values triggered codegen/build/load/symbol resolution")
    # Deliberately not added to prepare/codegen: this is an overlapping observation.
    if record["sql_to_pcuda_ms"] > record["total_query_ms"] + 0.003:
        raise ValueError("Frontend exceeds the continuous query interval")


def analyze(args):
    source = json.loads((args.series / "series.json").read_text())
    runner = module("acceptance_metrics", "olap_bench/tools/run_olap_bench.py")
    quality = module("acceptance_diagnostics", "test/pcuda_stage7/validate.py")
    stats = module("acceptance_statistics", "test/pcuda_stage5/analyze_series.py").distribution
    oracle = json.loads(args.q11_oracle.read_text())
    results = []
    for session in source["order"]:
        directory = args.series / session["directory"]
        stdout = (directory / "stdout.txt").read_text()
        stderr = (directory / "process.stderr").read_text() + (directory / "timer.stderr").read_text()
        rows = []
        for i, item in enumerate(session["items"]):
            text = stdout.split(f"__ACCEPTANCE_{i}__\n", 1)[-1].split(f"__ACCEPTANCE_{i+1}__\n", 1)[0]
            record = dict(index=i, **item, **runner.parse_metrics(text), accepted=False)
            try:
                if any(stdout.count(f"__ACCEPTANCE_{n}__\n") != 1 for n in (i, i+1)) or runner.has_sql_error(text):
                    raise ValueError("Missing marker or SQL error")
                check_record(record, i == 0)
                query = session["query"]
                actual = cli_rows(text, REFERENCE_GROUPS[query])
                expected = reference_rows((ROOT / f"test/queries/result/scale10/{query[1:]}.txt").read_text())
                if item["phase"] == "changed":
                    expected = Counter({row:count for row,count in expected.items() if 1993 <= row[0] <= 1996 and row[1] == 235 and row[2] == 235})
                record["file_matches"] = actual == expected
                record["rows"] = [list(row) for row in actual.elements()]
                if query == "q11":
                    record["cpu_oracle_matches"] = actual == Counter({(oracle["sql_q11"],):1})
                    if expected != Counter({(oracle["year_1993"],):1}) or not record["cpu_oracle_matches"] or oracle["sql_q11"] != oracle["year_1993"] + oracle["january_1_1994"]:
                        raise ValueError("Independent Q1.1 boundary oracle mismatch")
                elif not record["file_matches"]:
                    raise ValueError("SF10 file mismatch")
                record["accepted"] = True
            except (ValueError, KeyError) as error:
                record["error"] = str(error)
            rows.append(record)
        result = dict(query=session["query"], session=session["session"], traces=session["traces"], directory=session["directory"], records=rows,
                      unexpected_diagnostics=quality.unexpected_diagnostics(stderr),
                      process_time=read_time(directory / "process.time.json"),
                      first_result=json.loads((directory / "first-result.json").read_text()))
        result["traces_ms"] = {str(p.relative_to(directory)):trace_summary(p) for p in directory.glob("db-pcuda-*/*/*.json") if p.name.endswith((".clang.json", ".lld.json"))}
        result["accepted"] = session["returncode"] == 0 and not session["timeout"] and not result["unexpected_diagnostics"] and all(r["accepted"] for r in rows)
        results.append(result)
    aggregate = []
    for query in source["queries"]:
        for traced in (False, True):
            sessions = [s for s in results if s["query"] == query and s["traces"] == traced]
            for phase in ("cold", "warm", "changed"):
                rows = [r for s in sessions for r in s["records"] if r["phase"] == phase]
                if not rows:
                    continue
                fields = ["total_query_ms", "sql_to_pcuda_ms", "generated_execute_ms", "host_fetch_ms", "library_load_ms", *runner.MODULE_TIMING_FIELDS]
                entry = dict(query=query, traces=traced, phase=phase, accepted=all(s["accepted"] for s in sessions),
                             metrics={field:stats([r[field] for r in rows]) for field in fields})
                if phase == "cold":
                    entry["startup"] = {field:stats([s["records"][0][field] for s in sessions]) for field in runner.OBSERVATION_TIMING_FIELDS if field != "sql_to_pcuda_ms"}
                    entry["spawn_to_first_ms"] = stats([s["first_result"]["process_start_to_first_result_ms"] for s in sessions])
                aggregate.append(entry)
    report = dict(success=all(s["accepted"] for s in results), sessions=results, aggregate=aggregate,
                  confirmation_sample=source["sessions"] >= 7 and source["runs"] >= 9,
                  file_mismatches=sum(not r.get("file_matches", False) for s in results for r in s["records"]),
                  note="Q1.1 file comparison remains false, separately checked by a freshly run independent CPU oracle; no SLA implied by accepted correctness")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Correctness/counters: {report['success']}; independent cold processes: {len(results)}; file mismatches: {report['file_mismatches']}")
    for item in aggregate:
        if not item["traces"]:
            print(item["query"], item["phase"], {k:round(v["median"], 3) for k,v in item["metrics"].items() if k in ("total_query_ms", "sql_to_pcuda_ms", "generated_execute_ms", "build_elapsed_ms")})
    return 0 if report["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("measure")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--sessions", type=int, default=7)
    run.add_argument("--runs", type=int, default=9)
    run.add_argument("--query", choices=tuple(REFERENCE_GROUPS), action="append")
    run.add_argument("--timing-model", choices=("pcuda-modules-v1",), required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--series", type=Path, required=True)
    analysis.add_argument("--q11-oracle", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "measure" and (args.sessions < 1 or args.runs < 2):
        parser.error("Require positive sessions and at least one warm execution")
    return measure(args) if args.mode == "measure" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
