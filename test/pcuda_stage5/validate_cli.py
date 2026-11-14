#!/usr/bin/env python3
## @file
## @brief PCUDA-module SQL validation; Python organizes processes and analyzes only completed transcripts.
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
from native_timing import timed_command, read_time, trace_summary
from sql_baseline import REFERENCE_GROUPS, reference_rows, cli_rows, run_session


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workload(mode, runs, changed_parameters=False):
    if changed_parameters and (mode != "performance" or runs < 2):
        raise ValueError("Changed-parameter measurements require performance mode and at least two runs")
    old = module_at("stage1_oracles", "test/pcuda_stage1/validate_cli.py")
    if mode == "manifest":
        items = old.workload(True, True)
        items = [(name, sql, "null-zero" if oracle == "legacy-rejection" else oracle, arg) for name, sql, oracle, arg in items]
        extra = module_at("stage2_oracles", "test/pcuda_stage2/validate_cli.py")
        return items + extra.boundary_queries()
    queries = list(REFERENCE_GROUPS) if mode == "ssb" else ["q33"]
    items = [(f"{query}-{run}", (ROOT / f"test/queries/sql/{query}.sql").read_text(),
              "ssb" if mode == "ssb" else None, query)
             for query in queries for run in range(runs)]
    if changed_parameters:
        sql = (ROOT / "test/queries/sql/q33.sql").read_text().replace("231", "235").replace(">=1992", ">=1993").replace("<= 1997", "<= 1996")
        items += [(f"q33-parameters-{run}", sql, None, "q33") for run in range(runs - 1)]
    return items


def reuse_verified(records):
    """Check all ready repeats, including the very first execution with changed ordinary values."""
    ready = [record for record in records if "-parameters-" in record["id"] or int(record["id"].split("-")[-1]) > 0]
    return all(record.get("module_" + key + "_calls") == 0 for record in ready
               for key in ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--mode", choices=("manifest", "ssb", "performance"), required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--changed-parameters", action="store_true")
    parser.add_argument("--timing-model",choices=("pcuda-modules-v1","pcuda-modules-overlap-exp-v1"),default="pcuda-modules-v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/10")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.mode + ("-traced-" if args.traces else "-plain-"), dir=args.output_root.resolve()))
    config = output / "crystal.conf"
    config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\n"
                      "memory_guard_enabled=on\nreuse_scratch_buffers=true\njit_keep_artifacts=on\n"
                      f"jit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    items = workload(args.mode,args.runs,args.changed_parameters)
    commands = ["\\dump off", "\\timing on", "\\limit 5000"]
    for i, (_, sql, _, _) in enumerate(items):
        commands += [f"\\echo __STAGE5_{i}__", re.sub(r"--[^\n]*", "", sql).strip().rstrip(";") + ";"]
    commands += [f"\\echo __STAGE5_{len(items)}__", "\\q"]
    stdin = "\n".join(commands) + "\n"
    (output / "stdin.sql").write_text(stdin)
    env = dict(os.environ, LC_ALL="C", CRYSTAL_CONFIG=str(config), ACPP_APPDB_DIR=str(output / "device-cache"), TMPDIR="/tmp/opencode")
    env["CRYSTAL_JIT_OVERLAP_EXPERIMENT"]="on" if args.timing_model=="pcuda-modules-overlap-exp-v1" else "off"
    command = timed_command([args.db_cli.resolve()],output / "process.time.json",output / "process.stderr")
    (output / "command.json").write_text(json.dumps({"argv":command,"environment":{k:v for k,v in env.items() if k.startswith(("CRYSTAL_","ACPP_","HIP_","HSA_"))}},indent=2))
    rc, stdout, timer_stderr, timeout = run_session(command,stdin,ROOT,env,2400)
    stderr = (output / "process.stderr").read_text() + timer_stderr
    (output / "stdout.txt").write_text(stdout)
    (output / "stderr.txt").write_text(stderr)
    timing = module_at("module_timing","olap_bench/tools/run_olap_bench.py")
    old = module_at("stage1_oracles","test/pcuda_stage1/validate_cli.py")
    boundary = module_at("stage2_oracles","test/pcuda_stage2/validate_cli.py")
    records = []
    for i, (name, sql, oracle, arg) in enumerate(items):
        record = {"id":name,"sql":sql,"oracle":oracle,"success":False}
        text = stdout.split(f"__STAGE5_{i}__\n",1)[-1].split(f"__STAGE5_{i+1}__\n",1)[0]
        record.update(timing.parse_metrics(text))
        try:
            if stdout.count(f"__STAGE5_{i}__\n") != 1 or stdout.count(f"__STAGE5_{i+1}__\n") != 1:
                raise ValueError("Missing/duplicate session marker")
            if record.get("reported_timing_model") != args.timing_model:
                raise ValueError("Wrong timing model")
            columns, rows, count = old.parse_rows(text)
            record.update(columns=columns,rows=rows,row_count=count)
            if oracle == "ssb":
                actual = cli_rows(text,REFERENCE_GROUPS[arg])
                expected = reference_rows((ROOT / f"test/queries/result/scale10/{arg[1:]}.txt").read_text())
                if actual != expected:
                    raise ValueError("SF10 reference mismatch")
            elif oracle:
                expected = [(0,)] if oracle == "null-zero" else boundary.expected_rows(oracle,arg,args.data.resolve())
                record["expected"] = expected
                if rows != expected or count != len(expected):
                    raise ValueError("Independent CPU oracle mismatch")
            if timing.has_sql_error(text):
                raise ValueError("SQL error in transcript")
            record["success"] = True
        except (ValueError, IndexError) as error:
            record["error"] = str(error)
        records.append(record)
    no_build = reuse_verified(records) if args.mode != "manifest" else True
    traces = {str(path.relative_to(output)):trace_summary(path) for path in sorted(output.glob("db-pcuda-*/*/*.json")) if path.name.endswith((".clang.json",".lld.json"))}
    errors = [line for line in stderr.splitlines() if "[ERROR]" in line or "Fatal error:" in line]
    report = {"success":rc==0 and not timeout and not errors and all(r["success"] for r in records) and no_build,
              "mode":args.mode,"timing_model":args.timing_model,"traces":args.traces,"changed_parameters":args.changed_parameters,"returncode":rc,"timeout":timeout,
              "binary_sha256":hashlib.sha256(args.db_cli.read_bytes()).hexdigest(),"records":records,"stderr_errors":errors,
              "warm_reuse_verified":no_build,"process_time":read_time(output/"process.time.json") if not timeout else None,
              "trace_intervals_ms":traces,"caches":"new process/module/device directory; device cache retained within process; OS/HIP/driver caches not cleared",
              "summary":"direct total_query_ms after data ready to CPU QueryResult; no oracle/printing inside; native process time is NOT time to first result"}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Artifacts: {output}\nPassed {sum(r['success'] for r in records)}/{len(records)}; warm reuse: {no_build}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
