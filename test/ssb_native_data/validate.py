#!/usr/bin/env python3
"""Validate native SQL/manual SSB at any generated scale; analyze only completed processes.

The independent C++ oracle is run after GPU clients. Optional --tables verifies
every physical column against original dbgen fields. Historical scale10 numeric
reference files and reports are deliberately not changed or used as Text oracles.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, read_time
from sql_baseline import run_session

QUERIES = ("q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33", "q34", "q41", "q42", "q43")
GROUPS = {
    "q11": (), "q12": (), "q13": (), "q21": ("d_year", "p_brand1"), "q22": ("d_year", "p_brand1"), "q23": ("d_year", "p_brand1"),
    "q31": ("d_year", "c_nation", "s_nation"), "q32": ("d_year", "c_city", "s_city"), "q33": ("d_year", "c_city", "s_city"),
    "q34": ("d_year", "c_city", "s_city"), "q41": ("d_year", "c_nation"), "q42": ("d_year", "s_nation", "p_category"),
    "q43": ("d_year", "s_city", "p_brand1")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PARSER = module("native_typed_rows", "test/pcuda_stage1/validate_cli.py")
TIMING = module("native_module_metrics", "olap_bench/tools/run_olap_bench.py")
DIAGNOSTICS = module("native_diagnostics", "test/pcuda_stage7/validate.py")


def canonical(query, columns, rows):
    """Reorder keys and measure, retaining their actual Text/integer/NULL types."""
    keys = [columns.index(name) for name in GROUPS[query]]
    measure = [index for index in range(len(columns)) if index not in keys]
    if len(measure) != 1:
        raise ValueError("Invalid SSB result schema")
    return [tuple(row[index] for index in keys + measure) for row in rows]


def process(argv, stdin, directory, env):
    directory.mkdir()
    binary_hash = hashlib.sha256(Path(argv[0]).read_bytes()).hexdigest()
    command = timed_command(argv, directory / "native.time", directory / "native.stderr")
    (directory / "command.json").write_text(json.dumps(dict(argv=command, binary_sha256=binary_hash, environment={key: value for key, value in env.items()
        if key.startswith(("CRYSTAL_", "ACPP_", "HIP_", "HSA_"))}), indent=2) + "\n")
    (directory / "stdin.sql").write_text(stdin)
    rc, stdout, timer_stderr, timeout = run_session(command, stdin, ROOT, env, 3600)
    stderr = (directory / "native.stderr").read_text() + timer_stderr
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    return dict(returncode=rc, timeout=timeout, binary_sha256=binary_hash, native_process_time=read_time(directory / "native.time") if not timeout else None,
                diagnostics=DIAGNOSTICS.unexpected_diagnostics(stderr)), stdout, stderr


def workload(trials, manifest):
    items = []
    for query in QUERIES:
        sql = (ROOT / "test/queries/sql" / (query + ".sql")).read_text().strip().rstrip(";")
        for trial in range(trials):
            items.append(dict(id=f"{query}-{trial}", query=query, sql=sql, ready=trial > 0))
    if manifest:
        source = json.loads((ROOT / "olap_bench/manifest.json").read_text())
        for category in source["classes"]:
            for query in category["queries"]:
                if query["status"] not in ("current", "current_heavy"):
                    continue
                sql = (ROOT / "olap_bench" / query["path"]).read_text()
                for index, statement in enumerate(re.sub(r"--[^\n]*", "", sql).split(";")):
                    if statement.strip():
                        items.append(dict(id=f"{query['id']}-{index}", query=None, sql=statement.strip(), ready=False))
        for index, sql in enumerate(targeted_queries()):
            items.append(dict(id=f"native-targeted-{index}", query=None, sql=sql, ready=False, dimension_oracle=True))
    return items


def targeted_queries():
    """Parameter, exact-number, NULL/3VL, Text grouping and numeric HAVING checks."""
    items = [f"SELECT COUNT(*) FROM supplier WHERE s_suppkey>{value}" for value in (1, 2, 99)]
    items += ["SELECT COUNT(*) FROM supplier WHERE s_region>NULL", "SELECT COUNT(*) FROM supplier WHERE s_region=NULL OR s_region>'ASIA'",
              "SELECT SUM(CASE WHEN s_region=NULL THEN 1 ELSE 0 END) FROM supplier", "SELECT 9007199254740993 FROM supplier LIMIT 1",
              "SELECT 1.25 FROM supplier LIMIT 1"]
    items += [f"SELECT SUM(CASE WHEN s_region>'ASIA' THEN s_suppkey+{value} ELSE 0 END) FROM supplier" for value in (3, 7)]
    items += [f"SELECT SUM(CASE WHEN s_region>'ASIA' THEN {first} ELSE {last} END) FROM supplier" for first, last in ((10, 10), (11, 13))]
    for addition, threshold, limit in ((3, 0, 3), (7, 40000, 5), (3, 0, 0)):
        items.append(f"SELECT s_region,SUM(s_suppkey+{addition}) AS total FROM supplier GROUP BY s_region HAVING total>{threshold} ORDER BY total DESC,s_region LIMIT {limit}")
    items += ["SELECT SUM(s_suppkey) AS total,s_region FROM supplier GROUP BY s_region ORDER BY 1 DESC,2 LIMIT 2"]
    items += [f"SELECT s_nation,s_region FROM supplier WHERE s_suppkey>0 ORDER BY s_nation,s_region LIMIT {limit}" for limit in (2049, 4096, 4096)]
    return items + items[:3]


def sql_records(items, stdout, reference, timing_model, dimension_database=None):
    records = []
    for index, item in enumerate(items):
        begin, end = f"__NATIVE_SSB_{index}__\n", f"__NATIVE_SSB_{index+1}__\n"
        record = dict(**item, success=False)
        try:
            if stdout.count(begin) != 1 or stdout.count(end) != 1:
                raise ValueError("Missing or duplicate session marker")
            text = stdout.split(begin, 1)[1].split(end, 1)[0]
            record.update(TIMING.parse_metrics(text))
            if TIMING.has_sql_error(text) or record.get("reported_timing_model") != timing_model:
                raise ValueError("SQL/timing diagnostic")
            columns, rows, count = PARSER.parse_rows(text)
            record.update(columns=columns, rows=rows, row_count=count)
            if item["query"]:
                actual = canonical(item["query"], columns, rows)
                expected = reference["sql"][item["query"]]
                record["expected"] = expected
                if count != len(expected) or Counter(actual) != Counter(map(tuple, expected)):
                    raise ValueError("Independent native SSB oracle mismatch")
            elif item.get("dimension_oracle"):
                expected = dimension_database.execute(item["sql"]).fetchall()
                record["expected"] = expected
                if rows != expected or count != len(expected):
                    raise ValueError("Independent dimension SQLite mismatch")
            if item["ready"] and any(record.get("module_" + key + "_calls") != 0 for key in
                                    ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")):
                raise ValueError("Immediate repeat caused module work")
            record["success"] = True
        except (ValueError, KeyError, IndexError) as error:
            record["error"] = str(error)
        records.append(record)
    return records


def validate_inputs(build, data, sql_only=False, tables=None):
    """Check all selected clients and inputs before creating output or starting SQL/GPU work."""
    names = ["db_cli", "bench/loader/ssb_reference"]
    if not sql_only:
        names += ["ssb_" + query for query in QUERIES]
    missing = [build / name for name in names if not (build / name).is_file() or not os.access(build / name, os.X_OK)]
    if missing:
        message = "Missing or non-executable validation binaries:\n" + "\n".join("  " + str(path) for path in missing)
        message += "\nBuild all targets in the selected GPU CMake build first:\n  cmake --build " + shlex.quote(str(build)) + " -j4"
        if not sql_only:
            message += "\nThe default runs SQL and manual SSB. --sql-only skips manual clients, but still requires db_cli and ssb_reference."
        raise ValueError(message)
    manifest = data / "dataset.manifest"
    if not manifest.is_file() or not os.access(manifest, os.R_OK):
        raise ValueError("Missing or unreadable native dataset manifest: " + str(manifest) +
                         "\nSelect a generated dataset with --data; prepare one with bash bench/prepare_ssb.sh.")
    if tables is not None:
        missing = [tables / name for name in ("lineorder.tbl", "supplier.tbl", "customer.tbl", "part.tbl", "date.tbl")
                   if not (tables / name).is_file() or not os.access(tables / name, os.R_OK)]
        if missing:
            raise ValueError("Missing or unreadable original dbgen tables:\n" + "\n".join("  " + str(path) for path in missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ssb/sf10")
    parser.add_argument("--tables", type=Path, help="Optional original .tbl directory: verify all 58 columns byte-for-byte")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/ssb-native-validation")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--manifest", action="store_true", help="Also execute current/current_heavy manifest statements")
    parser.add_argument("--sql-only", action="store_true", help="Skip manual SSB; db_cli and the independent ssb_reference oracle are still required")
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--timing-model", choices=("pcuda-modules-v1", "pcuda-modules-overlap-exp-v1"), default="pcuda-modules-v1")
    args = parser.parse_args()
    if not 1 <= args.trials <= 1000:
        parser.error("--trials must be 1..1000")
    build, data = args.build.resolve(), args.data.resolve()
    try:
        validate_inputs(build, data, args.sql_only, args.tables)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="native-", dir=args.output_root.resolve()))
    manifest = data / "dataset.manifest"
    config = output / "crystal.conf"
    config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\nmemory_guard_enabled=on\nreuse_scratch_buffers=true\n"
                      f"jit_keep_artifacts=on\njit_artifact_dir={output}\njit_traces={'on' if args.traces else 'off'}\n")
    env = dict(os.environ, LC_ALL="C", CRYSTAL_DATA_DIR=str(data), CRYSTAL_CONFIG=str(config),
               CRYSTAL_JIT_OVERLAP_EXPERIMENT="on" if args.timing_model.endswith("exp-v1") else "off", TMPDIR="/tmp/opencode")
    items = workload(args.trials, args.manifest)
    commands = ["\\dump off", "\\timing on", "\\limit 10000"]
    for index, item in enumerate(items):
        commands += [f"\\echo __NATIVE_SSB_{index}__", item["sql"] + ";"]
    commands += [f"\\echo __NATIVE_SSB_{len(items)}__", "\\q"]
    env["ACPP_APPDB_DIR"] = str(output / "sql-device-cache")
    session, stdout, _ = process([str(build / "db_cli")], "\n".join(commands) + "\n", output / "sql", env)
    manual_raw = []
    if not args.sql_only:
        for query in QUERIES:
            env["ACPP_APPDB_DIR"] = str(output / (query + "-device-cache"))
            status, text, _ = process([str(build / ("ssb_" + query)), str(data), str(args.trials)], "", output / query, env)
            manual_raw.append((query, status, text))
    # No Python/CPU oracle/trace analysis takes place inside the native measured chain.
    argv = [str(build / "bench/loader/ssb_reference"), str(data)]
    if args.tables:
        argv.append(str(args.tables.resolve()))
    oracle = subprocess.run(argv, text=True, capture_output=True, check=False)
    (output / "oracle.command.json").write_text(json.dumps(dict(argv=argv, returncode=oracle.returncode), indent=2))
    (output / "oracle.stdout").write_text(oracle.stdout)
    (output / "oracle.stderr").write_text(oracle.stderr)
    if oracle.returncode:
        raise RuntimeError(f"Independent oracle failed; see {output}/oracle.stderr")
    reference = json.loads(oracle.stdout)
    (output / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    dimension_database = module("native_dimension_oracles", "test/pcuda_stage7_text/validate_cli.py").sqlite_oracle(data) if args.manifest else None
    records = sql_records(items, stdout, reference, args.timing_model, dimension_database)
    manual_records = []
    for query, status, text in manual_raw:
        record = dict(id=query, **status, success=False)
        try:
            blocks = re.findall(r"__SSB_RESULT_BEGIN__\n(.*?)__SSB_RESULT_END__", text, re.S)
            if status["returncode"] or status["timeout"] or status["diagnostics"] or len(blocks) != args.trials:
                raise ValueError("Manual process/diagnostic/result count failure")
            rows = [[json.loads(line) for line in block.splitlines() if line.strip()] for block in blocks]
            expected = reference["manual"][query]
            record.update(rows=rows, expected=expected)
            if any(Counter(map(tuple, trial)) != Counter(map(tuple, expected)) for trial in rows):
                raise ValueError("Independent manual Text SSB oracle mismatch")
            record["success"] = True
        except (ValueError, KeyError) as error:
            record["error"] = str(error)
        manual_records.append(record)
    success = session["returncode"] == 0 and not session["timeout"] and not session["diagnostics"] and all(r["success"] for r in records + manual_records)
    report = dict(success=success, data=str(data), dataset_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  timing_model=args.timing_model, traces=args.traces, session=session, records=records, manual=manual_records,
                  row_counts=reference["row_counts"], raw_columns_verified=reference["raw_columns_verified"],
                  scope="Independent full SQL/manual SSB payloads; other manifest records are execution checks, not full payload parity",
                  q11="SQL includes 1994-01-01; manual excludes it. Both checked independently, no historical file rewrite.",
                  measurement="Correctness series, not a performance/SLA comparison. Native whole-process time is not SQL summary.")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Artifacts: {output}\nSQL {sum(r['success'] for r in records)}/{len(records)}; manual {sum(r['success'] for r in manual_records)}/{len(manual_records)}; success={success}")
    for record in records + manual_records:
        if not record["success"]:
            print(record["id"], record.get("error"))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
