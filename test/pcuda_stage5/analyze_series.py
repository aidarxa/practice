#!/usr/bin/env python3
"""Offline SQL Q3.3 oracle and distributions; never used inside a measured process."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from sql_baseline import REFERENCE_GROUPS, reference_rows


def distribution(values):
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values), "raw": values} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path)
    args = parser.parse_args()
    directory = args.series.resolve()
    source = directory / "series.json"
    series = json.loads(source.read_text())
    expected = reference_rows((ROOT / "test/queries/result/scale10/33.txt").read_text())
    # Reference order is d_year,c_city,s_city,revenue, independently of SELECT order.
    changed = Counter({row: count for row, count in expected.items() if row[1] == row[2] == 235 and 1993 <= row[0] <= 1996})
    failures = []
    phases = {mode: {phase: [] for phase in ("cold", "warm", "parameters")} for mode in ("plain", "traced")}
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    process_times = {mode: [] for mode in phases}
    sizes = {mode: [] for mode in phases}
    traces = []
    checked = 0
    for process in series["order"]:
        report_file = directory / process["reports"][0]
        hashes[str(report_file)] = hashlib.sha256(report_file.read_bytes()).hexdigest()
        report = json.loads(report_file.read_text())
        mode = "traced" if process["traces"] else "plain"
        if not report["success"]:
            failures.append(str(report_file) + ": runner failed")
        process_times[mode].append(report["process_time"])
        modules = list(report_file.parent.glob("db-pcuda-*/*/*.so"))
        sizes[mode].append(sum(module.stat().st_size for module in modules))
        for record in report["records"]:
            phase = "parameters" if "-parameters-" in record["id"] else ("cold" if record["id"].endswith("-0") else "warm")
            columns = [name.lower() for name in record["columns"]]
            indices = [columns.index(name) for name in REFERENCE_GROUPS["q33"]]
            indices += [next(index for index in range(len(columns)) if index not in indices)]
            actual = Counter(tuple(row[index] for index in indices) for row in record["rows"])
            if actual != (changed if phase == "parameters" else expected):
                failures.append(str(report_file) + ": " + record["id"] + " oracle mismatch")
            if phase != "cold" and any(record.get("module_" + key + "_calls") != 0 for key in
                                       ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")):
                failures.append(str(report_file) + ": " + record["id"] + " unexpected build event")
            phases[mode][phase].append(record)
            checked += 1
        if process["traces"]:
            traces.append({name: sum(fields.get(name, 0) for path, fields in report["trace_intervals_ms"].items()
                                     if path.endswith(".clang.json"))
                           for name in ("ExecuteCompiler", "Total Frontend", "Total Backend", "Total Optimizer", "Total CodeGenPasses")})
            traces[-1]["ExecuteLinker"] = sum(fields.get("ExecuteLinker", 0) for path, fields in report["trace_intervals_ms"].items()
                                              if path.endswith(".lld.json"))
    fields = ("total_query_ms", "prepare_ms", "codegen_ms", "build_elapsed_ms", "compile_wall_sum_ms", "compile_user_sum_ms",
              "compile_system_sum_ms", "link_wall_sum_ms", "link_user_sum_ms", "link_system_sum_ms", "library_load_ms",
              "generated_execute_ms", "host_fetch_ms", "module_compiler_calls", "module_linker_calls", "module_loads_calls")
    summary = {mode: {phase: {field: distribution([record[field] for record in records]) for field in fields}
                      for phase, records in values.items()} for mode, values in phases.items()}
    result = {"success": series["success"] and not failures, "oracle_results_checked": checked, "failures": failures,
              "units": "ms except event counts and module bytes", "distributions": summary,
              "module_bytes": {mode: distribution(values) for mode, values in sizes.items()},
              "native_process_times": {mode: {key: distribution([value[key] for value in values]) for key in ("wall_ms", "user_ms", "system_ms")}
                                       for mode, values in process_times.items()},
              "trace_phase_sums_ms": {key: distribution([value[key] for value in traces]) for key in traces[0]} if traces else {},
              "input_sha256": hashes,
              "boundaries": "summary=direct total_query_ms; command wall sums overlap; process wall includes all 17 results and teardown, NOT startup-to-first-result; trace phases are nested and NOT additive"}
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checked {checked} independent results; failures={len(failures)}; {directory / 'analysis.json'}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
