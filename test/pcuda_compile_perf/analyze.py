#!/usr/bin/env python3
"""Offline export of every PERF-COLD distribution, without modifying raw reports."""

import argparse
import json
from pathlib import Path
import tempfile

from run import ROOT, sha256, stats, trace_summary
from sql_controls import variant_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("build/pcuda-compile-perf"))
    args = parser.parse_args()
    out = Path(tempfile.mkdtemp(prefix="analysis-", dir=args.artifacts.resolve()))
    result = {"inputs": {}, "series": [], "sql_controls": [], "validation": [], "types": []}
    tables = []
    for path in sorted(args.artifacts.glob("series-*/report.json")):
        data = json.loads(path.read_text())
        result["inputs"][str(path)] = sha256(path)
        rows = []
        commands = {c["label"]: c for c in data["commands"]}
        for entry in data["aggregate"]:
            row = dict(entry)
            row["method_version"] = data.get("method_version", 1)
            sessions = [s for s in data["sessions"] if s["name"] == row["name"] and s["diagnostic"] == row["diagnostic"] and s.get("verified")]
            for phase in ("compile", "link"):
                for interval in ("user_ms", "system_ms"):
                    row[f"{phase}_{interval}_sum"] = stats([sum(m[phase + "_native_ms"][interval] for m in s["modules"]) for s in sessions])
                for name in ("ExecuteCompiler", "Total Frontend", "Total Backend", "Total Optimizer", "Total PerformPendingInstantiations", "ExecuteLinker"):
                    values = [sum(m.get(phase + "_trace_ms", {}).get(name, 0) for m in s["modules"]) for s in sessions
                              if any(name in m.get(phase + "_trace_ms", {}) for m in s["modules"])]
                    row[phase + "_trace_" + name] = stats(values)
            pch_commands = [commands[s["pch_preparation_label"]] for s in sessions if s["pch_preparation_label"]]
            for interval in ("wall_ms", "user_ms", "system_ms"):
                row["pch_" + interval] = stats([c["native_ms"][interval] for c in pch_commands])
            row["pch_bytes"] = stats([s["pch_size_bytes"] for s in sessions if "pch_size_bytes" in s])
            row["load_ms"] = stats([s["execution"]["load_ms"] for s in sessions])
            row["process_wall_ms"] = stats([json.loads((Path(s["directory"]) / "process.time.json").read_text())["wall_s"] * 1000 for s in sessions])
            rows.append(row)
        result["series"].append({"report": str(path), "rows": rows, "session_count": len(data["sessions"]),
                                 "method_version": data.get("method_version", 1),
                                 "verified_query_results": sum(len(s["execution"]["runs"]) for s in data["sessions"] if s.get("verified"))})
        tables += [str(path) + " method_version=" + str(data.get("method_version", 1)),
                   "variant | traces | summary median [min,max] (V1 excludes query preparation!) | compile wall sum | link wall sum | build elapsed | first execute | fetch | warm execute | bytes | PCH wall"]
        for row in rows:
            def cell(name):
                value = row[name]
                return f"{value['median']:.3f} [{value['min']:.3f},{value['max']:.3f}]" if value else "not measured"
            tables.append(" | ".join([row["name"], str(row["diagnostic"]), *(cell(k) for k in
                ("summary_ms", "compile_wall_sum_ms", "link_wall_sum_ms", "build_elapsed_ms", "first_execute_ms",
                 "first_fetch_ms", "warm_execute_ms", "module_bytes_sum", "pch_wall_ms"))]))
        tables.append("")
    for path in sorted(args.artifacts.glob("sql-controls-*/report.json")):
        data = json.loads(path.read_text())
        result["inputs"][str(path)] = sha256(path)
        groups = []
        for variant in ("q33-default", "q33-O0", "q21-q33-O0"):
            sessions = [s for s in data["sessions"] if variant_of(s["name"]) == variant and not s["debug"]]
            records = [r for s in sessions for r in s["records"] if r["query"] == "q33" and r["mode"] == "cold_sql"]
            row = {"variant": variant, **{k: stats([r[k] for r in records]) for k in
                   ("total_query_ms", "compile_ms", "host_fetch_ms", "generated_execute_ms")}}
            natives = []
            for session in sessions:
                sub = json.loads(Path(session["report"]).read_text())
                events = [e for e in sub["compiler_events"] if e["phase"] == "compile_and_link"]
                natives.append(events[-1]["wall_ms"])
            row["native_acpp_compile_link_wall_ms"] = stats(natives)
            groups.append(row)
        result["sql_controls"].append({"report": str(path), "corrected_aggregate": groups,
            "correction": "Original q33-O0 aggregate used suffix matching and mixed two query orders. Raw sessions are correct. Exact variant grouping used here; raw report unchanged."})
    for path in sorted(args.artifacts.glob("validation-*/report.json")):
        data = json.loads(path.read_text())
        result["inputs"][str(path)] = sha256(path)
        details = []
        for session in data["sessions"]:
            directory = Path(session["directory"])
            modules = []
            for name in ("seed", "transform", "block_scan", "add_offsets", "block_reduce", "block_reduce_unsigned", "atomic_probe"):
                so = directory / (name + ".so")
                module = {"name": name, "sha256": sha256(so), "size_bytes": so.stat().st_size}
                for phase in ("compile", "link"):
                    timing = json.loads((directory / (name + "-" + phase + ".time.json")).read_text())
                    module[phase + "_native_ms"] = {k.replace("_s", "_ms"): v * 1000 for k, v in timing.items()}
                    module[phase + "_trace_ms"] = trace_summary(directory / (name + "-" + phase + "-trace.json"))
                modules.append(module)
            details.append({"series": session["series"], "variant": session["variant"], "module_artifacts": modules,
                            "module_fixtures": 72, "pending_unload_checks": 3,
                            "collective_fixtures": len(session["collectives"]["fixtures"]),
                            "verified": session["modules"]["verified"] and session["collectives"]["verified"]})
        result["validation"].append({"report": str(path), "sessions": details,
            "correction": "Original sessions.modules was overwritten by the modules suite result. Module metadata reconstructed offline from preserved objects/traces/timers; raw report unchanged."})
    for path in sorted(args.artifacts.glob("types-bench-*/report.json")):
        data = json.loads(path.read_text())
        result["inputs"][str(path)] = sha256(path)
        result["types"].append({"report": str(path), "success": data["success"], "measurements": data["measurements"]})
    history = args.artifacts / "history-audit.json"
    if history.exists():
        data = json.loads(history.read_text())
        sessions = [s for n, v in data.items() for s in v.get("sessions", []) if s.get("available")]
        hashes = [s["source_hashes"] for s in sessions]
        binaries = [s["binary_sha256"] for s in sessions]
        trace_files = []
        for s in sessions:
            event = s["q33_compiler_event"]
            for arg in event.get("argv", []):
                if arg.startswith(("-ftime-trace=", "-Wl,--time-trace=")):
                    path = Path(arg.split("=", 1)[1])
                    trace_files.append({"path": str(path), "exists": path.exists(), "sha256": sha256(path) if path.exists() else None,
                                        "summary": trace_summary(path) if path.exists() else None})
        result["historical_crosscheck"] = {"source_hash_sets_equal": all(h == hashes[0] for h in hashes),
            "current_sources_match": all(sha256(ROOT / p) == h for p, h in hashes[0].items()),
            "binary_hashes_equal": len(set(binaries)) == 1, "db_cli_sha256": binaries[0], "trace_files": trace_files}
    result["source_sha256"] = {str(p): sha256(p) for p in Path(__file__).parent.glob("*") if p.is_file()}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "tables.txt").write_text("\n".join(tables) + "\n")
    print(out)


if __name__ == "__main__":
    main()
