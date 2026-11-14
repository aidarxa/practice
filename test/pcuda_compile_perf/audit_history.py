#!/usr/bin/env python3
"""Read (never rewrite) stage-0 historical measurements and referenced traces."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def stats(values):
    return {"n": len(values), "raw": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stage = root / "build/pcuda-stage0"
    reports = {}
    for name in ("q33-bench-knd6vwtl", "sql-matrix-oe5qo0yx", "baseline-series-g4b_dmw4"):
        path = stage / name / "report.json"
        if not path.exists():
            reports[name] = {"available": False, "path": str(path)}
            continue
        data = json.loads(path.read_text())
        item = {"available": True, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if "q33-bench" in name:
            rows = [x for x in data["measurements"] if x["optimization"] == "O0" and x["pch"]]
            item["prototype_O0_pch"] = {field: stats([x[field] for x in rows])
                                        for field in ("compile_ms", "link_ms", "cold_module_path_ms")}
            item["prototype_O0_pch"]["summary_ms"] = None
            item["trace_examples"] = []
            for x in rows:
                for module in x["modules"]:
                    trace = path.parent / (x["name"] + "-" + module["name"] + "-compile-trace.json")
                    item["trace_examples"].append({"path": str(trace), "exists": trace.exists(),
                        "sha256": hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None,
                        "fields": module["compiler_trace_ms"]})
        else:
            rows = data["measurements"] if "matrix" in name else data["series"]
            if "matrix" in name:
                rows = [x for x in rows if x["optimization"] == "O0" and x["pch"]]
            records = [r for x in rows for r in x["records"] if r["query"] == "q33" and r["mode"] == "cold_sql"]
            item["q33_cold"] = {f: stats([r[f] for r in records]) for f in
                                 ("total_query_ms", "compile_ms", "host_fetch_ms", "generated_execute_ms")}
            item["sessions"] = []
            native = []
            for row in rows:
                subpath = Path(row["report"])
                if not subpath.exists():
                    item["sessions"].append({"path": str(subpath), "available": False})
                    continue
                sub = json.loads(subpath.read_text())
                events = sub.get("compiler_events", [])
                # The matrix has one SQL; the full baseline's ninth SQL is Q3.3.
                query_events = [e for e in events if any(str(a).endswith(".cpp") for a in e.get("argv", []))]
                index = 0 if "matrix" in name else 8
                event = query_events[index] if len(query_events) > index else {}
                if "wall_ms" in event:
                    native.append(event["wall_ms"])
                cold_order = [r["query"] for r in sub.get("records", []) if r["mode"] == "cold_sql"]
                item["sessions"].append({"path": str(subpath), "available": True,
                    "source_hashes": sub.get("source_sha256"), "binary_sha256": sub.get("db_cli_sha256"),
                    "top_level_fields": list(sub), "cold_order": cold_order, "q33_compiler_event": event,
                    "q21_fetch_ms": [r["host_fetch_ms"] for r in sub.get("records", [])
                                     if r["query"] == "q21" and r["mode"] == "cold_sql"]})
            item["q33_native_acpp_compile_link_wall_ms"] = stats(native)
        reports[name] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
