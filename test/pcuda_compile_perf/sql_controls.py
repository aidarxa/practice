#!/usr/bin/env python3
"""Small historical-comparison controls; not another SYCL optimization matrix."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from run import STAGE, stats


def variant_of(name):
    return name.partition("-")[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-compile-perf"))
    args = parser.parse_args()
    if args.series < 1:
        parser.error("series must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="sql-controls-", dir=args.output_root.resolve()))
    print(f"Artifacts: {out}", flush=True)
    report = {"sessions": [], "success": False, "scope": "historical controls, compiler traces ON; no SYCL flag search"}
    variants = ("q33-default", "q33-O0", "q21-q33-O0")

    def session(name, extra, debug=False):
        argv = [sys.executable, "-B", str(STAGE / "sql_baseline.py"), "--db-cli", str(args.db_cli.resolve()),
                "--acpp-include", str(args.acpp_include.resolve()), "--runs", "3", "--jit-optimization", "O0",
                "--jit-pch", "on", "--output-root", str(out), *extra]
        env = {**os.environ, "LC_ALL": "C", "ACPP_ADAPTIVITY_LEVEL": "1", "ACPP_ALLOCATION_TRACKING": "1"}
        if debug:
            env["ACPP_DEBUG_LEVEL"] = "3"
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=240)
        (out / (name + ".stdout")).write_text(result.stdout)
        (out / (name + ".stderr")).write_text(result.stderr)
        match = re.search(r"Artifacts: (.+)", result.stdout)
        report["sessions"].append({"name": name, "argv": argv, "returncode": result.returncode,
                                    "debug": debug, "report": str(Path(match[1]) / "report.json") if match else None})
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: exit={result.returncode}", flush=True)
        if not match:
            raise RuntimeError("SQL runner did not produce an artifact path")

    try:
        for series in range(args.series):
            for name in variants[series % 3:] + variants[:series % 3]:
                extra = ["--query", "q33"] if name.startswith("q33") else ["--query", "q21", "--query", "q33"]
                if name.endswith("O0"):
                    extra += ["--linker-opt", "0"]
                session(f"s{series}-{name}", extra)
        session("debug-q21-q33", ["--query", "q21", "--query", "q33", "--linker-opt", "0"], debug=True)
        session("ssb-all13", [])
        # Offline: preserve subreports including the historical Q1.1 reference mismatch.
        for item in report["sessions"]:
            data = json.loads(Path(item["report"]).read_text())
            item["success"] = data["success"]
            item["records"] = data["records"]
            item["db_cli_sha256"] = data["db_cli_sha256"]
            item["source_sha256"] = data["source_sha256"]
            item["compiler_call_count"] = data["compiler_call_count"]
            item["compiler_cache_verified"] = data["compiler_cache_verified"]
        report["aggregate"] = []
        for variant in variants:
            records = [r for s in report["sessions"] if variant_of(s["name"]) == variant and not s["debug"]
                       for r in s["records"] if r["mode"] == "cold_sql" and r["query"] == "q33"]
            report["aggregate"].append({"variant": variant, **{k: stats([r[k] for r in records]) for k in
                                          ("total_query_ms", "compile_ms", "host_fetch_ms", "generated_execute_ms")}})
        report["success"] = all(s["success"] for s in report["sessions"])
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "report.json")


if __name__ == "__main__":
    main()
