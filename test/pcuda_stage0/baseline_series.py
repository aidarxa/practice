#!/usr/bin/env python3
"""Native-time, untraced SQL baseline series, with explicit independent Q1.1 validation."""

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from sql_baseline import cli_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--q11-oracle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    args = parser.parse_args()
    if args.series < 3 or args.runs < 3:
        parser.error("at least three series/runs required for acceptance baseline")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="baseline-series-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    oracle = json.loads(subprocess.check_output([str(args.q11_oracle.resolve()), str(args.data.resolve())], text=True, timeout=60))
    report = {"success": False, "series": [], "q11_oracle": oracle,
              "q11_reference_difference": "raw subreports remain unsuccessful; Q1.1 SQL correctness checked independently",
              "native_measurement": "Bash time + offline compiler/linker traces; no strace in these performance series"}
    try:
        for series in range(args.series):
            command = [sys.executable, "-B", str(Path(__file__).with_name("sql_baseline.py")),
                "--native-time", "--db-cli", str(args.db_cli.resolve()),
                "--acpp-include", str(args.acpp_include.resolve()), "--runs", str(args.runs),
                "--output-root", str(output)]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=300)
            (output / f"s{series}.stdout").write_text(proc.stdout)
            (output / f"s{series}.stderr").write_text(proc.stderr)
            if proc.returncode not in (0, 1):
                raise RuntimeError(proc.stderr[-2000:])
            path = Path(re.search(r"Artifacts: (.+)", proc.stdout)[1]) / "report.json"
            result = json.loads(path.read_text())
            if result.get("timed_out") or result.get("returncode") != 0 or not result.get("compiler_cache_verified"):
                raise RuntimeError("SQL session/counters failed")
            if len(result["records"]) != 13 * args.runs:
                raise RuntimeError("SQL series incomplete")
            stdout = path.with_name("stdout.txt").read_text()
            for record in result["records"]:
                if record["query"] != "q11":
                    if not record["result_verified"]:
                        raise RuntimeError(f"unexpected SSB mismatch: {record['query']}")
                else:
                    marker = f"__STAGE0_SQL__q11__{record['run']}__\n"
                    chunk = stdout.split(marker, 1)[1].split("__STAGE0_SQL__", 1)[0]
                    if cli_rows(chunk, ()) != Counter({(oracle["sql_q11"],): 1}):
                        raise RuntimeError("Q1.1 differs from independent CPU oracle")
            report["series"].append({"index": series, "report": str(path), "records": result["records"],
                "pch_prepare_ms": result["pch_prepare_wall_ms"], "compiler_calls": result["compiler_call_count"],
                "sql_correctness_verified": True, "all_original_references_match": result["success"]})
            print(f"s{series}: {13 * args.runs} SQL results verified (Q1.1 by independent CPU oracle)", flush=True)
        report["success"] = True
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
