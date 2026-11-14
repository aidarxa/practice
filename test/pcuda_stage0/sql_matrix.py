#!/usr/bin/env python3
"""Controlled JIT-only optimization/PCH comparison, keeping the host engine unchanged."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    args = parser.parse_args()
    if args.series < 1 or args.runs < 2:
        parser.error("invalid series/run counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sql-matrix-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    report = {"success": False, "measurements": [], "commands": [],
              "note": "host stays Release; PCH prepared even when not consumed, separately timed"}
    variants = [(o, p) for o in ("O0", "O3") for p in ("off", "on")]
    try:
        for series in range(args.series):
            for optimization, pch in variants[series % 4:] + variants[:series % 4]:
                name = f"s{series}-{optimization}-{pch}"
                command = [sys.executable, "-B", str(Path(__file__).with_name("sql_baseline.py")),
                    "--db-cli", str(args.db_cli.resolve()), "--acpp-include", str(args.acpp_include.resolve()),
                    "--acpp", args.acpp, "--query", "q33", "--runs", str(args.runs),
                    "--jit-optimization", optimization, "--jit-pch", pch, "--output-root", str(output)]
                command += ["--native-time", "--linker-opt", "0"]
                proc = subprocess.run(command, capture_output=True, text=True, timeout=240)
                (output / f"{name}.stdout").write_text(proc.stdout)
                (output / f"{name}.stderr").write_text(proc.stderr)
                report["commands"].append(command)
                if proc.returncode:
                    raise RuntimeError(f"{name} failed: {proc.stdout[-500:]} {proc.stderr[-2000:]}")
                path = Path(re.search(r"Artifacts: (.+)", proc.stdout)[1]) / "report.json"
                result = json.loads(path.read_text())
                if not result["success"] or not result["compiler_cache_verified"]:
                    raise RuntimeError(f"{name} not verified")
                report["measurements"].append({"series": series, "optimization": optimization,
                    "pch": pch == "on", "report": str(path), "records": result["records"],
                    "pch_prepare_wall_ms": result["pch_prepare_wall_ms"], "compiler_events": result["compiler_events"]})
                print(f"{name}: {len(result['records'])} results verified", flush=True)
        report["success"] = True
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
