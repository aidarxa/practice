#!/usr/bin/env python3
"""Organize sequential, rotated plain/traced SQL processes; all timers live in Bash/C++."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--runs", type=int, default=9)
    args = parser.parse_args()
    if args.sessions < 1 or args.runs < 2:
        parser.error("Require positive sessions and at least two executions")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sql-series-", dir=args.output_root.resolve()))
    environment = dict(os.environ, LC_ALL="C", TMPDIR="/tmp/opencode", ACPP_ADAPTIVITY_LEVEL="1", ACPP_ALLOCATION_TRACKING="1")
    records = []
    for session in range(args.sessions):
        for traced in ((False, True) if session % 2 == 0 else (True, False)):
            directory = output / f"session-{session}-{'traced' if traced else 'plain'}"
            directory.mkdir()
            argv = [sys.executable, str(Path(__file__).with_name("validate_cli.py").resolve()),
                    "--db-cli", str(args.db_cli.resolve()), "--mode", "performance", "--changed-parameters",
                    "--runs", str(args.runs), "--output-root", str(directory)]
            if traced:
                argv.append("--traces")
            with (directory / "stdout").open("w") as stdout, (directory / "stderr").open("w") as stderr:
                result = subprocess.run(argv, env=environment, stdout=stdout, stderr=stderr)
            records.append({"session": session, "traces": traced, "argv": argv, "returncode": result.returncode,
                            "reports": [str(path.relative_to(output)) for path in directory.glob("*/report.json")]})
    # No trace/CPU-oracle/aggregate analysis runs concurrently with a measured process.
    (output / "series.json").write_text(json.dumps({
        "success": all(record["returncode"] == 0 and len(record["reports"]) == 1 for record in records),
        "sessions_per_mode": args.sessions, "executions_per_session": args.runs,
        "changed_parameter_executions_per_session": args.runs - 1,
        "environment": {key: environment[key] for key in ("LC_ALL", "TMPDIR", "ACPP_ADAPTIVITY_LEVEL", "ACPP_ALLOCATION_TRACKING")},
        "order": records}, indent=2) + "\n")
    print(output)
    return 0 if all(record["returncode"] == 0 and len(record["reports"]) == 1 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
