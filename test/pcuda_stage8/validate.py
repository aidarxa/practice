#!/usr/bin/env python3
## @file
## @brief Reuse existing independent GPU/CLI/oracle clients; preserve their raw failures and scopes.
"""Sequential functional acceptance, not a performance series or replacement for per-client reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage7"))
from validate import process, unexpected_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ssb/sf10")
    parser.add_argument("--timing-model", choices=("pcuda-modules-v1","pcuda-modules-overlap-exp-v1"), required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="functional-", dir=args.output_root.resolve()))
    build = args.build.resolve()
    env = dict(os.environ, LC_ALL="C", TMPDIR="/tmp/opencode", ACPP_ADAPTIVITY_LEVEL="1", ACPP_ALLOCATION_TRACKING="1", PYTHONDONTWRITEBYTECODE="1")
    overlap=args.timing_model=="pcuda-modules-overlap-exp-v1"
    env["CRYSTAL_JIT_OVERLAP_EXPERIMENT"]="on" if overlap else "off"
    env["CRYSTAL_DATA_DIR"] = str(args.data.resolve())
    os.environ.update({name:env[name] for name in ("LC_ALL", "TMPDIR", "ACPP_ADAPTIVITY_LEVEL", "ACPP_ALLOCATION_TRACKING", "PYTHONDONTWRITEBYTECODE")})
    os.environ["CRYSTAL_JIT_OVERLAP_EXPERIMENT"]=env["CRYSTAL_JIT_OVERLAP_EXPERIMENT"]
    os.environ["CRYSTAL_DATA_DIR"]=env["CRYSTAL_DATA_DIR"]
    records = []
    for target in ("test_pcuda_text_resources", "test_pcuda_primitives_sql", "test_gpu_plan", "test_pcuda_sql_modules",
                   "test_pcuda_runtime_parameters", "test_pcuda_primitives_gpu", "test_pcuda_gpu_runtime", "test_manual_ssb_gpu"):
        directory = output / target
        arguments = [str(directory / "application")] if target in ("test_pcuda_sql_modules", "test_pcuda_runtime_parameters") else []
        if overlap and target=="test_pcuda_sql_modules":arguments.append("--overlap")
        record, _, stderr = process(build / target, arguments, "", directory)
        diagnostics = unexpected_diagnostics(stderr)
        expected = [line for line in diagnostics if target == "test_pcuda_gpu_runtime" and "test/pcuda_stage3/test_gpu_runtime.cpp:43 @ asynchronousError(): stage3 injected non-PCUDA async error" in line]
        record.update(name=target, expected_injected_diagnostics=expected, unexpected_diagnostics=[line for line in diagnostics if line not in expected])
        record["success"] = record["returncode"] == 0 and not record["timeout"] and not record["unexpected_diagnostics"]
        records.append(record)
        print(target, record["success"], flush=True)
    jobs = [
        ("text-gpu", ["test/pcuda_stage7_text/validate_gpu.py", "--build", build]),
        ("text-cli-plain", ["test/pcuda_stage7_text/validate_cli.py", "--build", build, "--data", args.data, "--timing-model",args.timing_model]),
        ("text-cli-traced", ["test/pcuda_stage7_text/validate_cli.py", "--build", build, "--data", args.data, "--traces", "--timing-model",args.timing_model]),
        ("native-ssb-manual-manifest", ["test/ssb_native_data/validate.py", "--build", build, "--data", args.data, "--manifest", "--trials", "3", "--timing-model",args.timing_model]),
    ]
    for name, argv in jobs:
        directory = output / name
        directory.mkdir()
        argv = [sys.executable, *map(str, argv), "--output-root", str(directory)]
        (directory / "command.json").write_text(json.dumps(dict(argv=argv, environment={k:v for k,v in env.items() if k.startswith(("ACPP_","CRYSTAL_")) or k in ("LC_ALL","TMPDIR")}), indent=2) + "\n")
        with (directory / "stdout").open("w") as stdout, (directory / "stderr").open("w") as stderr:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        reports = list(directory.glob("*/report.json"))
        record = dict(name=name, returncode=result.returncode, reports=[str(p.relative_to(output)) for p in reports], success=result.returncode == 0 and len(reports) == 1)
        records.append(record)
        print(name, record, flush=True)
    complete = all(r["success"] for r in records)
    (output / "report.json").write_text(json.dumps(dict(functional_checks_passed=complete, full_file_parity=False,timing_model=args.timing_model, records=records,
        note="Native datasets use fresh independent Text SSB oracles for both Q1.1 date bounds. Historical numeric file mismatches/reports are unchanged; other manifest checks are execution parity, not an oracle of every cell."), indent=2) + "\n")
    print(output, flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
