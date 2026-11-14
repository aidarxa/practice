#!/usr/bin/env python3
## @file
## @brief Preserve native timers, compiler commands/traces and actual tiny-data GPU checks.
"""Correctness only. Python runs outside the native measured process chain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, collect_events, read_time


def native_time(path):
    """@brief Preserve signal diagnostics in raw output, parsing only the native JSON line offline."""
    lines = Path(path).read_text().splitlines()
    value = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
    return {key.replace("_s", "_ms"): number * 1000 for key, number in value.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args()
    output = Path(tempfile.mkdtemp(prefix="gpu-fixture-", dir=args.output_root.resolve()))
    pch = Path("/dev/shm/jit_headers.h.gch")
    descriptor = os.open(pch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    report = {"purpose": "correctness-only", "success": False, "traces": args.traces}
    try:
        toolchain = output / "toolchain"
        toolchain.mkdir()
        observer = toolchain / "acpp"
        shutil.copyfile(ROOT / "test/pcuda_stage1/compiler_observer.sh", observer)
        observer.chmod(0o700)
        events = output / "compiler-events"
        events.mkdir()
        env = os.environ.copy()
        env.update({"LC_ALL": "C", "PATH": str(toolchain) + os.pathsep + env.get("PATH", ""),
                    "CRYSTAL_STAGE0_REAL_ACPP": shutil.which("acpp"), "CRYSTAL_STAGE0_EVENT_DIR": str(events),
                    "CRYSTAL_STAGE1_TRACES": "1" if args.traces else "0", "CRYSTAL_INCLUDE_DIR": str(ROOT / "include"),
                    "CRYSTAL_DEPS_INCLUDE_DIR": str(ROOT / "deps/include"), "ACPP_INCLUDE_DIR": str(args.acpp_include.resolve()),
                    "ACPP_APPDB_DIR": str(output / "device-cache"), "DB_STAGE2_FIXTURE_NONCE": uuid.uuid4().hex})
        binary = args.binary.resolve()
        command = timed_command([binary], output / "time.json", output / "stderr.txt")
        report.update({"command": command, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                       "environment": {k: v for k, v in env.items() if k.startswith(("ACPP_", "CRYSTAL_", "DB_STAGE2_", "HIP_", "HSA_"))}})
        with (output / "stdout.txt").open("w") as stdout:
            completed = subprocess.run(command, env=env, cwd=ROOT, stdout=stdout, timeout=900)
        report.update({"returncode": completed.returncode, "native_process": native_time(output / "time.json"),
                       "compiler_events": collect_events(events), "success": completed.returncode == 0})
        generated = output / "generated"
        generated.mkdir()
        manifest = []
        for event in report["compiler_events"]:
            for operand in event["argv"]:
                source = Path(operand)
                if source.suffix != ".cpp" or not source.is_file():
                    continue
                for artifact in (source, source.with_suffix(".so")):
                    if artifact.is_file():
                        shutil.copy2(artifact, generated / artifact.name)
                        manifest.append({"source": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
        (output / "generated-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        if pch.exists():
            shutil.move(pch, output / "legacy-pch.gch")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Artifacts:", output)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
