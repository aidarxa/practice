#!/usr/bin/env python3
"""Prepare a low tier, return a result, then compile/prepare/promote a high tier."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import tempfile
from run import HERE, ROOT, STAGE, PERF, VARIANTS, native, sha, stats

def verify(directory, execution):
    rows = execution["rows"]
    blob = (directory / "results.bin").read_bytes()
    if len(blob) != 8 * rows * len(execution["runs"]):
        return {"success": False, "reason": "wrong binary output size"}
    mismatches = []
    for index, run in enumerate(execution["runs"]):
        result = struct.unpack_from(f"<{rows}q", blob, index * rows * 8)
        for i, actual in enumerate(result):
            value = 9007199254740993 + i % 1009 - 504
            expected = (value % 997) * 7 + run["offset"] if i % 11 and value >= run["threshold"] else -(1 << 63)
            if actual != expected:
                mismatches.append({"run": index, "row": i, "actual": actual, "expected": expected})
                if len(mismatches) >= 10: break
    return {"success": not mismatches, "checked_elements": rows * len(execution["runs"]), "mismatches": mismatches}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--device-name", default="780M")
    parser.add_argument("--pin-driver", type=Path, help="diagnostic RTLD_NODELETE pin, never enabled implicitly")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--low-profiles", default="hiprtc03,clang00,clang11")
    parser.add_argument("--modes", default="idle,overlap,failure")
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--rows", type=int, default=262147)
    parser.add_argument("--diagnostics", choices=("both", "on", "off"), default="both")
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-jit-tiering"))
    args = parser.parse_args()
    names, modes = args.low_profiles.split(","), args.modes.split(",")
    if any(n not in VARIANTS or VARIANTS[n] is None for n in names) or any(m not in ("idle", "overlap", "failure") for m in modes):
        parser.error("invalid profile/mode")
    if args.series < 1 or not 2 <= args.runs <= 50 or not 1 <= args.rows <= 64 * 1024 * 1024:
        parser.error("invalid counts")
    acpp = shutil.which(args.acpp)
    if not acpp or not args.control.is_file(): parser.error("missing tool/control")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="tiered-", dir=args.output_root.resolve()))
    shutil.copy2(args.control, out / "libjit_control.so")
    args.control = out / "libjit_control.so"
    snapshots = out / "sources"
    snapshots.mkdir()
    inputs = [*HERE.glob("*.cpp"), *HERE.glob("*.hpp"), *HERE.glob("*.py"), *HERE.glob("*.sh"), HERE / "CMakeLists.txt",
              STAGE / "runtime.hpp", STAGE / "pch.hpp"]
    source_hashes = {str(p): sha(p) for p in inputs}
    for p in inputs:
        shutil.copy2(p, snapshots / (p.parent.name + "-" + p.name))
    print(f"Artifacts: {out}", flush=True)
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("JIT_POC_")}
    env.update(LC_ALL="C", ACPP_ADAPTIVITY_LEVEL="1", ACPP_ALLOCATION_TRACKING="1")
    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC", "-g0", *args.extra_flag]
    report = {"success": False, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "sessions": [], "preparation": [], "scope": "synthetic filter/project, not a SQL engine"}
    def prepare(label, cmd):
        code = native(cmd, out / label, env)
        report["preparation"].append({"label": label, "argv": list(map(str, cmd)), "returncode": code})
        if code: raise RuntimeError("preparation failed: " + label)
    def script(directory, tier, traced, failure=False):
        path = directory / (tier + "-build.sh")
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "unset LD_PRELOAD JIT_POC_DIRECTORY JIT_POC_NATIVE_TIMER"]
        if failure:
            lines += ["# Deliberately fail the optional background job, not the answered query.", "exit 17"]
            path.write_text("\n".join(lines) + "\n")
            return path
        compile_cmd = [*base, "-O0", "-include-pch", out / "headers.pch", f"-DJIT_TIER={0 if tier == 'low' else 1}",
            "-MD", "-MF", directory / (tier + ".d"),
            *([f"-ftime-trace={directory / (tier + '-compile.json')}"] if traced else []),
            "-c", HERE / "tier_kernel.cpp", "-o", directory / (tier + ".o")]
        link_cmd = [*base, "-O0", "-shared", "-fuse-ld=lld", "-Wl,-O0",
            *([f"-Wl,--time-trace={directory / (tier + '-link.json')}"] if traced else []),
            directory / (tier + ".o"), "-o", directory / (tier + ".so")]
        commands = []
        for phase, cmd in (("compile", compile_cmd), ("link", link_cmd)):
            prefix = directory / (tier + "-" + phase)
            lines.append(shlex.join(["bash", str(STAGE / "native_command.sh"), str(prefix), *map(str, cmd)]) + f" >{shlex.quote(str(prefix) + '.stdout')}")
            commands.append({"phase": phase, "tier": tier, "argv": list(map(str, cmd)), "prefix": str(prefix)})
            dry = subprocess.run([*map(str, cmd), "--acpp-dryrun"], env=env, capture_output=True, text=True, check=True)
            (directory / (tier + "-" + phase + "-effective.txt")).write_text(dry.stdout + dry.stderr)
        (directory / (tier + "-commands.json")).write_text(json.dumps(commands, indent=2) + "\n")
        path.write_text("\n".join(lines) + "\n")
        return path
    try:
        prepare("host-compile", [*base, "-O3", "-DNDEBUG", "-pthread", "-c", HERE / "tiered.cpp", "-o", out / "host.o"])
        prepare("host-link", [*base, out / "host.o", "-pthread", "-ldl", "-fuse-ld=lld",
            "-L" + str(args.control.resolve().parent), "-ljit_control", "-Wl,-rpath," + str(args.control.resolve().parent), "-o", out / "tiered"])
        dry = subprocess.run([*base, "--acpp-dryrun", "-O0", "-c", HERE / "tier_kernel.cpp"],
            capture_output=True, text=True, env=env, check=True)
        compiler = shlex.split(dry.stdout)[0]
        prepare("first-result-build", [compiler, *args.extra_flag, "-std=c++17", "-O3", HERE / "first_result.cpp", "-o", out / "first_result"])
        prepare("pch", [*base, "-O0", "-fpch-instantiate-templates", "-ftime-trace=" + str(out / "pch-trace.json"),
            "-c", "-x", "c++-header", STAGE / "pch.hpp", "-o", out / "headers.pch"])
        diagnostics = [False, True] if args.diagnostics == "both" else [args.diagnostics == "on"]
        variants = [(n, m) for n in names for m in modes]
        for series in range(args.series):
            ordered = variants[series % len(variants):] + variants[:series % len(variants)]
            for name, mode in ordered:
                for traced in diagnostics if series % 2 == 0 else diagnostics[::-1]:
                    directory = out / f"s{series}-{name}-{mode}-{'trace' if traced else 'plain'}"
                    directory.mkdir()
                    low = script(directory, "low", traced)
                    high = script(directory, "high", traced, mode == "failure")
                    policy = VARIANTS[name]
                    session_env = {**env, "LD_PRELOAD": str(args.control.resolve()),
                        "ACPP_APPDB_DIR": str(directory / "device-cache"),
                        "JIT_POC_DIRECTORY": str(directory), "JIT_POC_NATIVE_TIMER": str(HERE / "native_tool.sh"),
                        "JIT_POC_TRACES": str(int(traced))}
                    if args.pin_driver:
                        session_env["JIT_POC_DRIVER_PIN"] = str(args.pin_driver.resolve())
                    cmd = [out / "first_result", directory / "first-result.json", out / "tiered", args.device_name,
                           low, high, directory, *policy, mode, args.rows, args.runs]
                    code = native(cmd, directory / "process", session_env)
                    report["sessions"].append({"name": name, "mode": mode, "series": series, "traced": traced,
                        "directory": str(directory), "returncode": code, "argv": list(map(str, cmd)),
                        "environment": {k: v for k, v in session_env.items() if k.startswith(("ACPP_", "JIT_POC_")) or k == "LD_PRELOAD"}})
                    (out / "progress.json").write_text(json.dumps(report["sessions"], indent=2) + "\n")
                    print(f"{directory.name}: exit={code}", flush=True)
        # Offline oracle: no result analysis inside any measured process.
        for session in report["sessions"]:
            directory = Path(session["directory"])
            session["verified"] = False
            if session["returncode"]: continue
            execution = json.loads((directory / "process.stdout").read_text())
            session["execution"] = execution
            session["oracle"] = verify(directory, execution)
            session["jit"] = json.loads((directory / "jit-events.json").read_text())
            session["first_result"] = json.loads((directory / "first-result.json").read_text())
            expected_compiles = 1 if session["mode"] == "failure" else 2
            session["verified"] = session["oracle"]["success"] and execution["compile_count_final"] == expected_compiles
            session["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in directory.glob("*.so")}
        report["aggregate"] = []
        for name, mode in variants:
            for traced in diagnostics:
                group = [s for s in report["sessions"] if s["name"] == name and s["mode"] == mode and s["traced"] == traced and s["verified"]]
                entry = {"name": name, "mode": mode, "traced": traced, "sessions": len(group)}
                for key in ("summary_ms", "build_elapsed_ms", "load_ms", "background_build_ms", "background_prepare_ms", "background_total_ms"):
                    entry[key] = stats([s["execution"][key] for s in group])
                for tier in ("low-first", "low-during-background", "low-warm", "high-warm"):
                    for key in ("execute_ms", "fetch_ms", "total_ms"):
                        entry[tier + "_" + key] = stats([r[key] for s in group for r in s["execution"]["runs"] if r["tier"] == tier])
                report["aggregate"].append(entry)
        report["source_sha256"] = source_hashes
        report["sources_unchanged"] = all(sha(Path(p)) == h for p, h in source_hashes.items())
        report["control_sha256"] = sha(args.control)
        report["host_sha256"] = sha(out / "tiered")
        report["success"] = all(s["verified"] for s in report["sessions"])
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Offline verification: {report['success']}", flush=True)
    if not report["success"]: raise SystemExit(1)

if __name__ == "__main__":
    main()
