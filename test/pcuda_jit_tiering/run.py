#!/usr/bin/env python3
"""Isolated S2 experiments. Python only organizes; Bash/C++ do all timing."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / "test/pcuda_stage0"
PERF = ROOT / "test/pcuda_compile_perf"
VARIANTS = {
    "stock": None,
    "hiprtc33": (3, 3, 0),
    "hiprtc03": (0, 3, 0),
    "hiprtc30": (3, 0, 0),
    "hiprtc00": (0, 0, 0),
    "clang33": (3, 3, 1),
    "clang03": (0, 3, 1),
    "clang30": (3, 0, 1),
    "clang00": (0, 0, 1),
    "clang11": (1, 1, 1),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def stats(values):
    return {"n": len(values), "raw": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None

def native(argv, prefix, env):
    with Path(str(prefix) + ".stdout").open("w") as output:
        process = subprocess.Popen(["bash", str(STAGE / "native_command.sh"), str(prefix),
                                    *map(str, argv)], cwd=ROOT, env=env, stdout=output, start_new_session=True)
        try:
            return process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124

def build_script(directory, base, pch, traced):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "export LC_ALL=C",
             "unset LD_PRELOAD JIT_POC_DIRECTORY JIT_POC_NATIVE_TIMER",
             "TIMEFORMAT='{\"wall_s\":%6R,\"user_s\":%6U,\"system_s\":%6S}'",
             'timed() { local p=$1; shift; { time "$@" >"$p.stdout" 2>"$p.stderr"; } 2>"$p.time.json"; }',
             "pids=()"]
    commands = []
    for name in ("date", "customer", "supplier", "probe"):
        obj = directory / (name + ".o")
        compile_cmd = [*base, "-O0", "-include-pch", pch, "-MD", "-MF", directory / (name + ".d"),
                       *([f"-ftime-trace={directory / (name + '-compile.json')}"] if traced else []),
                       "-c", STAGE / f"q33_{name}.cpp", "-o", obj]
        link_cmd = [*base, "-O0", "-shared", "-fuse-ld=lld", "-Wl,-O0",
                    *([f"-Wl,--time-trace={directory / (name + '-link.json')}"] if traced else []),
                    obj, "-o", directory / (name + ".so")]
        lines.append("(")
        for phase, cmd in (("compile", compile_cmd), ("link", link_cmd)):
            prefix = directory / (name + "-" + phase)
            lines.append("timed " + shlex.join([str(prefix), *map(str, cmd)]))
            commands.append({"phase": phase, "module": name, "argv": list(map(str, cmd)), "prefix": str(prefix)})
        lines += [") &", 'pids+=("$!")']
    lines += ["status=0", 'for pid in "${pids[@]}"; do wait "$pid" || status=1; done', 'exit "$status"']
    script = directory / "build.sh"
    script.write_text("\n".join(lines) + "\n")
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return script

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device-name", default="780M")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--diagnostics", choices=("both", "on", "off"), default="both")
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-jit-tiering"))
    args = parser.parse_args()
    names = args.variants.split(",")
    if any(n not in VARIANTS for n in names) or args.series < 1 or not 2 <= args.runs <= 50:
        parser.error("invalid variant/count")
    acpp = shutil.which(args.acpp)
    if not acpp or not args.control.is_file(): parser.error("missing acpp/control library")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="q33-", dir=args.output_root.resolve()))
    shutil.copy2(args.control, out / "libjit_control.so")
    args.control = out / "libjit_control.so"
    snapshots = out / "sources"
    snapshots.mkdir()
    inputs = [*HERE.glob("*.cpp"), *HERE.glob("*.hpp"), *HERE.glob("*.py"), *HERE.glob("*.sh"), HERE / "CMakeLists.txt",
              PERF / "q33_cold.cpp", *STAGE.glob("q33*.cpp"), *STAGE.glob("q33*.hpp"), STAGE / "runtime.hpp", STAGE / "pch.hpp"]
    source_hashes = {str(p): sha(p) for p in inputs}
    for p in inputs:
        shutil.copy2(p, snapshots / (p.parent.name + "-" + p.name))
    print(f"Artifacts: {out}", flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JIT_POC_") and k != "LD_PRELOAD"}
    env.update(LC_ALL="C", ACPP_ADAPTIVITY_LEVEL="1", ACPP_ALLOCATION_TRACKING="1")
    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC", "-g0", *args.extra_flag]
    report = {"success": False, "scope": "Q3.3 SF10 prototype; no SQL parse/plan/codegen",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "environment": {k: v for k, v in env.items() if k.startswith(("ACPP_", "HIP_", "ROCM_")) or k == "LC_ALL"},
              "sessions": [], "preparation": []}
    def prepare(label, command):
        code = native(command, out / label, env)
        report["preparation"].append({"label": label, "argv": list(map(str, command)), "returncode": code})
        if code: raise RuntimeError("preparation failed: " + label)
    try:
        prepare("host-compile", [*base, "-O3", "-DNDEBUG", "-c", PERF / "q33_cold.cpp", "-o", out / "host.o"])
        prepare("host-link", [*base, out / "host.o", "-ldl", "-fuse-ld=lld", "-o", out / "q33"])
        dry = subprocess.run([*base, "--acpp-dryrun", "-O0", "-c", STAGE / "q33_probe.cpp"],
                             env=env, capture_output=True, text=True, check=True)
        (out / "effective-template.txt").write_text(dry.stdout + dry.stderr)
        compiler = shlex.split(dry.stdout)[0]
        prepare("first-result-build", [compiler, *args.extra_flag, "-std=c++17", "-O3", HERE / "first_result.cpp", "-o", out / "first_result"])
        diagnostics = [False, True] if args.diagnostics == "both" else [args.diagnostics == "on"]
        for traced in diagnostics:
            pch = out / ("trace.pch" if traced else "plain.pch")
            prepare("pch-" + str(traced), [*base, "-O0", "-fpch-instantiate-templates",
                *([f"-ftime-trace={out / 'pch-trace.json'}"] if traced else []),
                "-c", "-x", "c++-header", STAGE / "pch.hpp", "-o", pch])
        for series in range(args.series):
            ordered = names[series % len(names):] + names[:series % len(names)]
            for name in ordered:
                for traced in diagnostics if series % 2 == 0 else diagnostics[::-1]:
                    directory = out / f"s{series}-{name}-{'trace' if traced else 'plain'}"
                    directory.mkdir()
                    session_env = {**env, "ACPP_APPDB_DIR": str(directory / "device-cache")}
                    spec = VARIANTS[name]
                    if spec:
                        session_env.update(LD_PRELOAD=str(args.control.resolve()), JIT_POC_IR=str(spec[0]),
                            JIT_POC_TOOL=str(spec[1]), JIT_POC_CLANG=str(spec[2]), JIT_POC_TRACES=str(int(traced)),
                            JIT_POC_DIRECTORY=str(directory), JIT_POC_NATIVE_TIMER=str(HERE / "native_tool.sh"))
                    script = build_script(directory, base, out / ("trace.pch" if traced else "plain.pch"), traced)
                    command = [out / "first_result", directory / "first-result.json", out / "q33",
                               args.data.resolve(), args.device_name, script, directory, args.runs]
                    code = native(command, directory / "process", session_env)
                    report["sessions"].append({"name": name, "series": series, "traced": traced,
                        "directory": str(directory), "returncode": code, "policy": spec,
                        "argv": list(map(str, command)), "environment": {k: v for k, v in session_env.items()
                            if k.startswith(("JIT_POC_", "ACPP_")) or k == "LD_PRELOAD"}})
                    (out / "progress.json").write_text(json.dumps(report["sessions"], indent=2) + "\n")
                    print(f"{directory.name}: exit={code}", flush=True)
        # Offline analysis begins only after ALL measured processes have exited.
        expected = Counter()
        for line in (ROOT / "test/queries/result/scale10/33.txt").read_text().splitlines():
            if line.startswith("Result count:"): break
            expected[tuple(map(int, line.split()))] += 1
        for session in report["sessions"]:
            directory = Path(session["directory"])
            session["verified"] = False
            if session["returncode"]: continue
            execution = json.loads((directory / "process.stdout").read_text())
            session["execution"] = execution
            session["first_result"] = json.loads((directory / "first-result.json").read_text())
            session["verified"] = all(Counter(map(tuple, r["result"])) == expected for r in execution["runs"])
            if session["policy"]:
                session["jit"] = json.loads((directory / "jit-events.json").read_text())
            session["commands"] = json.loads((directory / "commands.json").read_text())
            for command in session["commands"]:
                command["native"] = json.loads(Path(command["prefix"] + ".time.json").read_text())
            session["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in directory.glob("*.so")}
        report["aggregate"] = []
        for name in names:
            for traced in diagnostics:
                group = [s for s in report["sessions"] if s["name"] == name and s["traced"] == traced and s["verified"]]
                entry = {"name": name, "traced": traced, "successful_sessions": len(group)}
                for key in ("summary_ms", "preparation_ms", "build_elapsed_ms", "load_ms", "startup_ms", "data_load_ms"):
                    entry[key] = stats([s["execution"][key] for s in group])
                for key in ("execute_ms", "fetch_ms", "total_ms"):
                    entry["first_" + key] = stats([s["execution"]["runs"][0][key] for s in group])
                    entry["warm_" + key] = stats([r[key] for s in group for r in s["execution"]["runs"][1:]])
                for kind in ("optimizer", "external_tool", "full_transformation"):
                    entry[kind + "_sum_ms"] = stats([sum(e["end_ms"] - e["start_ms"] for e in s["jit"]["events"] if e["kind"] == kind)
                        for s in group if "jit" in s])
                report["aggregate"].append(entry)
        report["source_sha256"] = source_hashes
        report["sources_unchanged"] = all(sha(Path(p)) == h for p, h in source_hashes.items())
        report["control_sha256"] = sha(args.control)
        report["success"] = all(s["verified"] for s in report["sessions"])
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Offline correctness: {report['success']}", flush=True)
    if not report["success"]: raise SystemExit(1)

if __name__ == "__main__":
    main()
