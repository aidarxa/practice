#!/usr/bin/env python3
"""Independent Q3.3: staged PCUDA compilation and process-local module execution."""

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from sql_baseline import reference_rows, run_session
from native_timing import timed_command, read_time, trace_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--linker-opt", choices=("0", "1"), default="0")
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    args = parser.parse_args()
    if args.series < 1 or args.runs < 2 or args.runs > 50:
        parser.error("invalid series/run counts")
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    source = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="q33-bench-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    report = {"success": False, "measurements": [], "commands": [],
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source.iterdir() if p.suffix in (".hpp", ".cpp", ".py")},
              "device_jit_cache": "fresh directory per process/series; OS caches not flushed"}

    def command(label, argv, env=None):
        timing_file, errors = output / f"{label}.time.json", output / f"{label}.command.stderr"
        timed = timed_command(argv, timing_file, errors)
        code, stdout, timer_stderr, timeout = run_session(timed, "", source, env or os.environ.copy(), 180)
        stderr = errors.read_text() + timer_stderr
        native_time = read_time(timing_file)
        ms = native_time["wall_ms"]
        (output / f"{label}.stdout").write_text(stdout)
        (output / f"{label}.stderr").write_text(stderr)
        report["commands"].append({"label": label, "argv": [str(x) for x in argv], **native_time, "returncode": code})
        if code or timeout:
            raise RuntimeError(f"{label} failed: {stderr[-3000:]}")
        return stdout, ms

    try:
        command("acpp-version", [acpp, "--acpp-version"])
        base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC", *args.extra_flag]
        command("host-compile", [*base, "-O3", "-DNDEBUG", "-MD", "-MF", output / "host.d",
                                 "-c", source / "q33_main.cpp", "-o", output / "host.o"])
        command("host-link", [*base, output / "host.o", "-ldl", "-o", output / "q33"])
        expected = reference_rows((source.parents[1] / "test/queries/result/scale10/33.txt").read_text())
        variants = [("O0", True)] if args.baseline_only else [(o, p) for o in ("O0", "O3") for p in (False, True)]
        for series in range(args.series):
            # Rotate order across series to reduce a fixed-order cache/thermal bias.
            ordered = variants[series % len(variants):] + variants[:series % len(variants)]
            for optimization, use_pch in ordered:
                name = f"s{series}-{optimization}-{'pch' if use_pch else 'no-pch'}"
                flags = [*base, f"-{optimization}"]
                pch_ms = 0.0
                if use_pch:
                    pch = output / f"{name}.pch"
                    _, pch_ms = command(f"{name}-pch", [*flags, f"-ftime-trace={output / (name + '-pch-trace.json')}", "-c", "-fpch-instantiate-templates",
                        "-x", "c++-header", source / "pch.hpp", "-o", pch])
                    if pch.read_bytes()[:4] != b"CPCH":
                        raise RuntimeError("invalid PCH")
                    flags += ["-include-pch", pch]
                modules = []
                for module in ("date", "customer", "supplier", "probe"):
                    obj, so = output / f"{name}-{module}.o", output / f"{name}-{module}.so"
                    compiler_trace = output / f"{name}-{module}-compile-trace.json"
                    linker_trace = output / f"{name}-{module}-link-trace.json"
                    _, compile_ms = command(f"{name}-{module}-compile", [*flags, f"-ftime-trace={compiler_trace}", "-g0", "-MD", "-MF", obj.with_suffix(".d"),
                        "-c", source / f"q33_{module}.cpp", "-o", obj])
                    if "sycl/sycl.hpp" in obj.with_suffix(".d").read_text():
                        raise RuntimeError("full SYCL dependency")
                    _, link_ms = command(f"{name}-{module}-link", [*base, "-shared", "-fuse-ld=lld",
                        f"-Wl,-O{args.linker_opt}", f"-Wl,--time-trace={linker_trace}", obj, "-o", so])
                    modules.append({"name": module, "compile_ms": compile_ms, "link_ms": link_ms,
                                    "compiler_trace_ms": trace_summary(compiler_trace),
                                    "linker_trace_ms": trace_summary(linker_trace),
                                    "size_bytes": so.stat().st_size, "path": str(so)})
                env = os.environ.copy()
                env["ACPP_APPDB_DIR"] = str(output / f"{name}-device-cache")
                stdout, process_ms = command(f"{name}-execute", [output / "q33", args.data.resolve(), args.device_name,
                    *(m["path"] for m in modules), str(args.runs)], env)
                execution = json.loads(stdout)
                for run in execution["runs"]:
                    if Counter(tuple(row) for row in run["result"]) != expected:
                        raise RuntimeError(f"Q3.3 SF10 result mismatch: {name}")
                compile_ms = sum(m["compile_ms"] for m in modules)
                link_ms = sum(m["link_ms"] for m in modules)
                report["measurements"].append({"name": name, "optimization": optimization, "pch": use_pch,
                    "linker": "lld", "linker_optimization": args.linker_opt, "measurement": "native Bash time + offline traces",
                    "series": series, "pch_prepare_ms": pch_ms, "modules": modules, "compile_ms": compile_ms,
                    "link_ms": link_ms, "execution": execution, "process_wall_ms": process_ms,
                    "cold_module_path_ms": compile_ms + link_ms + execution["load_ms"] +
                        execution["runs"][0]["execute_ms"] + execution["runs"][0]["fetch_ms"], "verified": True})
                print(f"{name}: SF10 verified; compile={compile_ms:.1f}, link={link_ms:.1f} ms", flush=True)
        report["success"] = True
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
