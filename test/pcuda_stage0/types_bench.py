#!/usr/bin/env python3
"""Compare actual lazy compilation, runtime tags, and upfront typed module preparation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from native_timing import timed_command, read_time, trace_summary
from sql_baseline import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--bounded-four", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    args = parser.parse_args()
    if args.series < 1:
        parser.error("invalid series count")
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    source = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="types-bench-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    report = {"success": False, "measurements": [], "commands": [],
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in source.iterdir() if p.suffix in (".cpp", ".hpp", ".py", ".sh")},
              "expression": "(input + p0) * p1 + p2", "parameters": 3,
              "types": ["signed int64", "float64"], "numeric_type_combinations": 8,
              "device_jit_cache": "fresh per strategy/series; adaptivity level explicitly 1; OS caches not flushed"}

    def run(label, argv, env=None):
        time_file, errors = output / f"{label}.time.json", output / f"{label}.stderr"
        command = timed_command(argv, time_file, errors)
        code, stdout, stderr, timeout = run_session(command, "", source, env or os.environ.copy(), 240)
        (output / f"{label}.stdout").write_text(stdout)
        timing = read_time(time_file)
        report["commands"].append({"label": label, "argv": [str(x) for x in argv], **timing, "returncode": code})
        if code or timeout:
            raise RuntimeError(f"{label} failed: {errors.read_text()[-4000:]} {stderr}")
        return stdout, timing

    try:
        run("version", [acpp, "--acpp-version"])
        base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC",
                "-ffp-contract=off", "-g0", *args.extra_flag]
        if args.bounded_four:
            base += ["-DSTAGE0_FOUR_PARAMS=1"]
            report.update({"parameters": 4, "numeric_type_combinations": 16,
                           "expression": "(input + p0) * p1 + p2 + p3", "typed_limit": 8})
        run("host-compile", [*base, "-O3", "-DNDEBUG", "-MD", "-MF", output / "host.d",
                             "-c", source / "typed_params_main.cpp", "-o", output / "host.o"])
        run("host-link", [*base, "-fuse-ld=lld", output / "host.o", "-ldl", "-o", output / "types"])
        pch = output / "types.pch"
        _, pch_time = run("pch", [*base, "-O0", "-c", "-fpch-instantiate-templates",
            f"-ftime-trace={output / 'pch-trace.json'}", "-x", "c++-header", source / "pch.hpp", "-o", pch])
        if pch.read_bytes()[:4] != b"CPCH":
            raise RuntimeError("invalid PCH")
        report["pch_prepare_ms"] = pch_time["wall_ms"]
        flags = output / "compile.flags"
        flags.write_text("\n".join([*base[1:], "-O0", "-include-pch", str(pch)]) + "\n")
        strategies = ["bounded"] if args.bounded_four else ["lazy", "tags", "eager"]
        for series in range(args.series):
            for strategy in strategies[series % len(strategies):] + strategies[:series % len(strategies)]:
                name = f"s{series}-{strategy}"
                directory = output / name
                directory.mkdir()
                env = os.environ.copy()
                env["ACPP_APPDB_DIR"] = str(directory / "device-cache")
                env["ACPP_ADAPTIVITY_LEVEL"] = "1"
                stdout, timing = run(name, [output / "types", strategy, directory, acpp, source, flags, args.device_name], env)
                result = json.loads(stdout)
                expected = 1 if strategy == "tags" else 9 if strategy == "bounded" else 8
                if not result["verified"] or result["compiler_calls"] != expected or result["linker_calls"] != expected:
                    raise RuntimeError(f"{name}: unexpected compiler counters")
                if any(c["fixture"] > 0 and c["new_compiles"] for c in result["cases"]):
                    raise RuntimeError("typed NULL/validity change caused compilation")
                if strategy == "bounded":
                    firsts = [c for c in result["cases"] if c["fixture"] == 0]
                    if len(firsts) != 16 or [c["new_compiles"] for c in firsts] != [1] * 9 + [0] * 7:
                        raise RuntimeError("typed limit/fallback transition was not respected")
                modules = []
                names, identities = set(), set()
                for library in sorted(directory.glob("*.so")):
                    object_ids = re.findall(rb"object-id=([0-9]+)\n", library.read_bytes())
                    symbols = subprocess.check_output(["nm", "-a", library], text=True)
                    kernels = re.findall(r"\b__acpp_free_kernel_name_(\S+)", symbols)
                    if len(object_ids) != 1 or len(kernels) != 1 or object_ids[0] in identities or kernels[0] in names:
                        raise RuntimeError("type module identity/kernel collision")
                    identities.update(object_ids); names.update(kernels)
                    stem = directory / library.stem
                    if "sycl/sycl.hpp" in stem.with_suffix(".d").read_text():
                        raise RuntimeError("full SYCL dependency in type module")
                    modules.append({"name": library.stem, "size_bytes": library.stat().st_size,
                        "compile_time": read_time(str(stem) + "-compile.time.json"),
                        "link_time": read_time(str(stem) + "-link.time.json"),
                        "compiler_trace_ms": trace_summary(str(stem) + "-compile-trace.json"),
                        "linker_trace_ms": trace_summary(str(stem) + "-link-trace.json")})
                if len(modules) != expected:
                    raise RuntimeError("missing type modules")
                report["measurements"].append({"strategy": strategy, "series": series, "result": result,
                    "modules": modules, "total_module_bytes": sum(m["size_bytes"] for m in modules),
                    "process_time": timing})
                print(f"{name}: mixed arithmetic, typed NULL, independent slots verified; {expected} modules", flush=True)
        report["success"] = True
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
