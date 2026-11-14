#!/usr/bin/env python3
"""Isolated PCUDA feasibility probe; never modifies the engine or user experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from native_timing import timed_command, read_time, trace_summary
from sql_baseline import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--device-name", required=True, help="PCUDA device name substring")
    parser.add_argument("--suite", choices=("modules", "collectives", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    source = Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    report = {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.iterdir()) if p.suffix in (".cpp", ".hpp", ".py", ".sh")},
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("ACPP_", "HIPSYCL_", "HSA_", "HIP_", "ROCR_"))},
        "acpp": acpp,
        "acpp_driver_sha256": hashlib.sha256(Path(acpp).read_bytes()).hexdigest(),
        "device_name_substring": args.device_name,
        "suite": args.suite,
        "device_jit_cache": "fresh isolated directory per variant; filesystem/OS caches not flushed",
        "commands": [],
        "variants": [],
        "success": False,
        "measurement": "native Bash time + offline Clang/lld traces; no Python measurement wrapper; acpp driver overhead retained",
    }

    def run(label, command, env=None):
        time_file, errors = output / f"{label}.time.json", output / f"{label}.command.stderr"
        timed = timed_command(command, time_file, errors)
        code, stdout, timer_stderr, timeout = run_session(timed, "", Path.cwd(), env or os.environ.copy(), args.timeout)
        stderr = errors.read_text() + timer_stderr
        native = read_time(time_file)
        wall_ms = native["wall_ms"]
        (output / f"{label}.stdout").write_text(stdout)
        (output / f"{label}.stderr").write_text(stderr)
        report["commands"].append({"label": label, "argv": [str(x) for x in command],
                                   **native, "returncode": code})
        if code or timeout:
            raise RuntimeError(f"{label} failed: {stderr[-4000:]}")
        return stdout, wall_ms

    try:
        run("acpp-version", [acpp, "--acpp-version"])
        run("clang-version", [acpp, "--version"])
        base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17",
                "-fPIC", "-g0", *args.extra_flag]
        report["base_flags"] = [str(x) for x in base]

        def check_dependencies(path):
            # PCUDA's internal libkernel headers are expected; the full SYCL API is not.
            if re.search(r"[/\\]sycl[/\\]sycl\.hpp(?:\s|$)", path.read_text()):
                raise RuntimeError(f"full SYCL header found in {path}")

        suites = {}
        if args.suite in ("modules", "all"):
            suites["probe"] = ("main", ("seed", "transform"), "execution")
        if args.suite in ("collectives", "all"):
            suites["collectives"] = ("collectives_main", ("block_scan", "add_offsets", "block_reduce",
                                                         "block_reduce_unsigned", "atomic_probe"),
                                      "collectives_execution")
        for executable, (main_source, _, _) in suites.items():
            obj = output / f"{executable}-host.o"
            deps = obj.with_suffix(".d")
            run(f"{executable}-host-compile", [*base, "-O3", "-DNDEBUG", "-MD", "-MF", deps,
                                              "-c", source / f"{main_source}.cpp", "-o", obj])
            check_dependencies(deps)
            run(f"{executable}-host-link", [*base, "-fuse-ld=lld", obj, "-ldl", "-o", output / executable])
        libraries, _ = run("host-libraries", ["ldd", output / next(iter(suites))])
        report["runtime_libraries_sha256"] = {
            str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in re.findall(r"libacpp[^\s]* => (\S+)", libraries)
        }
        hcf_ids = set()
        for optimization in ("-O0", "-O3"):
            # A distinct matching PCH is prepared for each optimization mode.
            pch = output / f"{optimization[1:]}.pch"
            _, pch_ms = run(f"{optimization[1:]}-pch", [*base, optimization, "-c",
                            f"-ftime-trace={output / (optimization[1:] + '-pch-trace.json')}",
                            "-MD", "-MF", output / f"{optimization[1:]}-pch.d",
                            "-fpch-instantiate-templates", "-x", "c++-header",
                            source / "pch.hpp", "-o", pch])
            if pch.read_bytes()[:4] != b"CPCH":
                raise RuntimeError("compiler did not produce a Clang PCH")
            check_dependencies(output / f"{optimization[1:]}-pch.d")
            for use_pch in (False, True):
                name = f"{optimization[1:]}-{'pch' if use_pch else 'no-pch'}"
                flags = [*base, optimization]
                if use_pch:
                    flags += ["-include-pch", pch]
                variant = {"name": name, "pch_prepare_ms": pch_ms, "modules": []}
                kernel_names = set()
                for module in (m for _, modules, _ in suites.values() for m in modules):
                    obj = output / f"{name}-{module}.o"
                    so = output / f"{name}-{module}.so"
                    _, compile_ms = run(f"{name}-{module}-compile",
                                        [*flags, f"-ftime-trace={output / (name + '-' + module + '-compile-trace.json')}", "-MD", "-MF", obj.with_suffix(".d"),
                                         "-c", source / f"{module}.cpp", "-o", obj])
                    check_dependencies(obj.with_suffix(".d"))
                    _, link_ms = run(f"{name}-{module}-link", [*base, "-shared", "-fuse-ld=lld", "-Wl,-O0",
                        f"-Wl,--time-trace={output / (name + '-' + module + '-link-trace.json')}", obj, "-o", so])
                    # Diagnostic of the tested AdaptiveCpp HCF format, not an engine API.
                    ids = re.findall(rb"object-id=([0-9]+)\n", so.read_bytes())
                    if len(ids) != 1 or ids[0] in hcf_ids:
                        raise RuntimeError(f"missing/duplicate HCF identifier in {so}")
                    hcf_ids.add(ids[0])
                    symbols, _ = run(f"{name}-{module}-symbols", ["nm", "-a", so])
                    kernels = re.findall(r"\b__acpp_free_kernel_name_(\S+)", symbols)
                    if len(kernels) != 1 or kernels[0] in kernel_names:
                        raise RuntimeError(f"missing/duplicate kernel name in {so}")
                    kernel_names.update(kernels)
                    variant["modules"].append({"name": module, "compile_ms": compile_ms,
                                                "link_ms": link_ms, "size_bytes": so.stat().st_size,
                                                "compiler_trace_ms": trace_summary(output / f"{name}-{module}-compile-trace.json"),
                                                "linker_trace_ms": trace_summary(output / f"{name}-{module}-link-trace.json"),
                                                "hcf_id": ids[0].decode(), "kernel_name": kernels[0]})
                for executable, (_, modules, result_field) in suites.items():
                    env = os.environ.copy()
                    # Separate fresh cache and process for each suite and build variant.
                    env["ACPP_APPDB_DIR"] = str(output / f"{name}-{executable}-device-cache")
                    stdout, wall_ms = run(f"{name}-{executable}-execute", [output / executable,
                        *(output / f"{name}-{m}.so" for m in modules), args.device_name], env)
                    result = json.loads(stdout)
                    if result.get("verified") is not True:
                        raise RuntimeError(f"{name}-{executable}: verification missing")
                    variant[result_field] = result
                    variant[f"{result_field}_process_wall_ms"] = wall_ms
                    variant[f"{result_field}_device_jit_cache_directory"] = env["ACPP_APPDB_DIR"]
                    if result_field == "execution":
                        variant["process_wall_ms"] = wall_ms
                        variant["device_jit_cache_directory"] = env["ACPP_APPDB_DIR"]
                report["variants"].append(variant)
                print(f"{name}: verified; " + ", ".join(
                    f"{m['name']} compile={m['compile_ms']:.1f} link={m['link_ms']:.1f} ms"
                    for m in variant["modules"]), flush=True)
        report["success"] = True
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
