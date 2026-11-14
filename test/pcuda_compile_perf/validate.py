#!/usr/bin/env python3
"""Validate replayed toolchain commands on independent stage-0 GPU fixtures."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from run import ROOT, STAGE, native, sha256, trace_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-compile-perf"))
    args = parser.parse_args()
    if args.series < 1:
        parser.error("series must be positive")
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="validation-", dir=args.output_root.resolve()))
    print(f"Artifacts: {out}", flush=True)
    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC", "-g0", *args.extra_flag]
    env = {**os.environ, "LC_ALL": "C", "ACPP_ADAPTIVITY_LEVEL": "1", "ACPP_ALLOCATION_TRACKING": "1"}
    report = {"success": False, "commands": [], "sessions": [], "scope": "correctness fixtures; runtime process wall is NOT summary"}

    def command(prefix, argv, command_env=env):
        code = native(argv, prefix, command_env)
        report["commands"].append({"prefix": str(prefix), "argv": [str(x) for x in argv], "returncode": code})
        if code:
            raise RuntimeError(f"command failed; see {prefix}.stderr")

    def dryrun(argv, path):
        result = subprocess.run([str(argv[0]), "--acpp-dryrun", *(str(x) for x in argv[1:])],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        path.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("dryrun failed")
        return shlex.split(result.stdout)

    suites = {"modules": ("main", ("seed", "transform")), "collectives": ("collectives_main",
              ("block_scan", "add_offsets", "block_reduce", "block_reduce_unsigned", "atomic_probe"))}
    try:
        for suite, (host, _) in suites.items():
            command(out / (suite + "-host-compile"), [*base, "-O3", "-DNDEBUG", "-c", STAGE / (host + ".cpp"), "-o", out / (suite + ".o")])
            command(out / (suite + "-host-link"), [*base, out / (suite + ".o"), "-ldl", "-fuse-ld=lld", "-o", out / suite])
        for series in range(args.series):
            for variant in (("baseline", "replay") if series % 2 == 0 else ("replay", "baseline")):
                directory = out / f"s{series}-{variant}"
                directory.mkdir()
                pch = directory / "headers.pch"
                command(directory / "pch", [*base, "-O0", "-fpch-instantiate-templates", "-x", "c++-header", "-c", STAGE / "pch.hpp",
                                            f"-ftime-trace={directory / 'pch-trace.json'}", "-o", pch])
                for _, modules in suites.values():
                    for module in modules:
                        obj, so = directory / (module + ".o"), directory / (module + ".so")
                        compile_argv = [*base, "-O0", "-include-pch", pch, f"-ftime-trace={directory / (module + '-compile-trace.json')}",
                                        "-MD", "-MF", directory / (module + ".d"), "-c", STAGE / (module + ".cpp"), "-o", obj]
                        compile_effective = dryrun(compile_argv, directory / (module + "-compile-dryrun.txt"))
                        command(directory / (module + "-compile"), compile_argv if variant == "baseline" else compile_effective)
                        link_argv = [*base, "-shared", "-fuse-ld=lld", "-Wl,-O0", "-Wl,--threads=1",
                                     f"-Wl,--time-trace={directory / (module + '-link-trace.json')}", obj, "-o", so]
                        link_effective = dryrun(link_argv, directory / (module + "-link-dryrun.txt"))
                        if variant == "replay":
                            result = subprocess.run([*link_effective, "-###"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                            (directory / (module + "-lld-dryrun.txt")).write_text(result.stdout + result.stderr)
                            lines = [shlex.split(line) for line in result.stderr.splitlines() if line.lstrip().startswith('"')]
                            lld = [line for line in lines if Path(line[0]).name == "ld.lld"]
                            if result.returncode or len(lld) != 1:
                                raise RuntimeError("cannot recover LLD command")
                            link_argv = lld[0]
                        command(directory / (module + "-link"), link_argv)
                for suite, (_, modules) in suites.items():
                    cache = directory / (suite + "-device-cache")
                    command(directory / suite, [out / suite, *(directory / (m + ".so") for m in modules), args.device_name],
                            {**env, "ACPP_APPDB_DIR": str(cache)})
                report["sessions"].append({"variant": variant, "series": series, "directory": str(directory)})
                print(f"s{series}-{variant}: completed", flush=True)
        # Offline analysis starts only after all measured commands have terminated.
        for entry in report["commands"]:
            prefix = Path(entry["prefix"])
            timing = json.loads(Path(str(prefix) + ".time.json").read_text())
            entry["native_ms"] = {k.replace("_s", "_ms"): v * 1000 for k, v in timing.items()}
        all_ids = []
        for session in report["sessions"]:
            directory = Path(session["directory"])
            session["module_artifacts"] = []
            for _, modules in suites.values():
                for module in modules:
                    so = directory / (module + ".so")
                    ids = re.findall(rb"object-id=([0-9]+)\n", so.read_bytes())
                    symbols = subprocess.check_output(["nm", "-a", so], text=True)
                    (directory / (module + "-symbols.txt")).write_text(symbols)
                    kernels = re.findall(r"\b__acpp_free_kernel_name_(\S+)", symbols)
                    if len(ids) != 1 or len(kernels) != 1 or "sycl/sycl.hpp" in (directory / (module + ".d")).read_text():
                        raise RuntimeError("invalid module structure/dependencies")
                    all_ids += ids
                    item = {"name": module, "sha256": sha256(so), "size_bytes": so.stat().st_size,
                            "hcf_id": ids[0].decode(), "kernel_name": kernels[0]}
                    for phase in ("compile", "link"):
                        item[phase + "_trace_ms"] = trace_summary(directory / (module + "-" + phase + "-trace.json"))
                    session["module_artifacts"].append(item)
            for suite in suites:
                execution = json.loads((directory / (suite + ".stdout")).read_text())
                session[suite] = execution
                if not execution.get("verified"):
                    raise RuntimeError("fixture verification failed")
        report["unique_hcf_ids"] = len(all_ids) == len(set(all_ids))
        report["sha256"] = {str(p): sha256(p) for p in [*STAGE.glob("*.hpp"), *STAGE.glob("*.cpp"), Path(__file__),
                                                      out / "modules", out / "collectives"]}
        report["success"] = report["unique_hcf_ids"]
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "report.json", flush=True)


if __name__ == "__main__":
    main()
