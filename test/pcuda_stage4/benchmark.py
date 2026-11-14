#!/usr/bin/env python3
## @file
## @brief Rotate prebuilt primitive comparisons; Python is outside every native measured chain.
"""No SQL, PCH or module cache is benchmarked. Baseline headers must be preserved beforehand."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/pcuda_stage0"))
from native_timing import timed_command, read_time


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--baseline-include", type=Path, required=True)
    parser.add_argument("--acpp", type=Path, required=True)
    parser.add_argument("--cxx", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--gcc-install-dir", type=Path)
    parser.add_argument("--cold", type=int, default=7)
    parser.add_argument("--build-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.cold < 1 or args.build_repeats < 1:
        parser.error("positive repetitions required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="primitive-bench-", dir=args.output_root.resolve()))
    flags = [f"--gcc-install-dir={args.gcc_install_dir.resolve()}"] if args.gcc_install_dir else []
    sdk, engine = args.sdk.resolve(), args.engine.resolve()
    report = {"success": False, "method_version": 2, "builds": [], "verifications": [], "runs": [], "method": "prebuilt primitives; no SQL; summary includes lazy buffers, initialization, execution and fetch; CPU oracle runs only in separate correctness processes, printing excluded", "cache": "new ACPP_APPDB_DIR per process, retained within all operations and warm repeats; OS/HIP/driver caches not cleared", "commands": []}

    def run(name, command, env=None, measured=True):
        prefix = output / name
        command = [str(x) for x in command]
        (output / (name + ".command.json")).write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n")
        actual = timed_command(command, str(prefix) + ".time.json", str(prefix) + ".stderr") if measured else command
        with Path(str(prefix) + ".stdout").open("w") as stdout, Path(str(prefix) + ".driver-stderr").open("w") as stderr:
            result = subprocess.run(actual, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=300)
        entry = {"name": name, "argv": command, "returncode": result.returncode}
        report["commands"].append(entry)
        with (output / "actions.jsonl").open("a") as log:
            log.write(json.dumps({**entry, "reason": "paired native primitive comparison"}, ensure_ascii=False) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {prefix}.stderr")
        return read_time(str(prefix) + ".time.json") if measured else {}

    binaries = {}
    try:
        # Alternate mode/flavour order between independent build replications.
        variants = [(flavour, traced) for traced in (False, True) for flavour in ("baseline", "pcuda")]
        for repetition in range(args.build_repeats):
            for flavour, traced in variants[::1 if repetition % 2 == 0 else -1]:
                name = f"{flavour}-{'trace' if traced else 'plain'}-build{repetition}"
                directory = output / name
                directory.mkdir()
                define = ["-DSTAGE4_BASELINE"] if flavour == "baseline" else []
                include = ["-I" + str(args.baseline_include.resolve())] if define else []
                common = ["-std=c++17", *flags, *define, *include, "-I" + str(ROOT / "include")]
                gpu = [args.acpp.resolve(), "--acpp-targets=generic", "--acpp-pcuda", *common, "-O0", "-g0", "-c", ROOT / "test/pcuda_stage4/benchmark_kernels.cpp", "-o", directory / "kernels.o"]
                if traced:
                    gpu += ["-ftime-trace"]
                compile_time = run(name + "-compile", gpu)
                if repetition == 0:
                    run(name + "-dryrun", [*gpu, "--acpp-dryrun"], measured=False)
                host = [args.cxx.resolve(), *common, "-isystem" + str(sdk / "include/AdaptiveCpp"), "-O3", "-DNDEBUG", "-c", ROOT / "test/pcuda_stage4/benchmark.cpp", "-o", directory / "host.o"]
                run(name + "-host", host)
                binary = directory / "benchmark"
                libraries = [engine / "liblegacy_plan_runtime.a", engine / "libpcuda_runtime.a", engine / "libphysical_execution.a", engine / "libquery_values.a", sdk / "lib/libacpp-rt.so", sdk / "lib/libacpp-common.so"]
                link = [args.acpp.resolve(), "--acpp-targets=generic", "--acpp-pcuda", *flags, "-fuse-ld=lld", "-Wl,-O0", "-Wl,-rpath," + str(sdk / "lib"), directory / "host.o", directory / "kernels.o", *libraries, "-ldl", "-o", binary]
                if traced:
                    link += ["-Wl,--time-trace=" + str(directory / "link.json")]
                link_time = run(name + "-link", link)
                if repetition == 0:
                    run(name + "-link-dryrun", [*link, "--acpp-dryrun", "-###"], measured=False)
                report["builds"].append({"flavour": flavour, "traced": traced, "repeat": repetition, "compile": compile_time, "link": link_time, "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "bytes": binary.stat().st_size})
                binaries[flavour, traced] = binary
        # Verify identical binaries separately: no CPU oracle runs inside the
        # measured processes or between their warm query intervals.
        for flavour, traced in variants:
            name = f"{flavour}-{'trace' if traced else 'plain'}-verify"
            env = os.environ.copy()
            env.update({"LC_ALL": "C", "ACPP_ADAPTIVITY_LEVEL": "1", "ACPP_ALLOCATION_TRACKING": "1", "ACPP_APPDB_DIR": str(output / (name + "-device-cache"))})
            run(name, [binaries[flavour, traced], 0, "verify"], env, measured=False)
            report["verifications"].append({"flavour": flavour, "traced": traced, "results": 36, "success": True})
        # No concurrent compiler/CPU oracle analysis runs during native process timing.
        for repetition in range(args.cold):
            order = variants[::1 if repetition % 2 == 0 else -1]
            for flavour, traced in order:
                name = f"{flavour}-{'trace' if traced else 'plain'}-run{repetition}"
                env = os.environ.copy()
                env.update({"LC_ALL": "C", "ACPP_ADAPTIVITY_LEVEL": "1", "ACPP_ALLOCATION_TRACKING": "1", "ACPP_APPDB_DIR": str(output / (name + "-device-cache"))})
                native = run(name, [binaries[flavour, traced], repetition % 4], env)
                records = [json.loads(line) for line in (output / (name + ".stdout")).read_text().splitlines() if line.startswith("{")]
                report["runs"].append({"flavour": flavour, "traced": traced, "repeat": repetition, "native": native, "startup": records[0], "records": records[1:], "environment": {k: env[k] for k in ("LC_ALL", "ACPP_ADAPTIVITY_LEVEL", "ACPP_ALLOCATION_TRACKING", "ACPP_APPDB_DIR")}})
        report["aggregates"] = []
        for flavour, traced in variants:
            for op in range(4):
                rows = [row for run in report["runs"] if run["flavour"] == flavour and run["traced"] == traced for row in run["records"] if row["op"] == op]
                report["aggregates"].append({"flavour": flavour, "traced": traced, "operation": op,
                    **{f"{mode}_{field}": distribution([r[field] for r in rows if (r["repeat"] == 0) == (mode == "cold")]) for mode in ("cold", "warm") for field in ("execute_ms", "fetch_ms", "summary_ms")}})
        report["success"] = True
    finally:
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "test/pcuda_stage4").glob("benchmark*") if p.is_file()}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Artifacts:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
