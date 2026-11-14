#!/usr/bin/env python3
"""Rotated PCUDA compile/link/cold-start experiments; analysis only after the series."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import tempfile
import time


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
STAGE = SOURCE.parent / "pcuda_stage0"
MODULES = ("date", "customer", "supplier", "probe")
VARIANTS = {
    "baseline": {},
    "no-pch": {"pch": "none"},
    "pch-no-instantiate": {"instantiate": False},
    "pch-preamble": {"pch": "preamble"},
    "no-exceptions": {"flags": ["-fno-exceptions"]},
    "no-rtti": {"flags": ["-fno-rtti"]},
    "no-unwind": {"flags": ["-fno-unwind-tables", "-fno-asynchronous-unwind-tables"]},
    "omit-frame": {"flags": ["-fomit-frame-pointer"]},
    "preoptimize": {"flags": ["-mllvm", "-acpp-sscp-preoptimize"], "unsupported_developer_option": True},
    "link-O1": {"link_flags": ["-Wl,-O1"]},
    "link-threads1": {"link_flags": ["-Wl,--threads=1"]},
    "link-threads4": {"link_flags": ["-Wl,--threads=4"]},
    "link-gc": {"link_flags": ["-Wl,--gc-sections"]},
    "link-bfd": {"linker": "bfd"},
    "link-gold": {"linker": "gold"},
    "link-icf-safe": {"link_flags": ["-Wl,--icf=safe"]},
    "link-strip": {"link_flags": ["-Wl,--strip-all"]},
    "link-sysv-hash": {"link_flags": ["-Wl,--hash-style=sysv"]},
    "clang-link": {"link_driver": "clang-replay"},
    "lld-link": {"link_driver": "lld-replay"},
    "clang-compile": {"compile_driver": "clang-replay"},
    "parallel4": {"jobs": 4},
    "adaptivity0": {"environment": {"ACPP_ADAPTIVITY_LEVEL": "0"}},
    "adaptivity2": {"environment": {"ACPP_ADAPTIVITY_LEVEL": "2"}},
    "no-tracking": {"environment": {"ACPP_ALLOCATION_TRACKING": "0"}},
    "fast-link": {"link_driver": "lld-replay", "link_flags": ["-Wl,--threads=1"]},
    "parallel-fast-link": {"jobs": 4, "link_driver": "lld-replay", "link_flags": ["-Wl,--threads=1"]},
    "replay-parallel": {"jobs": 4, "compile_driver": "clang-replay", "link_driver": "lld-replay",
                        "link_flags": ["-Wl,--threads=1"]},
    "replay-adaptivity0": {"jobs": 4, "compile_driver": "clang-replay", "link_driver": "lld-replay",
                           "link_flags": ["-Wl,--threads=1"], "environment": {"ACPP_ADAPTIVITY_LEVEL": "0"}},
    "preopt-replay-parallel": {"jobs": 4, "compile_driver": "clang-replay", "link_driver": "lld-replay",
                               "link_flags": ["-Wl,--threads=1"], "flags": ["-mllvm", "-acpp-sscp-preoptimize"],
                               "unsupported_developer_option": True},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    return {"n": len(values), "raw": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None


def trace_summary(path):
    events = json.loads(path.read_text())["traceEvents"]
    names = ("ExecuteCompiler", "Total Frontend", "Total Backend", "Total Optimizer",
             "Total CodeGenPasses", "ExecuteLinker", "Total InstantiateFunction", "Total PerformPendingInstantiations")
    return {name: max(e.get("dur", 0) for e in events if e.get("name") == name) / 1000
            for name in names if any(e.get("name") == name for e in events)}


def check_query_timing(execution):
    """Reject the old prepared-buffer boundary when reporting V2 summary."""
    if execution.get("method_version") != 2 or execution.get("preparation_in_summary") is not True:
        raise ValueError("V2 query preparation boundary is not declared")
    values = [execution[k] for k in ("summary_ms", "preparation_ms", "build_elapsed_ms", "load_ms")]
    values.append(execution["runs"][0]["total_ms"])
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid query timing")
    remainder = values[0] - sum(values[1:])
    if remainder < -0.01:
        raise ValueError("query preparation is outside summary or components overlap")
    return remainder  # Keep real gaps; do not force the remainder to zero.


def native(argv, prefix, env=None, cwd=ROOT):
    # No Python timer or proxy in this measured chain. Read files offline later.
    script = STAGE / "native_command.sh"
    with prefix.with_suffix(".stdout").open("w") as output:
        process = subprocess.Popen(["bash", str(script), str(prefix), *(str(x) for x in argv)],
                                   cwd=cwd, env=env, stdout=output, start_new_session=True)
        try:
            return process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--variants", default="baseline,no-pch,pch-no-instantiate,pch-preamble")
    parser.add_argument("--diagnostics", choices=("both", "on", "off"), default="both")
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-compile-perf"))
    args = parser.parse_args()
    names = args.variants.split(",")
    if not names or any(n not in VARIANTS for n in names) or args.series < 1 or not 2 <= args.runs <= 50:
        parser.error("invalid variants/series/runs")
    acpp = shutil.which(args.acpp)
    if not acpp:
        parser.error("acpp not found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="series-", dir=args.output_root.resolve()))
    print(f"Artifacts: {out}", flush=True)
    base = [acpp, "--acpp-targets=generic", "--acpp-pcuda", "-std=c++17", "-fPIC", "-g0", *args.extra_flag]
    env = {**os.environ, "LC_ALL": "C", "ACPP_ADAPTIVITY_LEVEL": "1", "ACPP_ALLOCATION_TRACKING": "1"}
    commands, sessions = [], []
    report = {"success": False, "method_version": 2, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "arguments": vars(args) | {"data": str(args.data.resolve()), "output_root": str(args.output_root)},
              "commands": commands, "sessions": sessions, "scope": "PCUDA prototype; no SQL parse/plan/codegen",
              "cache_policy": "new process and empty ACPP_APPDB_DIR per session; OS caches NOT flushed",
              "environment": {k: v for k, v in env.items() if k.startswith(("ACPP_", "HIP_", "ROCM_", "OMP_"))
                              or k in ("LC_ALL", "PATH", "LD_LIBRARY_PATH", "PYTHONPATH")}}

    def prepare(label, argv):
        argv = [str(x) for x in argv]
        code = native(argv, out / label, env)
        commands.append({"label": label, "argv": argv, "returncode": code, "preparation": True})
        if code:
            raise RuntimeError(f"preparation failed: {label}; see {out / (label + '.stderr')}")

    def effective(argv, path):
        result = subprocess.run([str(argv[0]), "--acpp-dryrun", *(str(x) for x in argv[1:])],
                                capture_output=True, text=True, env=env, cwd=ROOT, timeout=30)
        path.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("dryrun failed: " + str(path))
        parsed = shlex.split(result.stdout.strip())
        if not parsed or not Path(parsed[0]).is_file():
            raise RuntimeError("cannot parse effective command: " + str(path))
        return parsed

    try:
        prepare("host-compile", [*base, "-O3", "-DNDEBUG", "-c", SOURCE / "q33_cold.cpp", "-o", out / "host.o"])
        prepare("host-link", [*base, out / "host.o", "-ldl", "-fuse-ld=lld", "-o", out / "q33_cold"])
        replay = effective([*base, "-O0", "-c", STAGE / "q33_probe.cpp", "-o", out / "unused.o"], out / "compiler-dryrun.txt")
        compiler = replay[0]
        prepare("launcher", [compiler, *args.extra_flag, "-std=c++17", "-O3", SOURCE / "first_result.cpp", "-o", out / "first_result"])
        diagnostics = [False, True] if args.diagnostics == "both" else [args.diagnostics == "on"]
        variants = [(name, diagnostic) for name in names for diagnostic in diagnostics]
        for series in range(args.series):
            # Rotated variant order, alternating diagnostics order inside each variant.
            rotated = names[series % len(names):] + names[:series % len(names)]
            ordered = [(n, d) for n in rotated for d in (diagnostics if series % 2 == 0 else diagnostics[::-1])]
            for name, diagnostic in ordered:
                spec = VARIANTS[name]
                label = f"s{series}-{name}-{'trace' if diagnostic else 'plain'}"
                directory = out / label
                directory.mkdir()
                session_env = {**env, **spec.get("environment", {}), "ACPP_APPDB_DIR": str(directory / "device-cache")}
                pch_mode = spec.get("pch", "minimal")
                flags = [*base, "-O0", *spec.get("flags", [])]
                pch = directory / "headers.pch"
                pch_label = label + "-pch"
                if pch_mode != "none":
                    header = STAGE / ("q33_build.hpp" if pch_mode == "preamble" else "pch.hpp")
                    prepare(pch_label, [*flags, *( [f"-ftime-trace={directory / 'pch-trace.json'}"] if diagnostic else []),
                        *(["-fpch-instantiate-templates"] if spec.get("instantiate", True) else []),
                        "-MD", "-MF", directory / "pch.d", "-c", "-x", "c++-header", header, "-o", pch])
                    if pch.read_bytes()[:4] != b"CPCH":
                        raise RuntimeError("not a Clang PCH")
                    flags += ["-include-pch", pch]
                build_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "export LC_ALL=C",
                    "TIMEFORMAT='{\"wall_s\":%6R,\"user_s\":%6U,\"system_s\":%6S}'",
                    'timed() { local prefix=$1; shift; { time "$@" >"$prefix.stdout" 2>"$prefix.stderr"; } 2>"$prefix.time.json"; }']
                pipelines = []
                session_commands = []
                for module in MODULES:
                    obj, so = directory / (module + ".o"), directory / (module + ".so")
                    compile_argv = [*flags, *([f"-ftime-trace={directory / (module + '-compile-trace.json')}"] if diagnostic else []),
                        "-MD", "-MF", directory / (module + ".d"), "-c", STAGE / f"q33_{module}.cpp", "-o", obj]
                    link_argv = [*base, "-O0", "-shared", "-fuse-ld=" + spec.get("linker", "lld"), "-Wl,-O0",
                        *spec.get("link_flags", []), *([f"-Wl,--time-trace={directory / (module + '-link-trace.json')}"]
                            if diagnostic and spec.get("linker", "lld") == "lld" else []), obj, "-o", so]
                    compiler_effective = effective(compile_argv, directory / (module + "-compile-dryrun.txt"))
                    linker_effective = effective(link_argv, directory / (module + "-link-dryrun.txt"))
                    if spec.get("compile_driver") == "clang-replay":
                        compile_argv = compiler_effective
                    if spec.get("link_driver") == "clang-replay":
                        link_argv = linker_effective
                    elif spec.get("link_driver") == "lld-replay":
                        # Use an existing object solely to make -### retain the input argument.
                        dry_argv = [str(out / "host.o") if str(x) == str(obj) else str(x) for x in linker_effective]
                        dry = subprocess.run([*dry_argv, "-###"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=30)
                        (directory / (module + "-lld-dryrun.txt")).write_text(dry.stdout + dry.stderr)
                        lines = [shlex.split(line) for line in dry.stderr.splitlines() if line.lstrip().startswith('"')]
                        lld = [x for x in lines if Path(x[0]).name == "ld.lld"]
                        if dry.returncode or len(lld) != 1:
                            raise RuntimeError("cannot recover exact LLD command")
                        link_argv = [str(obj) if x == str(out / "host.o") else x for x in lld[0]]
                    pipeline = []
                    for phase, argv in (("compile", compile_argv), ("link", link_argv)):
                        prefix = directory / (module + "-" + phase)
                        pipeline.append("timed " + shlex.join([str(prefix), *(str(x) for x in argv)]))
                        session_commands.append({"module": module, "phase": phase, "argv": [str(x) for x in argv],
                                                 "prefix": str(prefix)})
                    pipelines.append(pipeline)
                if spec.get("jobs", 1) == 4:
                    build_lines.append("pids=()")
                    for pipeline in pipelines:
                        build_lines += ["(", *pipeline, ") &", 'pids+=("$!")']
                    build_lines += ['status=0', 'for pid in "${pids[@]}"; do wait "$pid" || status=1; done', 'exit "$status"']
                else:
                    build_lines += [line for pipeline in pipelines for line in pipeline]
                script = directory / "build.sh"
                script.write_text("\n".join(build_lines) + "\n")
                (directory / "commands.json").write_text(json.dumps(session_commands, indent=2) + "\n")
                argv = [out / "first_result", directory / "first-result.json", out / "q33_cold", args.data.resolve(),
                        args.device_name, script, directory, str(args.runs)]
                code = native(argv, directory / "process", session_env)
                sessions.append({"name": name, "series": series, "diagnostic": diagnostic, "directory": str(directory),
                    "specification": spec, "argv": [str(x) for x in argv], "returncode": code,
                    "device_jit_cache": session_env["ACPP_APPDB_DIR"], "environment": session_env | {},
                    "pch_preparation_label": pch_label if pch_mode != "none" else None})
                # Persist raw progress, but defer result/timing/trace analysis until all runs finish.
                (out / "progress.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "environment"} for s in sessions], indent=2) + "\n")
                print(f"{label}: exit={code}", flush=True)
        # Everything below is offline, after the final measured process has exited.
        expected = Counter()
        for line in (ROOT / "test/queries/result/scale10/33.txt").read_text().splitlines():
            if line.startswith("Result count:"): break
            expected[tuple(map(int, line.split()))] += 1
        for command in commands:
            prefix = out / command["label"]
            timing = json.loads(Path(str(prefix) + ".time.json").read_text())
            command["native_ms"] = {k.replace("_s", "_ms"): v * 1000 for k, v in timing.items()}
        for session in sessions:
            directory = Path(session["directory"])
            session["environment"] = {k: v for k, v in session["environment"].items()
                                      if k.startswith(("ACPP_", "HIP_", "ROCM_", "OMP_")) or k == "LC_ALL"}
            if session["returncode"]:
                session["verified"] = False
                continue
            result = json.loads((directory / "process.stdout").read_text())
            session["query_timing_remainder_ms"] = check_query_timing(result)
            session["execution"] = result
            session["first_result"] = json.loads((directory / "first-result.json").read_text())
            session["verified"] = all(Counter(tuple(row) for row in r["result"]) == expected for r in result["runs"])
            session["modules"] = []
            for module in MODULES:
                item = {"name": module, "size_bytes": (directory / (module + ".so")).stat().st_size,
                        "sha256": sha256(directory / (module + ".so"))}
                for phase in ("compile", "link"):
                    timing = json.loads((directory / (module + "-" + phase + ".time.json")).read_text())
                    item[phase + "_native_ms"] = {k.replace("_s", "_ms"): v * 1000 for k, v in timing.items()}
                    trace = directory / (module + "-" + phase + "-trace.json")
                    if trace.exists(): item[phase + "_trace_ms"] = trace_summary(trace)
                item["hcf_ids"] = [x.decode() for x in re.findall(rb"object-id=([0-9]+)\n", (directory / (module + ".so")).read_bytes())]
                deps = (directory / (module + ".d")).read_text()
                if "sycl/sycl.hpp" in deps:
                    session["verified"] = False
                session["modules"].append(item)
            ids = [h for m in session["modules"] for h in m["hcf_ids"]]
            session["unique_hcf_ids"] = len(ids) == 4 and len(set(ids)) == 4
            session["verified"] &= session["unique_hcf_ids"]
            if (directory / "headers.pch").exists():
                session["pch_sha256"] = sha256(directory / "headers.pch")
                session["pch_size_bytes"] = (directory / "headers.pch").stat().st_size
            session["compile_wall_sum_ms"] = sum(m["compile_native_ms"]["wall_ms"] for m in session["modules"])
            session["link_wall_sum_ms"] = sum(m["link_native_ms"]["wall_ms"] for m in session["modules"])
            session["component_sum_ms"] = session["compile_wall_sum_ms"] + session["link_wall_sum_ms"] + result["load_ms"] + result["runs"][0]["total_ms"]
            session["summary_minus_components_ms"] = result["summary_ms"] - session["component_sum_ms"]
            session["summary_minus_build_load_first_ms"] = result["summary_ms"] - result["build_elapsed_ms"] - result["load_ms"] - result["runs"][0]["total_ms"]
            session["summary_minus_prepare_build_load_first_ms"] = session["summary_minus_build_load_first_ms"] - result["preparation_ms"]
        aggregate = []
        for name, diagnostic in variants:
            group = [s for s in sessions if s["name"] == name and s["diagnostic"] == diagnostic and s.get("verified")]
            entry = {"name": name, "diagnostic": diagnostic, "successful_sessions": len(group), "method_version": 2}
            for metric in ("summary_ms", "build_elapsed_ms", "load_ms", "startup_ms", "data_load_ms", "preparation_ms"):
                entry[metric] = stats([s["execution"][metric] for s in group])
            for metric in ("compile_wall_sum_ms", "link_wall_sum_ms", "summary_minus_components_ms", "summary_minus_build_load_first_ms", "summary_minus_prepare_build_load_first_ms"):
                entry[metric] = stats([s[metric] for s in group])
            entry["process_start_to_first_result_ms"] = stats([s["first_result"]["process_start_to_first_result_ms"] for s in group])
            for metric in ("execute_ms", "fetch_ms", "total_ms"):
                entry["first_" + metric] = stats([s["execution"]["runs"][0][metric] for s in group])
                entry["warm_" + metric] = stats([r[metric] for s in group for r in s["execution"]["runs"][1:]])
            entry["module_bytes_sum"] = stats([sum(m["size_bytes"] for m in s["modules"]) for s in group])
            aggregate.append(entry)
        report["aggregate"] = aggregate
        inputs = list(SOURCE.glob("*.cpp")) + list(SOURCE.glob("*.py")) + list(STAGE.glob("*.hpp")) + list(STAGE.glob("q33*.cpp"))
        inputs += [Path(acpp), Path(compiler).resolve(), out / "q33_cold", out / "first_result"]
        inputs += [args.data.resolve() / n for n in ("DDATE0", "DDATE4", "CUSTOMER0", "CUSTOMER3", "SUPPLIER0", "SUPPLIER3", "LINEORDER5", "LINEORDER2", "LINEORDER4", "LINEORDER12")]
        report["sha256"] = {str(p): sha256(p) for p in inputs}
        report["success"] = all(s.get("verified") for s in sessions)
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Offline verification: {report['success']}; {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
