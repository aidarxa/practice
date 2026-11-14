#!/usr/bin/env python3
"""Observe the existing SYCL db_cli without changing its compiler commands or metrics."""

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

REFERENCE_GROUPS = {
    "q11": (), "q12": (), "q13": (),
    "q21": ("d_year", "p_brand1"), "q22": ("d_year", "p_brand1"), "q23": ("d_year", "p_brand1"),
    "q31": ("d_year", "c_nation", "s_nation"),
    "q32": ("d_year", "c_city", "s_city"), "q33": ("d_year", "c_city", "s_city"),
    "q34": ("d_year", "c_city", "s_city"), "q41": ("d_year", "c_nation"),
    "q42": ("d_year", "s_nation", "p_category"), "q43": ("d_year", "s_city", "p_brand1"),
}


def isolate_sql(sql, nonce):
    # This installed SQL parser rejects block comments. Random internal whitespace
    # changes the legacy raw-SQL hash without adding tokens or changing the AST.
    match = re.match(r"(select)\s+", sql, re.I)
    if not match:
        raise ValueError("expected SELECT at start of SSB query")
    spacing = "".join("\t" if bit == "1" else " " for bit in f"{int(nonce, 16):0128b}")
    return match[1] + spacing + sql[match.end():]


def reference_rows(text):
    """Read the first independent result, not repeated runs or their timings."""
    scalar = re.search(r"^Result:\s*(-?\d+)\s*$", text, re.M)
    if scalar:
        return Counter({(int(scalar[1]),): 1})
    rows = []
    for line in text.splitlines():
        if line.startswith("Result count:"):
            if int(line.split(":", 1)[1]) != len(rows):
                raise ValueError("reference row count mismatch")
            return Counter(rows)
        if re.fullmatch(r"-?\d+(?:\s+-?\d+)+", line.strip()):
            rows.append(tuple(map(int, line.split())))
    raise ValueError("reference has no complete result")


def cli_rows(text, groups):
    match = re.search(r"--- Result ---\s*\n(.*?)Rows returned:\s*(\d+)", text, re.S)
    if not match:
        raise ValueError("missing CLI result")
    lines = match[1].strip().splitlines()
    if not lines or not lines[0].startswith("Columns:"):
        raise ValueError("missing result column names")
    columns = [c.strip().lower() for c in lines.pop(0).split(":", 1)[1].split("|")]
    if len(set(columns)) != len(columns) or len(columns) != len(groups) + 1:
        raise ValueError("unexpected result schema")
    indices = [columns.index(name) for name in groups]
    aggregate = [i for i in range(len(columns)) if i not in indices]
    indices += aggregate
    rows = []
    for line in lines:
        line = re.sub(r"^Row \d+:\s*", "", line)
        cells = tuple(int(c.strip()) for c in line.split("|"))
        if len(cells) != len(columns):
            raise ValueError("result row width mismatch")
        rows.append(tuple(cells[i] for i in indices))
    if len(rows) != int(match[2]):
        raise ValueError("truncated CLI result")
    return Counter(rows)


def run_session(command, stdin, cwd, env, timeout):
    # Kill the whole compiler/CLI process group on timeout before moving its PCH.
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, cwd=cwd, env=env,
                          start_new_session=True) as proc:
        timed_out = False
        try:
            stdout, stderr = proc.communicate(stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return proc.returncode, stdout, stderr, timed_out


def compiler_cache_verified(events, records, query_count, timing_model):
    """@brief Verify compiler reuse without mistaking stage-2 physical lowering for another acpp call."""
    processes = (sum(e["phase"] == "pch_prepare" for e in events) == 1 and
                 sum(e["phase"] == "compile_and_link" for e in events) == query_count)
    warm = [row for row in records if row["run"] > 0]
    compilation = all(row["compile_ms"] == 0 for row in warm)
    if timing_model == "explicit-plan-v1":
        return processes and compilation
    return processes and compilation and all(row["codegen_ms"] == 0 for row in warm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-cli", required=True, type=Path)
    parser.add_argument("--acpp", default="acpp")
    parser.add_argument("--acpp-include", required=True, type=Path)
    parser.add_argument("--query", choices=tuple(REFERENCE_GROUPS), action="append")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timing-model", choices=("legacy-generated-v1", "explicit-plan-v1"), default="legacy-generated-v1")
    parser.add_argument("--trace-compiler", action="store_true")
    parser.add_argument("--jit-optimization", choices=("O0", "O3"))
    parser.add_argument("--jit-pch", choices=("on", "off"))
    parser.add_argument("--native-time", action="store_true", default=True,
                        help="Native timing is always enabled; option retained for compatibility")
    parser.add_argument("--linker-opt", choices=("0", "1"))
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output-root", type=Path, default=Path("build/pcuda-stage0"))
    args = parser.parse_args()
    if args.runs < 2 or args.timeout <= 0:
        parser.error("at least two runs and a positive timeout required")
    binary = args.db_cli.resolve()
    acpp = shutil.which(args.acpp)
    if not acpp or not binary.is_file() or not args.acpp_include.is_dir():
        parser.error("missing compiler, db_cli or AdaptiveCpp include directory")
    if args.trace_compiler and not shutil.which("strace"):
        parser.error("--trace-compiler requires strace")
    root = Path(__file__).resolve().parents[2]
    # Legacy engine has a global PCH filename. Refuse to overwrite an existing artifact.
    legacy_pch = Path("/dev/shm/jit_headers.h.gch")
    if legacy_pch.exists():
        parser.error(f"refusing to overwrite existing legacy PCH: {legacy_pch}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="sql-baseline-", dir=args.output_root.resolve()))
    print(f"Artifacts: {output}", flush=True)
    report = {"success": False, "db_cli": str(binary), "timing_model": args.timing_model,
               "warm_codegen_ms_expected_zero": args.timing_model == "legacy-generated-v1",
              "db_cli_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ("src", "include") for p in sorted((root / folder).rglob("*"))
                                if p.is_file()},
              "compile_link_separation": "Clang/lld traces plus native acpp wall; optional process tracing",
              "device_jit_cache": "fresh isolated directory at session start; retained within session",
              "os_caches": "not flushed", "records": []}
    owns_pch = False
    try:
        sys.dont_write_bytecode = True
        fd = os.open(legacy_pch, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owns_pch = True
        os.close(fd)
        report["acpp_version"] = subprocess.check_output([acpp, "--acpp-version"], text=True, timeout=15)
        build_cache = binary.parent / "CMakeCache.txt"
        if build_cache.exists():
            shutil.copyfile(build_cache, output / "engine-CMakeCache.txt")
            report["engine_cmake_cache_sha256"] = hashlib.sha256(build_cache.read_bytes()).hexdigest()
        catalog = Path("/dev/shm/catalog_cache.txt")
        report["catalog_cache_initial_sha256"] = hashlib.sha256(catalog.read_bytes()).hexdigest() if catalog.exists() else None
        # Reuse the canonical timing normalization rather than silently renaming SQL fields.
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("olap_timing", root / "olap_bench/tools/run_olap_bench.py")
        timing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timing)
        events = output / "compiler-events"
        events.mkdir()
        proxy = output / "toolchain"
        proxy.mkdir()
        wrapper = proxy / "acpp"
        shutil.copyfile(Path(__file__).with_name("compiler_proxy.sh"), wrapper)
        wrapper.chmod(0o700)
        config = output / "crystal.conf"
        config.write_text("dump_generated_code=off\nextended_timing=on\noutput_limit=off\n"
                          "memory_guard_enabled=on\nreuse_scratch_buffers=true\n")
        env = os.environ.copy()
        env.update({"PATH": str(proxy) + os.pathsep + env.get("PATH", ""),
                    "CRYSTAL_CONFIG": str(config), "CRYSTAL_INCLUDE_DIR": str(root / "include"),
                    "CRYSTAL_DEPS_INCLUDE_DIR": str(root / "deps/include"),
                    "ACPP_INCLUDE_DIR": str(args.acpp_include.resolve()),
                    "ACPP_APPDB_DIR": str(output / "device-cache")})
        if args.trace_compiler:
            env["CRYSTAL_STAGE0_TRACE_COMPILER"] = "1"
            report["compile_link_separation"] = "observed Clang/linker process intervals; tracing overhead included"
        if args.jit_optimization:
            env["CRYSTAL_STAGE0_OPTIMIZATION"] = args.jit_optimization
        if args.jit_pch:
            env["CRYSTAL_STAGE0_USE_PCH"] = args.jit_pch
        env["CRYSTAL_STAGE0_REAL_ACPP"] = acpp
        env["CRYSTAL_STAGE0_EVENT_DIR"] = str(events)
        report["native_time"] = "Bash time, Clang ftime-trace, lld time-trace; Python analysis after execution"
        if args.linker_opt is not None:
            env["CRYSTAL_STAGE0_LINK_OPT"] = args.linker_opt
        report["controlled_compiler_overrides"] = {"optimization": args.jit_optimization, "pch": args.jit_pch}
        report["environment"] = {k: v for k, v in env.items()
                                 if k.startswith(("ACPP_", "HIPSYCL_", "HSA_", "HIP_", "ROCR_", "CRYSTAL_"))}
        nonce = uuid.uuid4().hex
        report["artifact_isolation"] = {"nonce": nonce, "method": "128 bits of internal SELECT whitespace"}
        queries = args.query or list(REFERENCE_GROUPS)
        commands = ["\\dump off", "\\timing on", "\\limit all"]
        expected = []
        for query in queries:
            sql = (root / f"test/queries/sql/{query}.sql").read_text().strip()
            # The old engine hashes raw SQL for /tmp filenames: isolate artifacts per run.
            sql = isolate_sql(sql, nonce)
            reference = root / f"test/queries/result/scale10/{query[1:]}.txt"
            rows = reference_rows(reference.read_text())
            for run in range(args.runs):
                marker = f"__STAGE0_SQL__{query}__{run}__"
                commands += [f"\\echo {marker}", sql]
                expected.append((marker, query, run, rows))
        commands += ["\\echo __STAGE0_SQL_END__", "\\q"]
        stdin = "\n".join(commands) + "\n"
        (output / "stdin.sql").write_text(stdin)
        from native_timing import timed_command, read_time
        native_errors = output / "db-cli.command.stderr"
        native_time_file = output / "db-cli.time.json"
        timed = timed_command([str(binary)], native_time_file, native_errors)
        returncode, stdout, timer_stderr, timed_out = run_session(timed, stdin, root, env, args.timeout)
        stderr = native_errors.read_text() + timer_stderr
        if not timed_out:
            report["process_time"] = read_time(native_time_file)
            report["process_wall_ms"] = report["process_time"]["wall_ms"]
        report["returncode"] = returncode
        report["timed_out"] = timed_out
        (output / "stdout.txt").write_text(stdout)
        (output / "stderr.txt").write_text(stderr)
        for index, (marker, query, run, rows) in enumerate(expected):
            next_marker = expected[index + 1][0] if index + 1 < len(expected) else "__STAGE0_SQL_END__"
            chunk = stdout.split(marker + "\n", 1)[-1].split(next_marker + "\n", 1)[0]
            record = {"query": query, "run": run, "mode": "cold_sql" if not run else "warm_sql"}
            record.update(timing.normalize_timing_record(timing.parse_metrics(chunk)))
            try:
                if stdout.count(marker + "\n") != 1 or stdout.count(next_marker + "\n") != 1:
                    raise ValueError("missing or duplicate session marker")
                actual = cli_rows(chunk, REFERENCE_GROUPS[query])
                record["result_verified"] = actual == rows
                if actual != rows:
                    record["missing_rows"] = list((rows - actual).elements())[:5]
                    record["extra_rows"] = list((actual - rows).elements())[:5]
            except ValueError as error:
                record["result_verified"] = False
                record["verification_error"] = str(error)
            report["records"].append(record)
        from native_timing import collect_events
        report["compiler_events"] = collect_events(events)
        report["compiler_call_count"] = len(report["compiler_events"])
        report["pch_prepare_wall_ms"] = sum(e["wall_ms"] for e in report["compiler_events"]
                                            if e["phase"] == "pch_prepare")
        report["compiler_cache_verified"] = compiler_cache_verified(report["compiler_events"], report["records"], len(queries), args.timing_model)
        report["success"] = not timed_out and returncode == 0 and all(r["result_verified"] and
            r["total_query_ms"] is not None for r in report["records"]) and not timing.has_sql_error(stderr) and report["compiler_cache_verified"]
        print(f"Verified {sum(r['result_verified'] for r in report['records'])}/{len(expected)} results; "
              f"compiler calls: {report['compiler_call_count']}", flush=True)
    finally:
        # This file was exclusively created by this runner; retain it with the report.
        if owns_pch and legacy_pch.exists():
            shutil.move(str(legacy_pch), output / "legacy-jit_headers.pch")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
