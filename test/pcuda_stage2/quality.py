#!/usr/bin/env python3
## @file
## @brief Reproducible Clang AST/LLVM-coverage analogue for new stage-2 host components.
"""Uses actual executed regions, not estimated coverage. SDK/device implementations are outside scope."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage1_quality", ROOT / "test/pcuda_stage1/quality.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--gpu-binary", type=Path, required=True)
    parser.add_argument("--gpu-profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--acpp-include", type=Path, required=True)
    parser.add_argument("--flag", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    commands = []
    profiles = [str(path.resolve()) for path in args.gpu_profile]
    binaries = []
    for target in ("test_execution_plan", "test_module_abi", "test_optimizer"):
        binary = (args.build / target).resolve()
        binaries.append(str(binary))
        profile = output / (target + ".profraw")
        env = dict(os.environ, LLVM_PROFILE_FILE=str(profile))
        commands.append([str(binary)])
        with (output / (target + ".log")).open("w") as stdout:
            subprocess.run([binary], env=env, stdout=stdout, stderr=subprocess.STDOUT, check=True)
        profiles.append(str(profile))
    binaries.append(str(args.gpu_binary.resolve()))
    merge = [str(args.llvm_bin / "llvm-profdata"), "merge", "-sparse", *profiles, "-o", str(output / "merged.profdata")]
    commands.append(merge)
    subprocess.run(merge, check=True)
    command = [str(args.llvm_bin / "llvm-cov"), "export", binaries[0], "-instr-profile=" + str(output / "merged.profdata")]
    for binary in binaries[1:]:
        command += ["-object", binary]
    commands.append(command)
    with (output / "coverage.json").open("w") as stdout:
        subprocess.run(command, stdout=stdout, check=True)
    with (output / "coverage.txt").open("w") as stdout:
        subprocess.run([arg if arg != "export" else "report" for arg in command], stdout=stdout, check=True)
    coverage = json.loads((output / "coverage.json").read_text())
    reports = []
    files = ["src/core/" + name for name in ("parameter_value.cpp", "size_expression.cpp", "execution_plan.cpp", "physical_planner.cpp", "legacy_kernel_codegen.cpp", "legacy_plan_runtime.cpp")]
    files.append("include/core/module_contract.h")
    for filename in files:
        source = ROOT / filename
        ast = output / (source.stem + ".ast.json")
        command = [str(args.llvm_bin / "clang++"), "-std=c++17", *args.flag, "-I" + str(ROOT / "include"),
                   "-I" + str(ROOT / "deps/include"), "-isystem" + str(args.acpp_include.resolve()),
                   "-x", "c++", str(source), "-fsyntax-only", "-Xclang", "-ast-dump=json", "-Xclang", "-ast-dump-filter=db"]
        commands.append(command)
        with ast.open("w") as stdout, (output / (source.stem + ".stderr")).open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
        report = previous.analyze(source, ast, coverage)
        reports.append(report)
        (output / (source.stem + ".quality.json")).write_text(json.dumps(report, indent=2) + "\n")
        failures = [(row["name"], row["complexity"], round(row["region_coverage"], 3), round(row["crap_region_analogue"], 3))
                    for row in report["functions"] if row["crap_region_analogue"] >= 10]
        print(source.name, "functions:", len(report["functions"]), "CRAP>=10:", failures,
              "duplicates:", report["duplicates_min_50_tokens_rank_08"])
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    return 0 if all(row["all_new_functions_below_crap_10"] and not row["duplicates_min_50_tokens_rank_08"] for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
