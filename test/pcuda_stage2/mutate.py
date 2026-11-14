#!/usr/bin/env python3
## @file
## @brief Isolated semantic mutants of the host plan and ABI, never of the working tree.
"""Compile failures are not kills. A test deadline is a correctness failure, not a performance measurement."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage1_mutations", ROOT / "test/pcuda_stage1/mutate.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
MUTATIONS = [
    ("size-add-overflow", "size_expression.cpp", "if (b > std::numeric_limits<uint64_t>::max() - a)", "if (false)"),
    ("size-product-overflow", "size_expression.cpp", "if (a && b > std::numeric_limits<uint64_t>::max() / a)", "if (false)"),
    ("size-ceil-truncated", "size_expression.cpp", "a / b + static_cast<uint64_t>(ceil && a % b != 0)", "a / b"),
    ("integer-rounded", "parameter_value.cpp", "ParameterValue(std::optional<int64_t>{value})", "ParameterValue(std::optional<double>{static_cast<double>(value)})"),
    ("typed-null-valid", "execution_plan.cpp", "value.isNull() ? 0u : 1u", "1u"),
    ("branch-swapped", "execution_plan.cpp", "step.condition.evaluate(scalars_) ? step.yes : step.no", "step.condition.evaluate(scalars_) ? step.no : step.yes"),
    ("buffer-not-cleared", "execution_plan.cpp", "if (range.bytes) backend_.fill(range.data, step.value, range.bytes);", "(void)range;"),
    ("prefix-inclusive", "execution_plan.cpp", "offsets[i] = total; total = checkedSizeAdd(total, counts[i]);", "total = checkedSizeAdd(total, counts[i]); offsets[i] = total;"),
    ("copy-offset-lost", "execution_plan.cpp", "step.source_offset.evaluate(scalars_)", "0"),
    ("loop-bound-lost", "execution_plan.cpp", "if (iterations++ == step.maximum_iterations)", "if (false)"),
    ("zero-work-not-skipped", "execution_plan.cpp", "if (!grid) return;", "(void)grid;"),
    ("buffer-capture-lost", "physical_planner.cpp", "if (names.count(buffer.name)) contract.arguments.push_back(bufferBinding(buffer));", "if (false) contract.arguments.push_back(bufferBinding(buffer));"),
    ("abi-version-unchecked", "module_contract.h", "if (module.abi_version != DB_MODULE_ABI_VERSION || call.abi_version != DB_MODULE_ABI_VERSION)", "if (false)"),
    ("backend-unchecked", "module_contract.h", "return backend == DB_BACKEND_PCUDA || backend == DB_BACKEND_LEGACY_SYCL;", "return true;"),
    ("schema-type-unchecked", "module_contract.h", "if (expected.type == actual.type) return true;", "return true;"),
    ("launch-volume-overflow", "module_contract.h", "if (volume > std::numeric_limits<size_t>::max() / dimension)", "if (false)"),
    ("buffer-extent-overflow", "module_contract.h", "if (argument.element_count > std::numeric_limits<size_t>::max() / bytes)", "if (false)"),
]


def run(command, prefix, timeout=90):
    """@brief Archive commands and raw output outside any performance timing chain."""
    prefix.with_suffix(".command.json").write_text(json.dumps([str(x) for x in command], indent=2) + "\n")
    with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
        try:
            return subprocess.run(command, stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return "test-deadline"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--flag", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    args.build = args.build.resolve()
    for target in ("test_execution_plan", "test_module_abi"):
        if run([args.build / target], args.output / ("baseline-" + target)) != 0:
            raise RuntimeError("Baseline must pass before mutation")
    flags = [args.cxx, "-std=c++17", "-O0", "-UNDEBUG", *args.flag]
    records = []
    for name, filename, before, after in MUTATIONS:
        directory = args.output / name
        directory.mkdir()
        header = filename.endswith(".h")
        relative = (Path("include/core") if header else Path("src/core")) / filename
        original = (ROOT / relative).read_text()
        if original.count(before) != 1:
            raise ValueError(f"Mutation {name} must have exactly one target")
        source = directory / relative
        source.parent.mkdir(parents=True)
        source.write_text(original.replace(before, after, 1))
        includes = ["-I" + str(directory / "include"), "-I" + str(ROOT / "include")]
        record = {"name": name, "file": str(relative), "mutation": [before, after], "status": "build-failed"}
        binary = directory / "test"
        if header:
            command = flags + includes + [str(ROOT / "test/test_module_abi.cpp"), "-o", str(binary)]
            result = run(command, directory / "build")
        else:
            obj = directory / "mutant.o"
            result = run(flags + includes + ["-c", str(source), "-o", str(obj)], directory / "compile")
            if result == 0:
                objects = previous.active_objects(args.build, ("test_execution_plan", "physical_execution", "query_values"))
                objects = [obj if str(path).endswith(str(relative) + ".o") else path for path in objects]
                result = run(flags + [str(path) for path in objects] + ["-o", str(binary)], directory / "link")
        if result == 0:
            result = run([binary], directory / "run", timeout=5)
            record.update({"status": "survived" if result == 0 else "killed", "test_result": result})
        records.append(record)
    report = {"purpose": "targeted correctness mutations", "records": records,
              "success": all(row["status"] == "killed" for row in records)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({status: sum(row["status"] == status for row in records) for status in ("killed", "survived", "build-failed")})
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
