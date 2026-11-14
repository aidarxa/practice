#!/usr/bin/env python3
## @file
## @brief Compile isolated frontend mutants and run real behavioural executables.
"""@details This targeted mutation analogue never edits the working tree. Compile
failures are not killed mutants. It is not a replacement for a whole-project
mutation campaign, and no test result is inferred from source-text matching.
"""
import argparse
import hashlib
import json
import shlex
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MUTATIONS = [
    ("slot-collapse", "src/core/query_template.cpp", "std::make_unique<ParameterRefExpr>(slot)", "std::make_unique<ParameterRefExpr>(0)"),
    ("wrong-binding", "src/core/query_template.cpp", "values.at(parameter->slot)", "values.at(0)"),
    ("integer-rounded", "src/core/parameter_value.cpp", "ParameterValue(std::optional<int64_t>{value})", "ParameterValue(std::optional<double>{static_cast<double>(value)})"),
    ("slot-omitted-from-key", "src/core/query_template.cpp", "number(parameter->slot);", "number(0);"),
    ("float-context-lost", "src/core/query_template.cpp", "return ParameterType::Float64;\n    if (left", "return ParameterType::Int64;\n    if (left"),
    ("typed-null-lost", "src/core/query_template.cpp", "ParameterValue::null(schema[i])", "ParameterValue::null()"),
    ("sort-direction-lost", "src/core/query_template.cpp", "number(key.descending ? 1 : 0);", "number(0);"),
    ("limit-check-lost", "src/core/query_template.cpp", "if (result->hasLimit()) (void)result->literalLimit();", "(void)result->hasLimit();"),
    ("having-not-extracted", "src/core/query_template.cpp", "extractExpr(node.having_predicate, ParameterType::Int64, results);", "(void)results;"),
    ("parent-link-lost", "src/core/optimizer_rules.cpp", "replacement->parent_ = filter->parent_;", "replacement->parent_ = nullptr;"),
    ("ambiguity-accepted", "src/core/translator.cpp", 'if (resolved) throw std::runtime_error("Ambiguous column reference: " + column);', 'if (resolved) continue;'),
    ("ordinal-is-physical-slot", "src/core/translator.cpp", "aggregateOutputSlotForSelectIndex(stmt, checkedOrderOrdinal(order_expr, select_count))", "checkedOrderOrdinal(order_expr, select_count)"),
]


def run(command, prefix):
    """@brief Preserve full command/stdout/stderr for correctness-only subprocesses."""
    prefix.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
    with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
        return subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=120).returncode


def active_objects(build, targets):
    """@brief Use the build manifest, not stale object files left after source extraction."""
    objects = []
    for entry in json.loads((build / "compile_commands.json").read_text()):
        argv = entry.get("arguments") or shlex.split(entry["command"])
        if "-o" not in argv:
            continue
        obj = Path(argv[argv.index("-o") + 1])
        if not obj.is_absolute():
            obj = Path(entry["directory"]) / obj
        if any(target + ".dir" in obj.parts for target in targets):
            objects.append(obj)
    return sorted(objects)


def check_mutant(mutation, args, output):
    """@brief One mutation, one changed translation unit, three real test clients."""
    name, filename, before, after = mutation
    directory = output / name
    directory.mkdir()
    original = (ROOT / filename).read_text()
    if original.count(before) != 1:
        raise ValueError(f"Mutation {name} no longer has one precise target")
    source = directory / Path(filename).name
    source.write_text(original.replace(before, after, 1))
    obj = directory / "mutant.o"
    flags = [args.cxx, "-std=c++17", "-O0", "-UNDEBUG", "-I" + str(ROOT / "include"), "-I" + str(ROOT / "deps/include"), *args.flag]
    record = {"name": name, "source": filename, "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
              "mutation": [before, after], "status": "build-failed", "tests": []}
    if run(flags + ["-c", str(source), "-o", str(obj)], directory / "compile") != 0:
        return record
    for target in ("test_query_template", "test_translator", "test_optimizer"):
        objects = active_objects(args.build, (target, "query_values", "physical_execution"))
        objects = [obj if str(path).endswith(filename + ".o") else path for path in objects]
        binary = directory / target
        link = flags + [str(path) for path in objects] + ["-L" + str(ROOT / "deps/lib"), "-lsqlparser",
            "-Wl,-rpath," + str(ROOT / "deps/lib"), "-L" + str(args.acpp_lib), "-lacpp-rt", "-lacpp-common",
            "-Wl,-rpath," + str(args.acpp_lib), "-o", str(binary)]
        if run(link, directory / (target + "-link")) != 0:
            return record
        rc = run([str(binary)], directory / (target + "-run"))
        record["tests"].append({"target": target, "returncode": rc})
        if rc != 0:
            record["status"] = "killed"
            return record
    record["status"] = "survived"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--acpp-lib", type=Path, required=True)
    parser.add_argument("--flag", action="append", default=[])
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.acpp_lib = args.acpp_lib.resolve()
    args.output.mkdir(parents=True)
    args.output = args.output.resolve()
    for target in ("test_query_template", "test_translator", "test_optimizer"):
        if run([str(args.build / target)], args.output / ("baseline-" + target)) != 0:
            raise RuntimeError("Baseline tests must pass before mutation")
    records = [check_mutant(mutation, args, args.output) for mutation in MUTATIONS]
    report = {"purpose": "targeted correctness mutations; no performance timing", "records": records,
              "success": all(record["status"] == "killed" for record in records)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({status: sum(record["status"] == status for record in records) for status in ("killed", "survived", "build-failed")})
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
