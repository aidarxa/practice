#!/usr/bin/env python3
"""Build/run an explicit host recipe, archive raw LLVM profiles and export coverage."""
import argparse
from pathlib import Path

from common import Session, ToolError, digest, entry, expand, load_recipe, positive, run_phase, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--flag", action="append", default=[], help="Additional compiler flags via {flags}")
    parser.add_argument("--llvm-cov", default="llvm-cov")
    parser.add_argument("--llvm-profdata", default="llvm-profdata")
    parser.add_argument("--build-timeout", type=positive, default=120)
    parser.add_argument("--test-timeout", type=positive, default=30)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    recipe, inputs = load_recipe(args.recipe, project)
    if not recipe.get("objects") or not any("{flags}" in command for command in recipe["build"]):
        raise ToolError("Coverage requires objects and a build argv element {flags} for instrumentation")
    session = Session(args.output)
    write_json(session.output / "recipe.json", recipe)
    write_json(session.output / "inputs.json", {str(project / name): sha for name, sha in inputs.items()})
    for label, tool in (("cxx", args.cxx), ("llvm-cov", args.llvm_cov), ("llvm-profdata", args.llvm_profdata)):
        session.checked([tool, "--version"], project, label + "-version")
    out = session.output / "build"
    out.mkdir()
    profiles = session.output / "profiles"
    profiles.mkdir()
    flags = [*args.flag, "-fprofile-instr-generate", "-fcoverage-mapping"]
    result = run_phase(session, recipe, "build", project, out, args.cxx, args.build_timeout, flags=flags)
    if result["status"] != "passed":
        raise ToolError("Coverage build failed; see build command logs")
    result = run_phase(session, recipe, "test", project, out, args.cxx, args.test_timeout,
                       env={"LLVM_PROFILE_FILE": str(profiles / "%p-%m.profraw")})
    if result["status"] != "passed":
        raise ToolError("Coverage tests must pass before exporting profiles")
    raw = sorted(profiles.glob("*.profraw"))
    if not raw:
        raise ToolError("No profiles produced; verify instrumentation and the test recipe")
    merged = session.output / "merged.profdata"
    session.checked([args.llvm_profdata, "merge", "-sparse", *map(str, raw), "-o", str(merged)], project,
                    "merge", strict_stderr=True)
    objects = expand(recipe["objects"], project, out, args.cxx)
    options = [objects[0], "-instr-profile=" + str(merged), "-compilation-dir=" + str(project)]
    for obj in objects[1:]:
        options += ["-object", obj]
    exported = session.checked([args.llvm_cov, "export", *options], project, "export", strict_stderr=True)
    session.checked([args.llvm_cov, "report", *options], project, "coverage-summary", strict_stderr=True)
    if any(digest(project / name) != sha for name, sha in inputs.items()):
        raise ToolError("Source inputs changed during coverage collection")
    # Publish only after every check passes; keep unsuccessful raw exports in export.stdout.
    coverage = session.output / "coverage.json"
    coverage.write_bytes(exported.read_bytes())
    return session.finish({"tool": "coverage", "success": True, "project": str(project),
                           "coverage_sha256": digest(coverage),
                           "source_sha256": inputs, "binary_sha256": {obj: digest(obj) for obj in objects},
                           "profiles": [str(path) for path in raw],
                           "scope": "instrumented host code only; no GPU device coverage claim"})


if __name__ == "__main__":
    raise SystemExit(entry(main))
