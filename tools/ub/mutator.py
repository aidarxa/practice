#!/usr/bin/env python3
"""Discover AST operator mutants and execute explicit recipes in independent source copies."""
import argparse
import json
from pathlib import Path
import re
import stat

from common import (Session, ToolError, digest, entry, load_recipe, positive, relative_path,
                    run_phase, snapshot, write_json)
from cpp import LEXER, analysis_parser, collect, extent, walk

OPERATORS = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==",
             "&&": "||", "||": "&&", "+": "-", "-": "+", "*": "/", "/": "*"}


def discover(args):
    session = Session(args.output)
    functions = collect(args, session)
    project = args.project.resolve(strict=True)
    mutations = {}
    for function in functions:
        relative = str(function.source.relative_to(project))
        data = function.source.read_bytes()
        for node in walk(function.body):
            if node.get("kind") != "BinaryOperator" or node.get("opcode") not in OPERATORS:
                continue
            children = node.get("inner", [])
            if len(children) != 2 or not all(extent(child) for child in children):
                continue
            begin, end = extent(children[0])[1], extent(children[1])[0]
            if not function.start <= begin < end <= function.end:
                continue
            text = data[begin:end].decode("utf-8")
            matches = [match for match in LEXER.finditer(text) if match.lastgroup not in ("space", "comment")]
            if len(matches) != 1 or matches[0].group() != node["opcode"]:
                continue  # Never guess an operator location inside a macro expansion.
            offset = begin + len(text[:matches[0].start()].encode("utf-8"))
            key = (relative, offset)
            mutations[key] = {"file": relative, "offset": offset,
                              "before": node["opcode"], "after": OPERATORS[node["opcode"]],
                              "source_sha256": digest(function.source), "line": data.count(b"\n", 0, offset) + 1}
    if not mutations:
        raise ToolError("No supported non-macro binary operators in the selected scope")
    records = [{"id": f"operator-{index}", **row} for index, (_, row) in enumerate(sorted(mutations.items()))]
    return session.finish({"tool": "mutator-discover", "success": True, "mutations": records,
                           "scope": "builtin relational, logical and arithmetic binary operators; not exhaustive"})


def validate_mutations(records, project, inputs):
    if not isinstance(records, list) or not records:
        raise ToolError("Mutation campaign must not be empty")
    result = []
    names = set()
    for row in records:
        name = row["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name == "baseline" or name in names:
            raise ToolError(f"Invalid, reserved or duplicate mutation id: {name}")
        names.add(name)
        relative = str(relative_path(row["file"]))
        if relative not in inputs:
            raise ToolError(f"Mutation target must be a declared input: {relative}")
        original = (project / relative).read_bytes()
        before, after = row["before"].encode("utf-8"), row["after"].encode("utf-8")
        if not before or before == after:
            raise ToolError(f"Mutation {name} is empty or changes nothing")
        if "source_sha256" in row and row["source_sha256"] != inputs[relative]:
            raise ToolError(f"Stale mutation source hash: {name}")
        offset = row.get("offset")
        if offset is None:
            if original.count(before) != 1:
                raise ToolError(f"Mutation {name} must have exactly one target, or a hashed byte offset")
            offset = original.index(before)
        elif "source_sha256" not in row or type(offset) is not int or offset < 0:
            raise ToolError(f"Mutation {name} needs a nonnegative byte offset and source_sha256")
        if original[offset:offset + len(before)] != before:
            raise ToolError(f"Mutation target no longer matches: {name}")
        result.append({**row, "file": relative, "offset": offset, "source_sha256": inputs[relative]})
    return result


def check_copy(session, recipe, project, inputs, directory, args, mutation=None):
    source, out = directory / "source", directory / "build"
    snapshot(project, inputs, source)
    out.mkdir()
    session.event("snapshot", reason="One source copy and build directory per actor; no working-tree edits", directory=str(source))
    if mutation:
        target = source / mutation["file"]
        data = target.read_bytes()
        offset = mutation["offset"]
        changed = data[:offset] + mutation["after"].encode("utf-8") + data[offset + len(mutation["before"].encode("utf-8")):]
        target.chmod(target.stat().st_mode | stat.S_IWUSR)  # Only our owned copy, never the original.
        target.write_bytes(changed)
        session.event("mutate", reason="Apply one verified patch to an owned copy", mutation=mutation)
    label = directory.relative_to(session.output).as_posix() + "/"
    result = run_phase(session, recipe, "build", source, out, args.cxx, args.build_timeout, args.flag, label=label)
    if result["status"] != "passed":
        return {"status": "build-timeout" if result["status"] == "timeout" else "build-failed", "result": result}
    result = run_phase(session, recipe, "test", source, out, args.cxx, args.test_timeout, label=label)
    status = {"passed": "survived", "failed": "killed", "timeout": "timeout"}[result["status"]]
    return {"status": status, "result": result}


def run(args):
    project = args.project.resolve(strict=True)
    recipe, inputs = load_recipe(args.recipe, project)
    records = recipe.get("mutations")
    if args.mutations:
        document = json.loads(args.mutations.read_text(encoding="utf-8"))
        records = document["mutations"] if isinstance(document, dict) else document
    mutations = validate_mutations(records, project, inputs)
    session = Session(args.output)
    write_json(session.output / "recipe.json", recipe)
    write_json(session.output / "inputs.json", inputs)
    session.checked([args.cxx, "--version"], project, "cxx-version")
    baseline_dir = session.output / "baseline"
    baseline = check_copy(session, recipe, project, inputs, baseline_dir, args)
    if baseline["status"] != "survived":
        raise ToolError("Fresh isolated baseline must build and pass before mutation")
    if any(digest(baseline_dir / "source" / name) != sha for name, sha in inputs.items()):
        raise ToolError("Baseline commands changed source inputs; put generated files under {out}")
    results = []
    # Copy from the verified baseline, not changing shared or concurrently edited project sources.
    for mutation in mutations:
        outcome = check_copy(session, recipe, baseline_dir / "source", inputs, session.output / "mutants" / mutation["id"], args, mutation)
        results.append({**mutation, **outcome})
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("killed", "survived", "build-failed", "build-timeout", "timeout")}
    evaluated = counts["killed"] + counts["survived"]
    return session.finish({"tool": "mutator", "success": counts["killed"] == len(mutations), "counts": counts,
                           "mutation_score": counts["killed"] / evaluated if evaluated else None,
                           "unresolved": len(mutations) - evaluated, "mutations": results,
                           "scope": "only listed first-order mutants; timeouts/build failures are never kills"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover_parser = commands.add_parser("discover", parents=[analysis_parser("Generate operator mutations")], add_help=False)
    discover_parser.add_argument("--project", type=Path, default=Path.cwd())
    run_parser = commands.add_parser("run", help="Fresh baseline, then one isolated build/test per mutant")
    run_parser.add_argument("--project", type=Path, default=Path.cwd())
    run_parser.add_argument("--recipe", type=Path, required=True)
    run_parser.add_argument("--mutations", type=Path, help="Override recipe mutations with a list or discovery report")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--cxx", default="clang++")
    run_parser.add_argument("--flag", action="append", default=[])
    run_parser.add_argument("--build-timeout", type=positive, default=120)
    run_parser.add_argument("--test-timeout", type=positive, default=30)
    args = parser.parse_args()
    return discover(args) if args.command == "discover" else run(args)


if __name__ == "__main__":
    raise SystemExit(entry(main))
