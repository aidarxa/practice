#!/usr/bin/env python3
## @file
## @brief Offline AST complexity, LLVM region-coverage CRAP analogue and token duplication.
"""@details No external quality packages are required. The CRAP formula uses
LLVM code-region coverage, not crapper's statement coverage. Duplicates compare
complete function bodies with at least 50 lexical tokens; this is not dryer.
"""
import argparse
import difflib
import json
from pathlib import Path
import re


def json_stream(text):
    """@brief Read Clang's concatenated filtered AST objects."""
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        value, position = decoder.raw_decode(text, position)
        yield value


def complexity(node):
    """@brief Cyclomatic complexity includes short-circuit operators and case labels."""
    if node.get("kind") == "LambdaExpr":
        return 0  # lambda call operators are analyzed separately
    decisions = {"IfStmt", "WhileStmt", "DoStmt", "ForStmt", "CXXForRangeStmt", "CaseStmt",
                 "ConditionalOperator", "BinaryConditionalOperator", "CXXCatchStmt"}
    own = node.get("kind") in decisions or (node.get("kind") == "BinaryOperator" and node.get("opcode") in ("&&", "||"))
    return int(own) + sum(complexity(child) for child in node.get("inner", []) if child)


def functions(node):
    """@brief Find defined functions, retaining linkage identities for coverage matching."""
    if node.get("kind") in ("FunctionDecl", "CXXMethodDecl", "CXXConstructorDecl", "CXXDestructorDecl"):
        body = next((child for child in node.get("inner", []) if child.get("kind") == "CompoundStmt"), None)
        if body and node.get("mangledName"):
            yield node, body
    for child in node.get("inner", []):
        if child:
            yield from functions(child)


def tokens(text):
    """@brief Ignore whitespace/comments, but keep identifiers, operators and literal tokens."""
    pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|\w+|[^\s]'
    return [token for token in re.findall(pattern, text, flags=re.S) if not token.startswith(("//", "/*"))]


def coverage_identities(function):
    """@brief Match Itanium complete/base constructor aliases without inventing executed regions."""
    identity = function["mangledName"]
    identities = {identity}
    if function["kind"] == "CXXConstructorDecl":
        identities.add(identity.replace("C1E", "C2E").replace("C1I", "C2I"))
    if function["kind"] == "CXXDestructorDecl":
        identities.add(identity.replace("D1E", "D2E"))
    return identities


def analyze(source, ast, coverage, selected=None):
    """@brief Join actual compiled ASTs to executed regions; never infer coverage from source strings."""
    encoded = source.read_bytes()  # Clang offsets count UTF-8 bytes, not Python characters.
    roots = [root for root in json_stream(ast.read_text()) if Path(root.get("loc", {}).get("file", "")) == source]
    measured = coverage["data"][0]["functions"]
    result = []
    bodies = []
    seen = set()
    source_bodies = set()
    for root in roots:
        for function, body in functions(root):
            if selected is not None and function["name"] not in selected:
                continue
            identity = function["mangledName"]
            if identity in seen:
                continue
            seen.add(identity)
            start = body["range"]["begin"]["offset"]
            end = body["range"]["end"]["offset"] + body["range"]["end"]["tokLen"]
            aliases = coverage_identities(function)
            match = next((item for item in measured if item["name"].split(":")[-1] in aliases), None)
            regions = [region for region in match["regions"] if region[7] == 0] if match else []
            fraction = sum(region[4] > 0 for region in regions) / len(regions) if regions else 0
            cc = 1 + complexity(body)
            crap = cc * cc * (1 - fraction) ** 3 + cc
            entry = {"name": function["name"], "mangled": identity, "line": encoded.count(b"\n", 0, start) + 1,
                     "complexity": cc, "region_coverage": fraction, "crap_region_analogue": crap}
            result.append(entry)
            # Template instantiations share one source body: they are not copied implementations.
            if (start, end) not in source_bodies:
                bodies.append((entry, tokens(encoded[start:end].decode())))
                source_bodies.add((start, end))
    duplicates = []
    for index, (left, a) in enumerate(bodies):
        for right, b in bodies[index + 1:]:
            if min(len(a), len(b)) < 50:
                continue
            rank = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            if rank >= 0.8:
                duplicates.append({"left": left["name"], "right": right["name"], "rank": rank})
    return {"source": str(source), "functions": result, "duplicates_min_50_tokens_rank_08": duplicates,
            "all_new_functions_below_crap_10": bool(result) and all(row["crap_region_analogue"] < 10 for row in result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ast", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functions", help="Optional comma-separated scope; omitted means all source definitions")
    args = parser.parse_args()
    selected = set(args.functions.split(",")) if args.functions else None
    report = analyze(args.source, args.ast, json.loads(args.coverage.read_text()), selected)
    report["selected_functions"] = sorted(selected) if selected else None
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Analyzed {len(report['functions'])} functions; CRAP-region gate: "
          f"{report['all_new_functions_below_crap_10']}; duplicate pairs: {len(report['duplicates_min_50_tokens_rank_08'])}")
    return 0 if report["all_new_functions_below_crap_10"] and not report["duplicates_min_50_tokens_rank_08"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
