#!/usr/bin/env python3
"""C++17 complexity + actual LLVM code-region coverage; CRAP-region, not statement CRAP."""
import json
from pathlib import Path
import re

from common import Session, ToolError, digest, entry, positive
from cpp import analysis_parser, collect, complexity


def score(complexity_value, coverage):
    return complexity_value ** 2 * (1 - coverage) ** 3 + complexity_value


def aliases(node):
    identity = node.get("mangledName")
    result = {identity} if identity else set()
    kind = node["kind"]
    if kind == "CXXConstructorDecl":
        result.add(re.sub(r"C1(?=[EI])", "C2", identity or ""))
    elif kind == "CXXDestructorDecl":
        result.add(re.sub(r"D1(?=[EI])", "D2", identity or ""))
    return result - {""}


def coverage_rows(document):
    if document.get("type") != "llvm.coverage.json.export" or not document.get("data"):
        raise ToolError("Expected a complete llvm-cov export, including functions and regions")
    result = []
    for group in document["data"]:
        if "functions" not in group:
            raise ToolError("llvm-cov --summary-only/--skip-functions cannot be used")
        result.extend(group["functions"])
    return result


def measured(function, rows):
    matches = []
    for row in rows:
        if row["name"].rsplit(":", 1)[-1] not in aliases(function.node):
            continue
        regions = []
        for region in row["regions"]:
            # CodeRegion=0. Expansion, skipped, gap and branch records are not code regions.
            if len(region) < 8 or region[7] != 0:
                continue
            filenames = row["filenames"]
            if not 0 <= region[5] < len(filenames):
                raise ToolError("Invalid LLVM coverage file index")
            filename = Path(filenames[region[5]])
            if not filename.is_absolute():
                raise ToolError("Relative LLVM filenames need an absolute --compilation-dir at export")
            if filename.resolve() == function.source:
                regions.append(region)
        if regions:
            matches.append(regions)
    if not matches:
        return {"coverage_status": "missing", "region_coverage": None, "code_regions": 0}
    # Several binary exports can describe the same linkage identity. Do not choose the best one.
    layouts = [{tuple(region[:4]) for region in regions} for regions in matches]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ToolError(f"Incompatible region layouts for {function.node.get('name')}")
    counts = {}
    for regions in matches:
        for region in regions:
            key = tuple(region[:4])
            counts[key] = counts.get(key, 0) + region[4]
    fraction = sum(count > 0 for count in counts.values()) / len(counts)
    return {"coverage_status": "measured", "region_coverage": fraction, "code_regions": len(counts)}


def main():
    parser = analysis_parser(__doc__)
    parser.add_argument("--coverage", type=Path, required=True, help="Raw llvm-cov export JSON")
    parser.add_argument("--threshold", type=positive, default=10, help="Strictly below, default <10")
    args = parser.parse_args()
    rows = coverage_rows(json.loads(args.coverage.read_text(encoding="utf-8")))
    session = Session(args.output)
    functions = collect(args, session)
    provenance = "caller-supplied LLVM export; source freshness unverified"
    metadata = args.coverage.parent / "inputs.json"
    if metadata.is_file():
        completed = json.loads((args.coverage.parent / "report.json").read_text(encoding="utf-8"))
        if completed.get("tool") != "coverage" or completed.get("success") is not True:
            raise ToolError("Coverage collection did not complete successfully")
        hashes = json.loads(metadata.read_text(encoding="utf-8"))
        for source in {fn.source for fn in functions}:
            if hashes.get(str(source)) != digest(source):
                raise ToolError(f"Coverage source hash missing or stale: {source}")
        if completed.get("coverage_sha256") != digest(args.coverage):
            raise ToolError("Coverage export no longer matches its completed collection")
        provenance = "coverage.py source and export hashes verified"
    result = []
    for function in functions:
        measured_nodes = [function.body] + [child for child in function.node.get("inner", [])
                                           if child.get("kind") == "CXXCtorInitializer"]
        row = {**function.public(), "complexity": 1 + sum(complexity(node) for node in measured_nodes), **measured(function, rows)}
        row["crap_region"] = score(row["complexity"], row["region_coverage"]) if row["region_coverage"] is not None else None
        row["passed"] = row["crap_region"] is not None and row["crap_region"] < args.threshold
        result.append(row)
    report = {"tool": "crapper", "metric": "CC^2*(1-LLVM_code_region_coverage)^3+CC",
              "coverage_provenance": provenance,
              "threshold_exclusive": args.threshold, "functions": result, "success": all(row["passed"] for row in result)}
    return session.finish(report)


if __name__ == "__main__":
    raise SystemExit(entry(main))
