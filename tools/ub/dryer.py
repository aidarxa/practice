#!/usr/bin/env python3
"""Cross-file whole-function C++ token similarity; findings require a semantic review."""
import difflib
import math

from common import Session, ToolError, entry
from cpp import analysis_parser, collect, tokens


def fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ToolError("Rank must be in (0, 1]")
    return value


def compare(functions, minimum, threshold):
    bodies = {}
    for function in functions:
        key = (str(function.source), function.start, function.end)
        bodies.setdefault(key, (function.public(), tokens(function.text)))
    bodies = [(row, body) for row, body in bodies.values() if len(body) >= minimum]
    duplicates = []
    for index, (left, a) in enumerate(bodies):
        for right, b in bodies[index + 1:]:
            # Nested definitions share written bytes with their parent, rather
            # than being independent copies. Both still compare with others.
            if left["source"] == right["source"] and (
                left["start"] <= right["start"] and right["end"] <= left["end"] or
                right["start"] <= left["start"] and left["end"] <= right["end"]
            ):
                continue
            if 2 * min(len(a), len(b)) / (len(a) + len(b)) < threshold:
                continue
            # SequenceMatcher is asymmetric; compare both orders for a stable, conservative rank.
            rank = max(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(),
                       difflib.SequenceMatcher(None, b, a, autojunk=False).ratio())
            if rank >= threshold:
                duplicates.append({"left": left, "right": right, "rank": rank, "tokens": [len(a), len(b)]})
    return duplicates


def main():
    parser = analysis_parser(__doc__)
    parser.add_argument("--rank", type=fraction, default=0.8)
    parser.add_argument("--min-tokens", type=int, default=50)
    args = parser.parse_args()
    if args.min_tokens < 1:
        raise ToolError("--min-tokens must be positive")
    session = Session(args.output)
    functions = collect(args, session)
    duplicates = compare(functions, args.min_tokens, args.rank)
    return session.finish({"tool": "dryer", "scope": "whole function bodies, exact identifiers/literals; no partial clones",
                           "min_tokens": args.min_tokens, "rank_inclusive": args.rank,
                           "functions": [fn.public() for fn in functions], "duplicates": duplicates, "success": not duplicates})


if __name__ == "__main__":
    raise SystemExit(entry(main))
