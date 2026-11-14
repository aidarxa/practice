#!/usr/bin/env python3
## @file
## @brief Reuse the correctness CLI runner with independent local-Top-K progress boundaries.
"""Use --skip-manifest --output-limit 5000 for these 4096-row oracle checks."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage1_cli", ROOT / "test/pcuda_stage1/validate_cli.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def boundary_queries():
    """@brief K above half of a 4096-row chunk must terminate and keep correct ordering."""
    queries = []
    for limit in (2049, 4096, 4096):
        queries.append((f"topk-progress-{limit}-{len(queries)}",
            "SELECT s_nation,s_region FROM supplier WHERE s_suppkey>0 "
            f"ORDER BY s_nation,s_region LIMIT {limit}", "stage2-topk", limit))
    return queries


def expected_rows(oracle, argument, data):
    """@brief Sort actual int32 source columns on CPU, independently of GPU helpers."""
    if oracle != "stage2-topk":
        return previous.expected_rows(oracle, argument, data)
    import numpy as np
    keys = np.memmap(data / "SUPPLIER0", dtype="<i4", mode="r")
    nation = np.memmap(data / "SUPPLIER4", dtype="<i4", mode="r")
    region = np.memmap(data / "SUPPLIER5", dtype="<i4", mode="r")
    selected = np.flatnonzero(keys > 0)
    order = selected[np.lexsort((region[selected], nation[selected]))][:argument]
    return [(int(nation[index]), int(region[index])) for index in order]


if __name__ == "__main__":
    raise SystemExit(previous.main(boundary_queries(), expected_rows))
