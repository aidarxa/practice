#!/usr/bin/env python3
"""CPU regression tests for the independent oracle and reporting helpers."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from run import stats, build_script, VARIANTS
from tiered_run import verify
from analyze import trace

class Tools(unittest.TestCase):
    def test_stats_preserve_samples(self):
        self.assertEqual(stats([9, 1, 4]), {"n": 3, "raw": [9, 1, 4], "median": 4, "min": 1, "max": 9})
        self.assertIsNone(stats([]))

    def test_intermediate_ir_and_tool_levels_are_independent(self):
        self.assertEqual(VARIANTS["hiprtc03"], (0, 3, 0))
        self.assertEqual(VARIANTS["hiprtc30"], (3, 0, 0))
        self.assertEqual(VARIANTS["clang00"], (0, 0, 1))
        self.assertIsNone(VARIANTS["stock"])

    def test_native_build_and_no_python_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = build_script(directory, ["acpp", "--acpp-targets=generic", "--acpp-pcuda"], directory / "prebuilt.pch", True)
            text = path.read_text()
            self.assertIn('time "$@"', text)
            self.assertIn("unset LD_PRELOAD", text)
            self.assertNotIn("python", text)
            commands = json.loads((directory / "commands.json").read_text())
            self.assertEqual(len(commands), 8)
            self.assertEqual(sum(c["phase"] == "compile" for c in commands), 4)
            self.assertTrue(all("-O0" in c["argv"] for c in commands))
            self.assertTrue(all("-ftime-trace=" in " ".join(c["argv"]) for c in commands if c["phase"] == "compile"))
            self.assertTrue(all("--time-trace=" in " ".join(c["argv"]) for c in commands if c["phase"] == "link"))

    def test_oracle_detects_big_integer_comparison(self):
        # row 504 has v=2^53+1; threshold v+1 must reject it. A double
        # conversion would collapse both nearby values for some comparisons.
        rows = 1011
        threshold = 9007199254740994
        expected = []
        for i in range(rows):
            value = 9007199254740993 + i % 1009 - 504
            expected.append((value % 997) * 7 - 9 if i % 11 and value >= threshold else -(1 << 63))
        execution = {"rows": rows, "runs": [{"threshold": threshold, "offset": -9}]}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "results.bin").write_bytes(struct.pack(f"<{rows}q", *expected))
            self.assertTrue(verify(directory, execution)["success"])
            expected[504] = 1
            (directory / "results.bin").write_bytes(struct.pack(f"<{rows}q", *expected))
            self.assertFalse(verify(directory, execution)["success"])

    def test_oracle_rejects_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "results.bin").write_bytes(b"")
            self.assertFalse(verify(directory, {"rows": 1, "runs": [{"threshold": 0, "offset": 0}]})["success"])

    def test_nested_traces_are_not_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"traceEvents": [
                {"name": "ExecuteCompiler", "dur": 100000},
                {"name": "Total Backend", "dur": 40000},
                {"name": "Total Optimizer", "dur": 25000},
                {"name": "NestedPass", "dur": 20000}]}))
            self.assertEqual(trace(path), {"ExecuteCompiler": 100, "Total Backend": 40, "Total Optimizer": 25})

if __name__ == "__main__": unittest.main()
