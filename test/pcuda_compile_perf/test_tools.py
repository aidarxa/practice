#!/usr/bin/env python3
"""CPU tests; no assert-based checks that disappear under NDEBUG."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from run import check_query_timing, native, stats, trace_summary
from sql_controls import variant_of


class MeasurementTests(unittest.TestCase):
    def test_query_preparation_is_not_excluded_from_v2_summary(self):
        execution = {"method_version": 2, "preparation_in_summary": True,
                     "preparation_ms": 30, "build_elapsed_ms": 100, "load_ms": 1,
                     "runs": [{"total_ms": 50}], "summary_ms": 181.25}
        self.assertAlmostEqual(check_query_timing(execution), 0.25)
        with self.assertRaises(ValueError):
            check_query_timing({**execution, "summary_ms": 151})
        with self.assertRaises(ValueError):
            check_query_timing({**execution, "method_version": 1, "preparation_in_summary": False})

    def test_sql_variant_grouping_does_not_mix_query_orders(self):
        self.assertEqual(variant_of("s0-q33-O0"), "q33-O0")
        self.assertNotEqual(variant_of("s0-q21-q33-O0"), "q33-O0")

    def test_raw_distribution_is_preserved(self):
        values = [7, 1, 3]
        self.assertEqual(stats(values), {"n": 3, "raw": values, "median": 3, "min": 1, "max": 7})
        self.assertIsNone(stats([]))

    def test_nested_trace_intervals_are_not_added(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="perf-tools-") as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": [
                {"name": "ExecuteCompiler", "dur": 10000}, {"name": "Total Frontend", "dur": 6000},
                {"name": "Total Backend", "dur": 3000}, {"name": "Total Optimizer", "dur": 2000}]}))
            self.assertEqual(trace_summary(path), {"ExecuteCompiler": 10, "Total Frontend": 6,
                                                  "Total Backend": 3, "Total Optimizer": 2})

    def test_native_locale_errors_and_failure(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="perf-tools-") as directory:
            prefix = Path(directory) / "command"
            code = native(["bash", "-c", "echo output; echo error >&2; exit 9"], prefix,
                          {**os.environ, "LC_ALL": "ru_RU.UTF-8"})
            self.assertEqual(code, 9)
            self.assertEqual(prefix.with_suffix(".stdout").read_text().strip(), "output")
            self.assertEqual(prefix.with_suffix(".stderr").read_text().strip(), "error")
            timing = json.loads(prefix.with_suffix(".time.json").read_text())
            self.assertEqual(set(timing), {"wall_s", "user_s", "system_s"})
            self.assertTrue(all(v >= 0 for v in timing.values()))

    @unittest.skipUnless(os.environ.get("PERF_LAUNCHER"), "set PERF_LAUNCHER to the compiled native launcher")
    def test_first_result_marker_and_missing_marker(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode", prefix="perf-launcher-") as directory:
            output = Path(directory) / "result.json"
            result = subprocess.run([os.environ["PERF_LAUNCHER"], str(output), "/bin/bash", "-c",
                                     'printf R >&"$PERF_READY_FD"'], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreaterEqual(json.loads(output.read_text())["process_start_to_first_result_ms"], 0)
            result = subprocess.run([os.environ["PERF_LAUNCHER"], str(output), "/bin/true"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
