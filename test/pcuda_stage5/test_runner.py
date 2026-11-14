"""CPU-only behavioral checks for timing/dump parsing and the parameter-reuse workload."""
import json
from pathlib import Path
import tempfile
import unittest

import validate_cli as validation


class RunnerChecks(unittest.TestCase):
    def test_parameter_workload_has_distinct_phase(self):
        items = validation.workload("performance", 9, True)
        self.assertEqual(len(items), 17)
        self.assertEqual([row[0] for row in items[9:]], [f"q33-parameters-{i}" for i in range(8)])
        self.assertIn("c_city = 235 or c_city = 235", items[9][1])
        self.assertIn("d_year >=1993", items[9][1])
        self.assertIn("d_year <= 1996", items[9][1])
        self.assertIn("c_city = 231 or c_city = 235", items[0][1])

    def test_every_ready_parameter_record_requires_zero_events(self):
        zeros = {"module_" + key + "_calls": 0 for key in ("plan_codegen", "module_codegen", "compiler", "linker", "loads", "symbols")}
        first = dict(zeros, id="q33-0", module_compiler_calls=4)
        repeat = dict(zeros, id="q33-1")
        changed = dict(zeros, id="q33-parameters-0")
        self.assertTrue(validation.reuse_verified([first, repeat, changed]))
        self.assertFalse(validation.reuse_verified([first, repeat, dict(changed, module_plan_codegen_calls=1)]))
        del changed["module_symbols_calls"]
        self.assertFalse(validation.reuse_verified([first, changed]))

    def test_metrics_and_separate_dump(self):
        runner = validation.module_at("module_runner_test", "olap_bench/tools/run_olap_bench.py")
        text = ("Timing model: pcuda-modules-v1\nNative compile wall sum: 12.5 ms\n"
                "Native link wall sum: 9 ms\nModule build elapsed: 7.5 ms\n"
                "Module events: plan_codegen=1 module_codegen=4 compiler=4 linker=4 hits=2 misses=4 loads=4 symbols=12\n")
        metrics = runner.parse_metrics(text)
        self.assertEqual(metrics["compile_wall_sum_ms"], 12.5)
        self.assertEqual(metrics["link_wall_sum_ms"], 9)
        self.assertEqual(metrics["build_elapsed_ms"], 7.5)
        self.assertEqual(metrics["module_cache_hits"], 2)
        self.assertEqual(metrics["module_symbols_calls"], 12)
        self.assertIsNone(runner.normalize_timing_record({})["compile_ms"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner.save_generated_sources(directory, "q", "--- Generated Code ---\n// Inspection\n"
                                          "// ===== Separate module 0: scan =====\nfirst\n"
                                          "// ===== Separate module 1: build =====\nsecond\n--- End ---")
            self.assertEqual((directory / "q/module-0.cpp").read_text(), "first\n")
            self.assertEqual((directory / "q/module-1.cpp").read_text(), "second\n")
            self.assertEqual(len(json.loads((directory / "q/modules.json").read_text())["modules"]), 2)

    def test_depfile_has_toolchain_inputs_without_self_dependency(self):
        writer = validation.module_at("manifest_writer_test", "tools/pcuda_pch_manifest.py")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pch = directory / "ready.pch"
            plugin = directory / "a b#d$e.h"
            depfile = directory / "ready.pch.d"
            writer.write_depfile(depfile, pch, {pch, plugin})
            target, dependencies = depfile.read_text().split(": ", 1)
            self.assertEqual(target, str(pch))
            self.assertIn(r"a\ b\#d$$e.h", dependencies)
            self.assertNotIn("ready.pch", dependencies)

    def test_phase_constraints_and_raw_statistics(self):
        for mode, runs in (("manifest", 9), ("ssb", 9), ("performance", 1)):
            with self.assertRaises(ValueError):
                validation.workload(mode, runs, True)
        analyzer = validation.module_at("stage5_analyzer_test", "test/pcuda_stage5/analyze_series.py")
        self.assertIsNone(analyzer.distribution([]))
        self.assertEqual(analyzer.distribution([1, 9, 2]), {"n": 3, "median": 2, "min": 1, "max": 9, "raw": [1, 9, 2]})


if __name__ == "__main__":
    unittest.main()
