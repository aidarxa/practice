"""Stage-8 measurement fields must stay separate from historical component sums."""
import importlib.util
from pathlib import Path
import unittest
import measure

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_timing_stage8", ROOT / "olap_bench/tools/run_olap_bench.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TimingChecks(unittest.TestCase):
    def test_workload_isolated_cold_warm_changed(self):
        items = measure.workload("q33", 9)
        self.assertEqual([i["phase"] for i in items], ["cold"] + ["warm"]*8 + ["changed"]*8)
        self.assertIn("c_city = 235 or c_city = 235", items[9]["sql"])
        self.assertEqual(len(measure.workload("q11", 9)), 9)

    def test_ready_counters_and_direct_intervals_are_mandatory(self):
        record = dict(reported_timing_model="pcuda-modules-v1", sql_to_pcuda_ms=1, total_query_ms=10,
                      generated_execute_ms=8, host_fetch_ms=1, build_elapsed_ms=0)
        record.update({"module_"+name+"_calls":0 for name in measure.EVENTS})
        measure.check_record(record, False)
        for field in ("sql_to_pcuda_ms", "module_symbols_calls"):
            invalid = dict(record); del invalid[field]
            with self.assertRaises(ValueError): measure.check_record(invalid, False)
        with self.assertRaises(ValueError): measure.check_record(record, True)
        with self.assertRaises(ValueError): measure.check_record(dict(record, sql_to_pcuda_ms=11), False)

    def test_direct_frontend_is_not_an_additional_component(self):
        record = runner.parse_metrics("SQL to PCUDA time: 3.25 ms\nPrepare time: 1 ms\nCode generation time: 2 ms\n"
                                      "ACPP compilation time: 100 ms\nLibrary load time: 1 ms\n"
                                      "Generated execute time: 4 ms\nResult materialization/fetch time: 1 ms\n"
                                      "Total query time: 110 ms\n")
        runner.normalize_timing_record(record)
        self.assertEqual(record["sql_to_pcuda_ms"], 3.25)
        self.assertEqual(record["timing_component_sum_ms"], 109)
        self.assertEqual(record["timing_total_minus_components_ms"], 1)

    def test_missing_intervals_remain_unknown(self):
        record = runner.normalize_timing_record({})
        for field in ("sql_to_pcuda_ms", "startup_ms", "data_load_ms", "main_to_first_result_ms"):
            self.assertIsNone(record[field])

    def test_startup_is_separate_from_query(self):
        record = runner.parse_metrics("Startup time: 200 ms\nData loading time: 3000 ms\n"
                                      "Main to first result time: 3700 ms\nTotal query time: 500 ms\n")
        self.assertEqual(record["startup_ms"], 200)
        self.assertEqual(record["data_load_ms"], 3000)
        self.assertEqual(record["main_to_first_result_ms"], 3700)
        self.assertEqual(record["total_query_ms"], 500)


if __name__ == "__main__":
    unittest.main()
