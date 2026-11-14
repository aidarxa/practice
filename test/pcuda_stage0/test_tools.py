import unittest
from collections import Counter
import os
from pathlib import Path
import sys
import json
import tempfile

from sql_baseline import cli_rows, isolate_sql, reference_rows, run_session
from phase_trace import parse_trace, union_ms
from native_timing import read_time, timed_command, trace_summary


class ReferenceTests(unittest.TestCase):
    def test_sql_isolation_does_not_change_tokens(self):
        original = "select sum(lo_revenue) from lineorder;"
        isolated = isolate_sql(original, "a5123")
        self.assertEqual(original.split(), isolated.split())
        self.assertNotEqual(original, isolated)
        self.assertEqual(isolated, isolate_sql(original, "a5123"))

    def test_scalar_does_not_count_repetitions(self):
        self.assertEqual(reference_rows("Result: 123\nTotal time: 1 ms\nResult: 123\n"),
                         Counter({(123,): 1}))

    def test_grouped_first_run_and_multiplicity(self):
        text = "1992 8 50\n1992 8 50\nResult count: 2\nTotal time: 1 ms\n1992 8 50\n"
        self.assertEqual(reference_rows(text), Counter({(1992, 8, 50): 2}))

    def test_incomplete_reference(self):
        for text in ("1992 8 50\n", "1992 8 50\nResult count: 2\n"):
            with self.assertRaises(ValueError):
                reference_rows(text)


class CliTests(unittest.TestCase):
    def test_reorders_columns_not_values(self):
        text = ("--- Result ---\nColumns: c_city | s_city | d_year | revenue\n"
                "Row 0: 231 | 235 | 1992 | 123\nRows returned: 1\n--- End ---\n")
        self.assertEqual(cli_rows(text, ("d_year", "c_city", "s_city")),
                         Counter({(1992, 231, 235, 123): 1}))

    def test_scalar(self):
        self.assertEqual(cli_rows("--- Result ---\nColumns: SUM\n123\nRows returned: 1\n", ()),
                         Counter({(123,): 1}))

    def test_rejects_truncation_wrong_width_and_schema(self):
        for text in ("--- Result ---\nColumns: SUM\n123\nRows returned: 2\n",
                     "--- Result ---\nColumns: SUM\n123 | 4\nRows returned: 1\n",
                     "--- Result ---\nColumns: x | x\n1 | 2\nRows returned: 1\n",
                     "no result"):
            with self.assertRaises(ValueError):
                cli_rows(text, ())


class ProcessTests(unittest.TestCase):
    def test_native_time_separates_stderr_and_uses_json_numbers(self):
        with tempfile.TemporaryDirectory(prefix='stage0-native-test-', dir='/tmp/opencode') as directory:
            path = Path(directory)
            env = os.environ.copy()
            env['LC_ALL'] = 'ru_RU.UTF-8'
            command = timed_command(['/bin/sh', '-c', 'echo output; echo diagnostic >&2'], path / 'time.json', path / 'errors')
            code, stdout, _, timed_out = run_session(command, '', Path.cwd(), env, 5)
            self.assertEqual(code, 0)
            self.assertFalse(timed_out)
            self.assertEqual(stdout.strip(), 'output')
            self.assertEqual((path / 'errors').read_text().strip(), 'diagnostic')
            self.assertGreaterEqual(read_time(path / 'time.json')['wall_ms'], 0)

    def test_offline_trace_summary_does_not_sum_nested_events(self):
        with tempfile.TemporaryDirectory(prefix='stage0-trace-test-', dir='/tmp/opencode') as directory:
            path = Path(directory, 'trace.json')
            path.write_text(json.dumps({'traceEvents': [
                {'name': 'ExecuteCompiler', 'dur': 10000},
                {'name': 'Total Frontend', 'dur': 7000},
                {'name': 'Total Backend', 'dur': 2000}]}))
            self.assertEqual(trace_summary(path), {'ExecuteCompiler': 10, 'Total Frontend': 7, 'Total Backend': 2})

    def test_compiler_trace_nested_linker_and_unfinished_exec(self):
        text = ('1 10.000 execve("/usr/bin/clang++", [], []) <unfinished ...>\n'
                '1 10.001 <... execve resumed>) = 0\n'
                '2 10.800 execve("/usr/bin/ld.lld", [], []) = 0\n'
                '2 10.900 +++ exited with 0 +++\n'
                '1 11.000 +++ exited with 0 +++\n')
        phases = parse_trace(text)
        self.assertAlmostEqual(phases['compiler_process_ms'], 900)
        self.assertAlmostEqual(phases['linker_process_ms'], 100)
        self.assertAlmostEqual(union_ms([(1, 3), (2, 4), (5, 6)]), 4000)

    def test_timeout_preserves_output(self):
        code, stdout, _, timed_out = run_session(
            [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
            "", Path.cwd(), os.environ.copy(), 0.2)
        self.assertTrue(timed_out)
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout.strip(), "started")


if __name__ == "__main__":
    unittest.main()
