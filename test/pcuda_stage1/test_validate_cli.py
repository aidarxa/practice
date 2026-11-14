## @file
## @brief Behaviour tests for typed CLI parsing, including its zero-row placeholder.
import unittest

from validate_cli import parse_rows, targeted_queries, workload


class CliParsingTests(unittest.TestCase):
    def test_empty_multicolumn_placeholder_is_not_a_row(self):
        self.assertEqual(parse_rows("--- Result ---\nColumns: grp | total\n0\nRows returned: 0\n"),
                         (["grp", "total"], [], 0))

    def test_scalar_integer_keeps_precision(self):
        self.assertEqual(parse_rows("--- Result ---\nColumns: exact\n9007199254740993\nRows returned: 1\n")[1],
                         [(9007199254740993,)])

    def test_typed_null_and_float(self):
        self.assertEqual(parse_rows("--- Result ---\nColumns: a | b\nRow 0: NULL | 1.25\nRows returned: 1\n")[1],
                         [(None, 1.25)])

    def test_missing_result_rejected(self):
        with self.assertRaises(ValueError):
            parse_rows("[ERROR] SQL did not execute")

    def test_row_width_rejected(self):
        with self.assertRaises(ValueError):
            parse_rows("--- Result ---\nColumns: a | b\nRow 0: 1\nRows returned: 1\n")

    def test_workload_contains_repeats_and_explicit_legacy_limit(self):
        cases = targeted_queries()
        self.assertEqual(sum(case[0].endswith("-repeat") for case in cases), 3)
        self.assertEqual(sum(case[2] == "legacy-rejection" for case in cases), 1)
        self.assertEqual(len(workload(False, False)), len(cases))


if __name__ == "__main__":
    unittest.main()
