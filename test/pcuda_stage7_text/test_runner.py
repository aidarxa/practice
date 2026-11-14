"""Host-only checks of the independent oracles and lossless transcript parser."""
import importlib.util
import json
from pathlib import Path
import unittest

import validate_gpu

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("text_cli_rows", ROOT/"test/pcuda_stage1/validate_cli.py")
rows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rows)


class RunnerChecks(unittest.TestCase):
    def test_lossless_text_cells(self):
        values = ["", "NULL", "a ", "a|b", 'quote"slash\\\n\t', "Я🙂"]
        line = "Row 4: " + " | ".join(json.dumps(s,ensure_ascii=False) for s in values) + " | NULL | 9007199254740993 | -0.5"
        text = "--- Result ---\nColumns: " + " | ".join(["v"]*9) + "\n" + line + "\nRows returned: 1\n"
        _, actual, count = rows.parse_rows(text)
        self.assertEqual(count,1)
        self.assertEqual(actual,[tuple(values+[None,9007199254740993,-0.5])])

    def test_empty_text_is_a_row(self):
        self.assertEqual(rows.parse_rows('--- Result ---\nColumns: text\n""\nRows returned: 1\n')[1],[("",)])
        self.assertEqual(rows.parse_rows('--- Result ---\nColumns: text\n0\nRows returned: 0\n')[1],[])

    def test_marker_and_counters_are_mandatory(self):
        with self.assertRaises(ValueError):
            validate_gpu.parse_rows("no result")
        database = validate_gpu.fixture()
        item = dict(sql="SELECT c_name FROM customer WHERE c_custkey=1",oracle=None,error=None,types="T",ready=True)
        text = "__TEXT_ROWS__ 1 1\nT\n__TEXT_EVENTS__ 0 0 0 0 0 0\n"
        self.assertEqual(validate_gpu.check(item,text,database)["rows"],[("",)])
        with self.assertRaises(ValueError):
            validate_gpu.check(item,text.replace("0 0 0 0 0 0","0 1 0 0 0 0"),database)


if __name__ == "__main__":
    unittest.main()
