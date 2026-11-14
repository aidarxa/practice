## @file
## @brief Offline oracle/reuse scope and exact integer comparison checks.
import unittest
from validate_cli import equal, workload


class RunnerChecks(unittest.TestCase):
    def test_exact_integers(self):
        self.assertTrue(equal([[9007199254740993]], [[9007199254740993]]))
        self.assertFalse(equal([[9007199254740992]], [[9007199254740993]]))
        self.assertFalse(equal([[float(9007199254740993)]], [[9007199254740993]]))

    def test_null_shape_and_fraction(self):
        self.assertTrue(equal([[None, -0.5]], [[None, -0.5]]))
        self.assertFalse(equal([[0, -0.5]], [[None, -0.5]]))
        self.assertFalse(equal([], [[None]]))
        self.assertFalse(equal([[0]], [[0, 1]]))

    def test_reuse_scope(self):
        items = workload()
        self.assertEqual(len(items), 27)
        self.assertEqual(len({item[0] for item in items}), len(items))
        ready = {item[0] for item in items if item[4]}
        self.assertIn("case-3", ready)
        self.assertIn("scalar-6", ready)
        self.assertNotIn("scalar-4", ready)  # a previously missing integer schema is allowed to build
        self.assertNotIn("null-first-1", ready)


if __name__ == "__main__":
    unittest.main()
