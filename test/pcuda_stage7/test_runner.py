## @file
## @brief Offline checks of manual multiplicity, diagnostics and explicit text scope.
import unittest
from validate import manual_trials, unexpected_diagnostics, storage_workload


class RunnerChecks(unittest.TestCase):
    def test_manual_exact_integer_and_multiplicity(self):
        trials = manual_trials("__SSB_RESULT_BEGIN__\n-1 9007199254740993\n-1 9007199254740993\n__SSB_RESULT_END__\n")
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0][(-1, 9007199254740993)], 2)
        self.assertNotIn((-1, 9007199254740992), trials[0])
        self.assertEqual(manual_trials("no result"), [])

    def test_only_cold_cache_advisory_is_accepted(self):
        warning = "\x1b[;35m[AdaptiveCpp Warning] \x1b[0mkernel_cache: This application run has resulted in new binaries being JIT-compiled. extra detail\n"
        self.assertEqual(unexpected_diagnostics(warning), [])
        self.assertEqual(unexpected_diagnostics(warning + "Manual SSB error: wrong\n"), ["Manual SSB error: wrong"])
        self.assertEqual(unexpected_diagnostics("[AdaptiveCpp Warning] something else\n"), ["[AdaptiveCpp Warning] something else"])

    def test_explicit_text_and_ready_numeric_scope(self):
        items = storage_workload()
        self.assertEqual(len(items), 21)
        self.assertEqual(sum(item[2] == "unsupported" for item in items), 8)
        ready = [item for item in items if item[4]]
        self.assertEqual(len(ready), 4)
        self.assertTrue(all(item[2] == "filter" for item in ready))
        self.assertEqual(len({item[0] for item in items}), len(items))


if __name__ == "__main__":
    unittest.main()
