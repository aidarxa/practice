"""Behavioral checks for typed transcripts and the public preparation entry point."""
import importlib.util
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("native_validation_client", ROOT / "test/ssb_native_data/validate.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class Clients(unittest.TestCase):
    def test_mixed_typed_transcript(self):
        columns, rows, count = client.PARSER.parse_rows('--- Result ---\nColumns: year | city | value\n1993 | "A | B " | -9\nRows returned: 1\n')
        self.assertEqual((columns, rows, count), (["year", "city", "value"], [(1993, "A | B ", -9)], 1))

    def test_native_key_order(self):
        self.assertEqual(client.canonical("q21", ["SUM", "d_year", "p_brand1"], [(42, 1992, "MFGR#12")]), [(1992, "MFGR#12", 42)])
        with self.assertRaises(ValueError):
            client.canonical("q21", ["SUM", "d_year"], [])

    def test_empty_typed_rows(self):
        self.assertEqual(client.PARSER.parse_rows('--- Result ---\nColumns: city\nRows returned: 0\n'), (["city"], [], 0))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix="ssb-script-") as directory:
            path = Path(directory)
            (path / "user-file").write_text("keep me")
            result = subprocess.run(["bash", str(ROOT / "bench/prepare_ssb.sh"), "--output", str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Preparation artifacts:", result.stdout)
            self.assertEqual((path / "user-file").read_text(), "keep me")

    def test_scale_is_not_silently_truncated(self):
        for scale in ("0", "-1", "nan", "1.5", "100001", "1;touch x"):
            result = subprocess.run(["bash", str(ROOT / "bench/prepare_ssb.sh"), "--scale", scale], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Preparation artifacts:", result.stdout)

    def test_concurrent_import_does_not_overwrite(self):
        converter = os.environ.get("SSB_CONVERTER")
        tables = os.environ.get("SSB_TEST_TABLES")
        if not converter or not tables:
            self.skipTest("Set SSB_CONVERTER and SSB_TEST_TABLES for the real converter race")
        with tempfile.TemporaryDirectory(prefix="ssb-race-") as directory:
            output = Path(directory) / "columns"
            argv = [converter, "import", tables, str(output), "0.01", "concurrent-fixture"]
            processes = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
            results = [process.communicate(timeout=120) for process in processes]
            self.assertEqual(sorted(process.returncode for process in processes), [0, 1], results)
            result = subprocess.run([converter, "verify", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_full_public_preparation_and_output_isolation(self):
        if os.environ.get("SSB_RUN_PREPARATION") != "1":
            self.skipTest("Set SSB_RUN_PREPARATION=1 for a host-only CMake/dbgen integration run")
        with tempfile.TemporaryDirectory(prefix="ssb-public-") as directory:
            root = Path(directory)
            protected = root / "unrelated"
            protected.mkdir()
            sentinel = protected / "supplier.tbl"
            sentinel.write_text("user data must not be replaced")
            env = dict(os.environ, DSS_PATH=str(protected))
            result = subprocess.run(["bash", str(ROOT / "bench/prepare_ssb.sh"), "--scale", "0.001", "--output", str(root / "data")],
                                    env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(sentinel.read_text(), "user data must not be replaced")
            self.assertEqual(list(protected.iterdir()), [sentinel])
            run_line = next(line for line in result.stdout.splitlines() if line.startswith("Preparation artifacts: "))
            run = Path(run_line.split(": ", 1)[1])
            self.assertFalse((run / "tables").exists())
            self.assertTrue((root / "data/dataset.manifest").is_file())

    def test_invalid_dataset_is_rejected_before_gpu_selection(self):
        build = os.environ.get("SSB_TEST_BUILD")
        if not build:
            self.skipTest("Set SSB_TEST_BUILD for the real SQL/manual loader rejection checks")
        with tempfile.TemporaryDirectory(prefix="ssb-invalid-") as directory:
            env = dict(os.environ, CRYSTAL_CONFIG=str(ROOT / "crystal.conf.example"), CRYSTAL_DATA_DIR=directory,
                       CRYSTAL_DEVICE_NAME="__must_not_select_a_device__")
            for name, arguments in (("db_cli", []), ("ssb_q21", [directory, "1"])):
                result = subprocess.run([str(Path(build) / name), *arguments], env=env, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("dataset.manifest", result.stderr)


class Preflight(unittest.TestCase):
    """Exercise availability checks only; fixture executables must never run SQL/GPU work."""
    def fixture(self, root, manual=True):
        build, data = root / "build with spaces", root / "data"
        data.mkdir()
        (data / "dataset.manifest").write_text("CRYSTAL_SSB_NATIVE_1\n")
        names = ["db_cli", "bench/loader/ssb_reference"]
        if manual:
            names += ["ssb_" + query for query in client.QUERIES]
        for name in names:
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/sh\nexit 99\n")
            path.chmod(0o700)
        return build, data

    def rejects_before_sql(self, build, data, output, *flags):
        error = io.StringIO()
        argv = ["validate.py", "--build", str(build), "--data", str(data), "--output-root", str(output), *flags]
        with patch.object(client.sys, "argv", argv), redirect_stderr(error), \
                patch.object(client, "process", side_effect=AssertionError("SQL process started before preflight")) as process:
            with self.assertRaises(SystemExit) as result:
                client.main()
        self.assertEqual(result.exception.code, 2)
        process.assert_not_called()
        self.assertFalse(output.exists())
        self.assertNotIn("Traceback", error.getvalue())
        return error.getvalue()

    def test_missing_manual_rejected_before_sql_and_output(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            root = Path(directory)
            build, data = self.fixture(root, manual=False)
            error = self.rejects_before_sql(build, data, root / "results")
            self.assertIn("ssb_q11", error)
            self.assertIn("ssb_q43", error)
            self.assertIn("cmake --build", error)
            self.assertIn("--sql-only", error)
            self.assertIn("'" + str(build) + "'", error)

    def test_all_missing_binaries_are_reported(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            root = Path(directory)
            error = self.rejects_before_sql(root / "missing build", root / "data", root / "results")
            self.assertIn("db_cli", error)
            self.assertIn("ssb_reference", error)
            self.assertIn("ssb_q11", error)
            self.assertIn("ssb_q43", error)

    def test_sql_only_does_not_require_manual(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            build, data = self.fixture(Path(directory), manual=False)
            client.validate_inputs(build, data, sql_only=True)

    def test_sql_only_still_requires_cpu_oracle(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            root = Path(directory)
            build, data = self.fixture(root, manual=False)
            (build / "bench/loader/ssb_reference").unlink()
            error = self.rejects_before_sql(build, data, root / "results", "--sql-only")
            self.assertIn("ssb_reference", error)
            self.assertNotIn("ssb_q11", error)

    def test_nonexecutable_binary_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            build, data = self.fixture(Path(directory))
            (build / "ssb_q11").chmod(0o600)
            with self.assertRaisesRegex(ValueError, "ssb_q11"):
                client.validate_inputs(build, data)

    def test_complete_build_passes_availability_checks(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            build, data = self.fixture(Path(directory))
            client.validate_inputs(build, data)

    def test_missing_manifest_is_rejected_before_sql(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            root = Path(directory)
            build, data = self.fixture(root)
            (data / "dataset.manifest").unlink()
            error = self.rejects_before_sql(build, data, root / "results")
            self.assertIn("dataset.manifest", error)

    def test_optional_original_tables_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="ssb-preflight-") as directory:
            root = Path(directory)
            build, data = self.fixture(root)
            tables = root / "tables"
            tables.mkdir()
            for name in ("lineorder.tbl", "supplier.tbl", "customer.tbl", "part.tbl"):
                (tables / name).write_text("")
            error = self.rejects_before_sql(build, data, root / "results", "--tables", str(tables))
            self.assertIn("date.tbl", error)
            (tables / "date.tbl").write_text("")
            client.validate_inputs(build, data, tables=tables)


if __name__ == "__main__":
    unittest.main()
