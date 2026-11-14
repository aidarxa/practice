## @file
## @brief Behavioral regressions for measurement metadata, offline parsing and mutation isolation.
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    """@brief Load the actual runner/model, without subprocess timing or working-tree edits."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load("baseline", "test/pcuda_stage0/sql_baseline.py")
fixture = load("fixture", "test/pcuda_stage2/run_fixture.py")
mutations = load("mutations", "test/pcuda_stage1/mutate.py")
quality = load("quality_model", "test/pcuda_stage1/quality.py")
cli = load("stage2_cli", "test/pcuda_stage2/validate_cli.py")


class ToolsTests(unittest.TestCase):
    def test_lowering_is_not_compilation(self):
        events = [{"phase": "pch_prepare"}, {"phase": "compile_and_link"}]
        records = [{"run": 1, "compile_ms": 0, "codegen_ms": 0.2}]
        self.assertFalse(baseline.compiler_cache_verified(events, records, 1, "legacy-generated-v1"))
        self.assertTrue(baseline.compiler_cache_verified(events, records, 1, "explicit-plan-v1"))
        self.assertFalse(baseline.compiler_cache_verified(events + events[1:], records, 1, "explicit-plan-v1"))
        records[0]["compile_ms"] = 1
        self.assertFalse(baseline.compiler_cache_verified(events, records, 1, "explicit-plan-v1"))

    def test_native_signal_diagnostic_does_not_destroy_timer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "native.json"
            raw = 'shell: Aborted (core dumped)\n{"wall_s":1.5,"user_s":1,"system_s":0.1}\n'
            path.write_text(raw)
            self.assertEqual(fixture.native_time(path), {"wall_ms": 1500, "user_ms": 1000, "system_ms": 100})
            self.assertEqual(path.read_text(), raw)

    def test_stale_objects_are_not_link_inputs(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            stale = root / "CMakeFiles/test.dir/stale.o"
            stale.parent.mkdir(parents=True)
            stale.touch()
            (root / "compile_commands.json").write_text(json.dumps([{
                "directory": str(root), "command": "c++ -c active.cpp -o CMakeFiles/test.dir/active.o"}]))
            self.assertEqual(mutations.active_objects(root, ("test",)), [stale.with_name("active.o")])

    def test_utf8_offsets_and_template_instances(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            source = root / "sample.cpp"
            text = "// кириллица\nvoid f(){\n if(x){}\n" + " value += 1;\n" * 16 + "}\n"
            source.write_text(text)
            encoded = source.read_bytes()
            body = {"kind": "CompoundStmt", "range": {"begin": {"offset": encoded.index(b"{")},
                    "end": {"offset": encoded.rindex(b"}"), "tokLen": 1}}, "inner": [{"kind": "IfStmt"}]}
            functions = [{"kind": "FunctionDecl", "name": "f", "mangledName": name, "inner": [body]} for name in ("f_i", "f_d")]
            ast = root / "ast.json"
            ast.write_text(json.dumps({"kind": "NamespaceDecl", "loc": {"file": str(source)}, "inner": functions}))
            covered = {"data": [{"functions": [{"name": name, "regions": [[2,1,4,1,1,0,0,0]]} for name in ("f_i", "f_d")]}]}
            report = quality.analyze(source, ast, covered)
            self.assertEqual([row["line"] for row in report["functions"]], [2, 2])
            self.assertEqual(report["duplicates_min_50_tokens_rank_08"], [])

    def test_topk_oracle_from_independent_columns(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            for filename, values in (("SUPPLIER0", [1,0,3,4]), ("SUPPLIER4", [2,0,1,2]), ("SUPPLIER5", [4,0,3,1])):
                np.array(values, dtype="<i4").tofile(root / filename)
            self.assertEqual(cli.expected_rows("stage2-topk", 3, root), [(1,3),(2,1),(2,4)])


if __name__ == "__main__":
    unittest.main()
