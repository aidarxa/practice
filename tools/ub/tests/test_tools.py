"""Behavioral tests: real C++17 compilation, LLVM profiles and isolated mutants."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
CLANG = os.environ.get("UB_CXX", shutil.which("clang++") or "clang++")
LLVM_COV = os.environ.get("UB_LLVM_COV", shutil.which("llvm-cov") or "llvm-cov")
LLVM_PROFDATA = os.environ.get("UB_LLVM_PROFDATA", shutil.which("llvm-profdata") or "llvm-profdata")


class ToolsTests(unittest.TestCase):
    def setUp(self):
        preferred = Path("/tmp/opencode")
        self.temp = tempfile.TemporaryDirectory(prefix="ub-tests-", dir=preferred if preferred.is_dir() else None)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "проект с пробелом"
        self.root.mkdir()
        self.source = self.root / "logic.cpp"
        shutil.copyfile(TOOLS / "tests/fixtures/logic.cpp", self.source)

    def cli(self, name, *args, expected=0):
        command = [sys.executable, str(TOOLS / (name + ".py")), *map(str, args)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def analysis(self, name, out, *args, expected=0):
        return self.cli(name, "--source", self.source, "--clang", CLANG,
                        "--ast-filter", "fixture", "--output", self.root / out,
                        *args, expected=expected)

    def recipe(self):
        path = self.root / "recipe.json"
        recipe = {
            "version": 1, "inputs": ["logic.cpp"],
            "build": [["{cxx}", "-std=c++17", "-O0", "-UNDEBUG", "{flags}",
                       "{root}/logic.cpp", "-o", "{out}/test"]],
            "test": [["{out}/test"]], "objects": ["{out}/test"],
            "mutations": [
                {"id": "killed", "file": "logic.cpp", "before": "return x + 1;", "after": "return x + 2;"},
                {"id": "survived", "file": "logic.cpp", "before": "if (x) return 1;", "after": "if (x) return 2;"},
                {"id": "invalid-cpp", "file": "logic.cpp", "before": "return x + 1;", "after": "return missing;"},
                {"id": "deadline", "file": "logic.cpp", "before": "return x + 1;", "after": "while (true) {}"},
            ],
        }
        path.write_text(json.dumps(recipe))
        return path, recipe

    def coverage(self):
        path, _ = self.recipe()
        self.cli("coverage", "--project", self.root, "--recipe", path, "--output", self.root / "coverage",
                 "--cxx", CLANG, "--llvm-cov", LLVM_COV, "--llvm-profdata", LLVM_PROFDATA)
        return self.root / "coverage/coverage.json"

    def test_dryer_cross_file_and_exact_scope(self):
        second = self.root / "copy.cpp"
        second.write_text("namespace another { int copy(int x) { int total = x; total += 1; total *= 2; total -= 3; return total; } }")
        self.cli("dryer", "--source", self.source, "--source", second, "--clang", CLANG,
                 "--min-tokens", 20, "--output", self.root / "duplicates", expected=1)
        report = json.loads((self.root / "duplicates/report.json").read_text())
        self.assertTrue(any(row["left"]["source"] != row["right"]["source"] for row in report["duplicates"]))
        self.assertTrue(all(row["rank"] >= 0.8 for row in report["duplicates"]))
        self.assertFalse(any("identity" in row["left"]["name"] and "identity" in row["right"]["name"]
                              for row in report["duplicates"]))

    def test_dryer_does_not_compare_a_function_with_its_own_lambda(self):
        body = "{ int total=x; " + "total+=1; total*=2; total-=3; " * 6 + "return total; }"
        self.source.write_text("namespace fixture { int parent(int x) { auto child=[](int x) " + body + "; return child(x); } }")
        self.analysis("dryer", "nested", "--min-tokens", 20)
        report = json.loads((self.root / "nested/report.json").read_text())
        self.assertTrue(any(row["name"] == "operator()" for row in report["functions"]))
        self.assertEqual(report["duplicates"], [])
        # A different function copying the lambda body must still be found.
        self.source.write_text(self.source.read_text() + " namespace fixture { int copy(int x) " + body + " }")
        self.analysis("dryer", "nested-copy", "--min-tokens", 20, expected=1)
        duplicates = json.loads((self.root / "nested-copy/report.json").read_text())["duplicates"]
        self.assertTrue(any({row["left"]["name"], row["right"]["name"]} == {"operator()", "copy"} for row in duplicates))

    def test_empty_scope_is_error_and_output_is_exclusive(self):
        self.analysis("dryer", "empty", "--function", "does_not_exist", expected=2)
        self.analysis("dryer", "empty", expected=2)

    def test_complexity_coverage_unknown_and_lambda(self):
        coverage = self.coverage()
        self.analysis("crapper", "quality", "--coverage", coverage)
        report = json.loads((self.root / "quality/report.json").read_text())
        choose = next(row for row in report["functions"] if row["name"] == "choose")
        self.assertEqual(choose["complexity"], 3)
        self.assertGreater(choose["region_coverage"], 0)
        self.assertLess(choose["region_coverage"], 1)
        self.assertEqual(choose["line"], 3)
        missing = next(row for row in report["functions"] if row["name"] == "duplicate_b")
        self.assertEqual(missing["region_coverage"], 0)
        outer = next(row for row in report["functions"] if row["name"] == "with_lambda")
        self.assertEqual(outer["complexity"], 1)
        self.assertTrue(any(row["name"] == "operator()" and row["complexity"] == 2 for row in report["functions"]))
        self.assertTrue(any(row["name"] == "Owner" and row["complexity"] == 3 for row in report["functions"]))
        self.assertTrue(report["success"])  # A simple uncovered function can still have CRAP < 10.
        self.assertEqual(report["coverage_provenance"], "coverage.py source and export hashes verified")

    def test_defaulted_instantiated_moves_have_no_written_body(self):
        self.source.write_text("namespace fixture { struct Owner { int value; Owner()=default; "
                               "Owner(Owner&&)=default; Owner& operator=(Owner&&)=default; "
                               "int get() { return value; } }; "
                               "int check() { Owner a; a.value=3; Owner b(static_cast<Owner&&>(a)); "
                               "a=static_cast<Owner&&>(b); return a.get(); } } "
                               "int main() { return fixture::check()!=3; }")
        coverage = self.coverage()
        self.analysis("crapper", "defaulted", "--coverage", coverage)
        report = json.loads((self.root / "defaulted/report.json").read_text())
        self.assertEqual({row["name"] for row in report["functions"]}, {"get", "check"})
        self.analysis("dryer", "defaulted-dryer")

    def test_crapper_never_passes_missing_coverage_or_wrong_file(self):
        path = self.root / "coverage.json"
        path.write_text(json.dumps({"type": "llvm.coverage.json.export", "data": [{"functions": [{
            "name": "_ZN7fixture6chooseEi", "filenames": [str(self.root / "other.cpp")],
            "regions": [[3, 1, 6, 2, 1, 0, 0, 0]],
        }]}]}))
        self.analysis("crapper", "wrong", "--coverage", path, "--function", "choose", expected=1)
        row = json.loads((self.root / "wrong/report.json").read_text())["functions"][0]
        self.assertEqual(row["coverage_status"], "missing")
        self.assertIsNone(row["crap_region"])

    def test_threshold_is_strict_and_no_invented_coverage(self):
        from crapper import score
        self.assertEqual(score(3, 0.0), 12.0)
        self.assertEqual(score(10, 1.0), 10.0)
        self.assertEqual(score(2, 0.5), 2.5)

    def test_only_code_regions_from_the_selected_file_enter_crap(self):
        path = self.root / "coverage.json"
        path.write_text(json.dumps({"type": "llvm.coverage.json.export", "data": [{"functions": [{
            "name": "_ZN7fixture6chooseEi", "filenames": [str(self.source), str(self.root / "macro.h")],
            "regions": [[3, 1, 6, 2, 0, 0, 0, 0], [4, 1, 4, 6, 100, 0, 1, 1], [4, 1, 4, 6, 100, 1, 0, 0]],
        }]}]}))
        self.analysis("crapper", "zero", "--coverage", path, "--function", "choose", expected=1)
        row = json.loads((self.root / "zero/report.json").read_text())["functions"][0]
        self.assertEqual(row["region_coverage"], 0)
        self.assertEqual(row["code_regions"], 1)
        self.assertEqual(row["crap_region"], 12)

    def test_coverage_provenance_rejects_changed_sources(self):
        coverage = self.coverage()
        self.source.write_text(self.source.read_text() + "\n// source changed\n")
        result = self.analysis("crapper", "stale", "--coverage", coverage, expected=2)
        self.assertIn("stale", result.stderr)

    def test_coverage_warning_is_not_a_successful_export(self):
        wrapper = self.root / "llvm-cov-warning"
        wrapper.write_text(f"#!{sys.executable}\nimport subprocess,sys\n"
                           f"code=subprocess.call([{LLVM_COV!r},*sys.argv[1:]])\n"
                           "if sys.argv[1]=='export': print('warning: 2 functions have mismatched data',file=sys.stderr)\n"
                           "sys.exit(code)\n")
        wrapper.chmod(0o755)
        path, _ = self.recipe()
        self.cli("coverage", "--project", self.root, "--recipe", path, "--output", self.root / "warning",
                 "--cxx", CLANG, "--llvm-cov", wrapper, "--llvm-profdata", LLVM_PROFDATA, expected=2)
        self.assertFalse((self.root / "warning/coverage.json").exists())
        self.assertIn("mismatched", (self.root / "warning/export.stderr").read_text())

    def test_coverage_accepts_explicit_compiler_flags(self):
        self.source.write_text("#ifndef UB_BUILD_FLAG\n#error flag not forwarded\n#endif\n" + self.source.read_text())
        path, _ = self.recipe()
        self.cli("coverage", "--project", self.root, "--recipe", path, "--output", self.root / "flags",
                 "--cxx", CLANG, "--llvm-cov", LLVM_COV, "--llvm-profdata", LLVM_PROFDATA,
                 "--flag=-DUB_BUILD_FLAG=1")

    def test_lexer_preserves_raw_literals_and_operators(self):
        from cpp import tokens
        self.assertEqual(tokens('a /* comment */ && b // comment\n'), ["a", "&&", "b"])
        self.assertEqual(tokens('u8R"tag(/* not comment */ \\" )tag"'), ['u8R"tag(/* not comment */ \\" )tag"'])
        self.assertNotEqual(tokens("slot(0)"), tokens("slot(1)"))
        self.assertEqual(tokens("1.5e-2 + 0xffULL"), ["1.5e-2", "+", "0xffULL"])

    def test_compile_database_arguments_and_working_directory(self):
        inc = self.root / "include dir"
        inc.mkdir()
        (inc / "value.h").write_text("constexpr int VALUE = 3;")
        self.source.write_text('#include "value.h"\nint unique() { return VALUE; }')
        db = self.root / "compile_commands.json"
        db.write_text(json.dumps([{"directory": str(self.root), "file": "logic.cpp", "arguments": [
            CLANG, "-std=c++17", "-Iinclude dir", "-c", "logic.cpp", "-o", "unused.o", "-MD", "-MF", "unused.d"]}]))
        self.cli("dryer", "--source", self.source, "--compile-db", db, "--output", self.root / "db", "--clang", CLANG)
        self.assertFalse((self.root / "unused.o").exists())
        self.assertFalse((self.root / "unused.d").exists())

    def test_compile_database_never_silently_drops_driver_options(self):
        db = self.root / "compile_commands.json"
        db.write_text(json.dumps([{"directory": str(self.root), "file": "logic.cpp", "arguments": [
            CLANG, "-object-file-name=module.o", "-c", "logic.cpp", "-o", "unused.o"]}]))
        self.cli("dryer", "--source", self.source, "--compile-db", db, "--output", self.root / "driver-flag",
                 "--clang", CLANG)
        command = json.loads((self.root / "driver-flag/ast-0.command.json").read_text())
        self.assertIn("-object-file-name=module.o", command["argv"])
        db.write_text(json.dumps([{"directory": str(self.root), "file": "logic.cpp", "arguments": [
            CLANG, "--ub-unknown-option", "-c", "logic.cpp", "-o", "unused.o"]}]))
        self.cli("dryer", "--source", self.source, "--compile-db", db, "--output", self.root / "unknown-flag",
                 "--clang", CLANG, expected=2)
        self.assertIn("--ub-unknown-option", (self.root / "unknown-flag/ast-0.stderr").read_text())

    def test_compact_object_output_does_not_write_into_the_project(self):
        db = self.root / "compile_commands.json"
        db.write_text(json.dumps([{"directory": str(self.root), "file": "logic.cpp", "arguments": [
            CLANG, "-c", "logic.cpp", "-ounused.o"]}]))
        self.cli("dryer", "--source", self.source, "--compile-db", db, "--output", self.root / "compact-output", "--clang", CLANG)
        self.assertFalse((self.root / "unused.o").exists())

    def test_target_header_through_translation_unit(self):
        header = self.root / "sample.h"
        header.write_text("#pragma once\nnamespace fixture { inline int header_fn(int x) { if (x) return 1; return 0; } }\n")
        self.source.write_text('#include "sample.h"\nnamespace fixture { int in_tu() { return header_fn(1); } }')
        self.cli("dryer", "--source", header, "--tu", self.source, "--clang", CLANG, "--output", self.root / "header")
        functions = json.loads((self.root / "header/report.json").read_text())["functions"]
        self.assertEqual([row["name"] for row in functions], ["header_fn"])

    def test_mutator_classifies_without_editing_working_tree(self):
        recipe, _ = self.recipe()
        original = self.source.read_bytes()
        self.cli("mutator", "run", "--project", self.root, "--recipe", recipe, "--cxx", CLANG,
                 "--test-timeout", 0.2, "--output", self.root / "mutants", expected=1)
        report = json.loads((self.root / "mutants/report.json").read_text())
        self.assertEqual({row["id"]: row["status"] for row in report["mutations"]}, {
            "killed": "killed", "survived": "survived", "invalid-cpp": "build-failed", "deadline": "timeout"})
        self.assertEqual(report["mutation_score"], 0.5)
        self.assertFalse(report["success"])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual((self.root / "mutants/baseline/source/logic.cpp").read_bytes(), original)

    def test_mutator_rejects_traversal_ambiguous_patch_and_empty_campaign(self):
        path, recipe = self.recipe()
        for name, mutations in [("escape", [{"id": "x", "file": "../logic.cpp", "before": "x", "after": "y"}]),
                                ("ambiguous", [{"id": "x", "file": "logic.cpp", "before": "return total;", "after": "return 0;"}]),
                                ("empty", [])]:
            recipe["mutations"] = mutations
            path.write_text(json.dumps(recipe))
            self.cli("mutator", "run", "--project", self.root, "--recipe", path, "--cxx", CLANG,
                     "--output", self.root / name, expected=2)
        self.assertFalse((self.root.parent / "logic.cpp").exists())

    def test_mutator_baseline_must_be_rebuilt_and_pass(self):
        path, recipe = self.recipe()
        self.source.write_text("int main(){ return 1; }")
        recipe["mutations"] = [{"id": "x", "file": "logic.cpp", "before": "return 1;", "after": "return 0;"}]
        path.write_text(json.dumps(recipe))
        self.cli("mutator", "run", "--project", self.root, "--recipe", path, "--cxx", CLANG,
                 "--output", self.root / "bad-baseline", expected=2)
        self.assertFalse((self.root / "bad-baseline/x").exists())

    def test_mutation_id_cannot_collide_with_artifact_names(self):
        path, recipe = self.recipe()
        recipe["mutations"] = [{**recipe["mutations"][0], "id": "report.json"}]
        path.write_text(json.dumps(recipe))
        self.cli("mutator", "run", "--project", self.root, "--recipe", path, "--cxx", CLANG,
                 "--output", self.root / "artifact-name")
        report = json.loads((self.root / "artifact-name/report.json").read_text())
        self.assertEqual(report["counts"]["killed"], 1)

    def test_snapshot_preserves_executable_build_helpers(self):
        helper = self.root / "build-helper"
        helper.write_text("#!/bin/sh\nexit 0\n")
        helper.chmod(0o755)
        path, recipe = self.recipe()
        recipe["inputs"].append("build-helper")
        recipe["build"].insert(0, ["{root}/build-helper"])
        recipe["mutations"] = recipe["mutations"][:1]
        path.write_text(json.dumps(recipe))
        self.source.chmod(0o444)
        self.cli("mutator", "run", "--project", self.root, "--recipe", path, "--cxx", CLANG,
                 "--output", self.root / "executable-helper")
        self.assertEqual(helper.stat().st_mode & 0o777, 0o755)
        self.assertEqual(self.source.stat().st_mode & 0o777, 0o444)

    def test_mutator_rejects_stale_offsets_and_symlink_inputs(self):
        path, recipe = self.recipe()
        recipe["mutations"] = [{"id": "x", "file": "logic.cpp", "offset": 0,
                                "source_sha256": "stale", "before": "//", "after": "/*"}]
        path.write_text(json.dumps(recipe))
        self.cli("mutator", "run", "--project", self.root, "--recipe", path,
                 "--output", self.root / "stale-offset", expected=2)
        (self.root / "linked.cpp").symlink_to(self.source)
        recipe["inputs"] = ["linked.cpp"]
        path.write_text(json.dumps(recipe))
        self.cli("mutator", "run", "--project", self.root, "--recipe", path,
                 "--output", self.root / "linked", expected=2)

    def test_generated_mutations_can_be_executed_with_a_separate_recipe(self):
        path, _ = self.recipe()
        self.cli("mutator", "discover", "--source", self.source, "--project", self.root,
                 "--clang", CLANG, "--ast-filter", "fixture", "--function", "choose",
                 "--output", self.root / "auto")
        self.cli("mutator", "run", "--project", self.root, "--recipe", path, "--cxx", CLANG,
                 "--mutations", self.root / "auto/report.json", "--output", self.root / "auto-run", expected=1)
        report = json.loads((self.root / "auto-run/report.json").read_text())
        self.assertGreater(report["counts"]["killed"], 0)
        self.assertGreater(report["counts"]["survived"], 0)
        self.assertEqual(report["unresolved"], 0)

    def test_automatic_mutations_have_byte_offsets_hashes_and_no_comments(self):
        self.cli("mutator", "discover", "--source", self.source, "--project", self.root,
                 "--clang", CLANG, "--ast-filter", "fixture", "--function", "choose",
                 "--output", self.root / "generated")
        report = json.loads((self.root / "generated/report.json").read_text())
        data = self.source.read_bytes()
        self.assertGreaterEqual(len(report["mutations"]), 3)
        self.assertTrue(all(row["source_sha256"] == hashlib.sha256(data).hexdigest() for row in report["mutations"]))
        for row in report["mutations"]:
            offset = row["offset"]
            self.assertEqual(data[offset:offset + len(row["before"].encode())], row["before"].encode())

    def test_timeout_kills_descendants(self):
        from common import Session
        session = Session(self.root / "process")
        pid_file = self.root / "child.pid"
        script = ("import subprocess,sys,time; from pathlib import Path; "
                  "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                  f"Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)")
        result = session.run([sys.executable, "-c", script], self.root, "deadline", timeout=0.3)
        self.assertEqual(result["status"], "timeout")
        child = Path("/proc") / pid_file.read_text() / "stat"
        try:
            state = child.read_text().split()[2]
        except FileNotFoundError:
            return  # Reaped between /proc lookup and read is also a successful kill.
        self.assertIn(state, ("Z", "X"))


if __name__ == "__main__":
    unittest.main()
