## @file
## @brief Tests of the offline quality analogue, separate from engine correctness.
import unittest

from quality import complexity, coverage_identities, functions, json_stream, tokens


class QualityTests(unittest.TestCase):
    def test_concatenated_ast_objects(self):
        self.assertEqual(list(json_stream(' {"kind":"A"}\n{"kind":"B"} ')),
                         [{"kind": "A"}, {"kind": "B"}])

    def test_short_circuit_and_switch_decisions(self):
        tree = {"kind": "CompoundStmt", "inner": [
            {"kind": "IfStmt", "inner": [{"kind": "BinaryOperator", "opcode": "&&"}]},
            {"kind": "CaseStmt"}, {"kind": "CXXForRangeStmt"}]}
        self.assertEqual(1 + complexity(tree), 5)

    def test_lambda_is_not_double_counted(self):
        self.assertEqual(complexity({"kind": "LambdaExpr", "inner": [{"kind": "IfStmt"}]}), 0)

    def test_ast_reference_is_not_a_statement(self):
        self.assertEqual(complexity({"kind": "CompoundStmt", "inner": [{"id": "reference"}, {"kind": "IfStmt"}]}), 1)

    def test_destructor_is_a_measured_function(self):
        destructor = {"kind": "CXXDestructorDecl", "name": "~Owner", "mangledName": "owner-destructor",
                      "inner": [{"kind": "CompoundStmt"}]}
        self.assertEqual(len(list(functions(destructor))), 1)

    def test_itanium_constructor_alias_is_exact(self):
        constructor = {"kind": "CXXConstructorDecl", "mangledName": "_ZN2db5OwnerC1Ev"}
        self.assertIn("_ZN2db5OwnerC2Ev", coverage_identities(constructor))
        constructor["kind"] = "FunctionDecl"
        self.assertEqual(coverage_identities(constructor), {"_ZN2db5OwnerC1Ev"})

    def test_tokens_keep_identity_not_comments(self):
        self.assertEqual(tokens("a /* ignored */ + b // ignored\n"), ["a", "+", "b"])
        self.assertNotEqual(tokens("slot(0)"), tokens("slot(1)"))
        self.assertEqual(tokens('"/* not a comment */"'), ['"/* not a comment */"'])


if __name__ == "__main__":
    unittest.main()
