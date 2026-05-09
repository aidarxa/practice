// src/core/optimizer.cpp
//
// LEGACY: The old Planner / QueryOptimizer / CodeGenerator implementation has
// been removed as part of the Part 3 architectural refactoring.
//
// The new pipeline lives in:
//   src/core/translator.cpp       — QueryTranslator (AST → OperatorTree)
//   src/core/optimizer_rules.cpp  — Optimizer / PredicatePushdownRule
//   src/core/visitor.cpp          — JITOperatorVisitor (OperatorTree → C++ JIT code)
