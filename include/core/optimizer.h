#pragma once

// ============================================================================
// LEGACY: This file previously contained the old monolithic pipeline:
//   Planner, QueryOptimizer, CodeGenerator, LogicalPlan, PhysicalPlan,
//   Kernel, DeviceBuffer, OpBlockLoad, OpBlockFilter, etc.
//
// All of these have been removed in Part 3 of the architectural refactoring.
//
// The new pipeline:
//   db::QueryTranslator    (core/translator.h)
//   db::Optimizer          (core/optimizer_rules.h)
//   db::JITOperatorVisitor (core/visitor.h)
// ============================================================================