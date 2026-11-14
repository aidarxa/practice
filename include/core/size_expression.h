#pragma once
/** @file
 * @brief Checked, unsigned runtime size expressions; never used for signed SQL arithmetic.
 */
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace db {
using PlanScalars = std::unordered_map<std::string, uint64_t>;

/**
 * @brief Immutable expression tree for dimensions, offsets, capacities and control flow.
 * @details Ordinary SQL arithmetic remains typed expression code. These expressions
 * operate only on nonnegative sizes/counters and reject overflow/underflow rather
 * than wrapping. An execution owns its scalar environment, not the shared plan.
 */
class SizeExpr {
public:
    /** @brief Operations are grouped by purpose to keep evaluation independently testable. */
    enum class Op { Constant, Variable, Add, Subtract, Multiply, Divide, CeilDivide,
                    Minimum, Maximum, Less, LessEqual, Equal, And, Or, ShiftLeft,
                    ShiftRight, BitAnd, BitOr, NextPowerOfTwo };
    /** @brief Construct a nonnegative constant, including the neutral default zero. */
    SizeExpr(uint64_t value = 0);
    /** @brief Create a checked runtime reference. Unknown names are errors. */
    static SizeExpr reference(std::string name);
    /** @brief Construct an operation over already valid expression trees. */
    static SizeExpr operation(Op op, SizeExpr left, SizeExpr right = 0);
    /** @brief Evaluate against one execution's environment. */
    uint64_t evaluate(const PlanScalars& values) const;
    /** @brief Human-readable expression for diagnostics/plan inspection, not executable C++. */
    std::string describe() const;
private:
    struct Node;
    explicit SizeExpr(std::shared_ptr<const Node> node);
    std::shared_ptr<const Node> node_;
};
/** @brief Named scalar reference for the physical-plan DSL. */
SizeExpr variable(const std::string& name);
/** @brief Checked arithmetic constructors; no unsigned wrapping. */
SizeExpr operator+(SizeExpr a, SizeExpr b);
SizeExpr operator-(SizeExpr a, SizeExpr b);
SizeExpr operator*(SizeExpr a, SizeExpr b);
SizeExpr operator/(SizeExpr a, SizeExpr b);
/** @brief Overflow-safe ceil division, including a zero numerator. */
SizeExpr ceilDiv(SizeExpr a, SizeExpr b);
/** @brief Construct bounded capacity/selection expressions. */
SizeExpr minimum(SizeExpr a, SizeExpr b);
SizeExpr maximum(SizeExpr a, SizeExpr b);
/** @brief Construct predicates returning zero or one. */
SizeExpr less(SizeExpr a, SizeExpr b);
SizeExpr lessEqual(SizeExpr a, SizeExpr b);
SizeExpr equal(SizeExpr a, SizeExpr b);
SizeExpr both(SizeExpr a, SizeExpr b);
SizeExpr either(SizeExpr a, SizeExpr b);
/** @brief Construct checked bit operations used by sorting/radix control flow. */
SizeExpr shiftLeft(SizeExpr a, SizeExpr b);
SizeExpr shiftRight(SizeExpr a, SizeExpr b);
SizeExpr bitAnd(SizeExpr a, SizeExpr b);
SizeExpr bitOr(SizeExpr a, SizeExpr b);
SizeExpr nextPowerOfTwo(SizeExpr a);
/** @brief Shared checked primitives for runtime memory accounting. */
uint64_t checkedSizeAdd(uint64_t a, uint64_t b);
uint64_t checkedSizeMultiply(uint64_t a, uint64_t b);
} // namespace db
