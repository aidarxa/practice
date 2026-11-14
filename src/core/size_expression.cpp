#include "core/size_expression.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace db {
struct SizeExpr::Node {
    Op op;
    uint64_t constant = 0;
    std::string name;
    std::shared_ptr<const Node> left;
    std::shared_ptr<const Node> right;
};

uint64_t checkedSizeAdd(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("runtime size addition overflow");
    return a + b;
}
uint64_t checkedSizeMultiply(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) throw std::overflow_error("runtime size multiplication overflow");
    return a * b;
}

namespace {
uint64_t divideSize(uint64_t a, uint64_t b, bool ceil) {
    if (!b) throw std::runtime_error("runtime size division by zero");
    return a / b + static_cast<uint64_t>(ceil && a % b != 0);
}
uint64_t subtractSize(uint64_t a, uint64_t b) {
    if (b > a) throw std::overflow_error("runtime size subtraction underflow");
    return a - b;
}
uint64_t shiftSize(uint64_t a, uint64_t b, bool left) {
    if (b >= 64) throw std::overflow_error("runtime shift exceeds width");
    if (!left) return a >> b;
    if (a > (std::numeric_limits<uint64_t>::max() >> b)) throw std::overflow_error("runtime left shift overflow");
    return a << b;
}
uint64_t paddedSize(uint64_t value) {
    uint64_t result = 1;
    while (result < value) result = checkedSizeMultiply(result, 2);
    return result;
}
using Op = SizeExpr::Op;
uint64_t arithmetic(Op op, uint64_t a, uint64_t b) {
    switch (op) {
        case Op::Add: return checkedSizeAdd(a, b);
        case Op::Subtract: return subtractSize(a, b);
        case Op::Multiply: return checkedSizeMultiply(a, b);
        case Op::Divide: return divideSize(a, b, false);
        case Op::CeilDivide: return divideSize(a, b, true);
        case Op::Minimum: return std::min(a, b);
        case Op::Maximum: return std::max(a, b);
        default: throw std::logic_error("not a size arithmetic operation");
    }
}
uint64_t predicate(Op op, uint64_t a, uint64_t b) {
    switch (op) {
        case Op::Less: return a < b;
        case Op::LessEqual: return a <= b;
        case Op::Equal: return a == b;
        case Op::And: return a && b;
        case Op::Or: return a || b;
        default: throw std::logic_error("not a size predicate");
    }
}
uint64_t bits(Op op, uint64_t a, uint64_t b) {
    switch (op) {
        case Op::ShiftLeft: return shiftSize(a, b, true);
        case Op::ShiftRight: return shiftSize(a, b, false);
        case Op::BitAnd: return a & b;
        case Op::BitOr: return a | b;
        case Op::NextPowerOfTwo: return paddedSize(a);
        default: throw std::logic_error("not a size bit operation");
    }
}
} // namespace

SizeExpr::SizeExpr(uint64_t value) : node_(std::make_shared<Node>(Node{Op::Constant, value, {}, {}, {}})) {}
SizeExpr::SizeExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
SizeExpr SizeExpr::reference(std::string name) {
    if (name.empty()) throw std::invalid_argument("empty runtime scalar name");
    return SizeExpr(std::make_shared<Node>(Node{Op::Variable, 0, std::move(name), {}, {}}));
}
SizeExpr SizeExpr::operation(Op op, SizeExpr left, SizeExpr right) {
    if (op <= Op::Variable || op > Op::NextPowerOfTwo) throw std::invalid_argument("invalid size operation");
    return SizeExpr(std::make_shared<Node>(Node{op, 0, {}, left.node_, right.node_}));
}
uint64_t SizeExpr::evaluate(const PlanScalars& values) const {
    const auto& n = *node_;
    if (n.op == Op::Constant) return n.constant;
    if (n.op == Op::Variable) return values.at(n.name);
    const auto a = SizeExpr(n.left).evaluate(values);
    if (n.op == Op::And && !a) return 0;
    if (n.op == Op::Or && a) return 1;
    const auto b = SizeExpr(n.right).evaluate(values);
    if (n.op <= Op::Maximum) return arithmetic(n.op, a, b);
    if (n.op <= Op::Or) return predicate(n.op, a, b);
    return bits(n.op, a, b);
}
std::string SizeExpr::describe() const {
    if (node_->op == Op::Constant) return std::to_string(node_->constant);
    if (node_->op == Op::Variable) return "$" + node_->name;
    return "op" + std::to_string(static_cast<int>(node_->op)) + "(" + SizeExpr(node_->left).describe() + "," + SizeExpr(node_->right).describe() + ")";
}
SizeExpr variable(const std::string& name) { return SizeExpr::reference(name); }
SizeExpr operator+(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Add, std::move(a), std::move(b)); }
SizeExpr operator-(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Subtract, std::move(a), std::move(b)); }
SizeExpr operator*(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Multiply, std::move(a), std::move(b)); }
SizeExpr operator/(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Divide, std::move(a), std::move(b)); }
SizeExpr ceilDiv(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::CeilDivide, std::move(a), std::move(b)); }
SizeExpr minimum(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Minimum, std::move(a), std::move(b)); }
SizeExpr maximum(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Maximum, std::move(a), std::move(b)); }
SizeExpr less(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Less, std::move(a), std::move(b)); }
SizeExpr lessEqual(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::LessEqual, std::move(a), std::move(b)); }
SizeExpr equal(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Equal, std::move(a), std::move(b)); }
SizeExpr both(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::And, std::move(a), std::move(b)); }
SizeExpr either(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::Or, std::move(a), std::move(b)); }
SizeExpr shiftLeft(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::ShiftLeft, std::move(a), std::move(b)); }
SizeExpr shiftRight(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::ShiftRight, std::move(a), std::move(b)); }
SizeExpr bitAnd(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::BitAnd, std::move(a), std::move(b)); }
SizeExpr bitOr(SizeExpr a, SizeExpr b) { return SizeExpr::operation(Op::BitOr, std::move(a), std::move(b)); }
SizeExpr nextPowerOfTwo(SizeExpr a) { return SizeExpr::operation(Op::NextPowerOfTwo, std::move(a)); }
} // namespace db
