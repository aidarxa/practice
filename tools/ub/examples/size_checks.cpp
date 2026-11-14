// Independent host client of the real checked-size API. No GPU SDK is required.
#include "core/size_expression.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("size check failed");
}

template<class Action>
void requireOverflow(Action action) {
    try { action(); }
    catch (const std::overflow_error&) { return; }
    throw std::runtime_error("expected overflow was not detected");
}

void addition() {
    const auto max = std::numeric_limits<uint64_t>::max();
    require(db::checkedSizeAdd(2, 3) == 5);
    require(db::checkedSizeAdd(max, 0) == max);
    requireOverflow([max] { (void)db::checkedSizeAdd(max, 1); });
}

void multiplication() {
    const auto max = std::numeric_limits<uint64_t>::max();
    require(db::checkedSizeMultiply(5, 7) == 35);
    require(db::checkedSizeMultiply(0, max) == 0);
    require(db::checkedSizeMultiply(max, 1) == max);
    requireOverflow([max] { (void)db::checkedSizeMultiply(max, 2); });
    requireOverflow([max] { (void)db::checkedSizeMultiply(2, max / 2 + 1); });
}

void expressions() {
    const auto max = std::numeric_limits<uint64_t>::max();
    require(db::ceilDiv(max, 2).evaluate({}) == max / 2 + 1);
    require(db::ceilDiv(0, 2).evaluate({}) == 0);
    const auto value = db::variable("rows") + 1;
    require(value.evaluate({{"rows", 10}}) == 11);
    require(value.evaluate({{"rows", 20}}) == 21);
    requireOverflow([] { (void)(db::SizeExpr(0) - 1).evaluate({}); });
}
}

int main() try {
    addition();
    multiplication();
    expressions();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
