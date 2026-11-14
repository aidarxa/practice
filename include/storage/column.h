#pragma once
/** @file @brief Physical encoding is independent of the numeric SQL type system. */
#include <cstddef>

namespace db {
/** @brief On-disk SSB encodings; byte flags are ASCII '0'/'1', not packed Int32. */
enum class PhysicalEncoding { Int32LE, AsciiBoolean, FixedText };
/** @brief Describe a complete physical row value, including unsupported text. */
struct ColumnStorage {
    PhysicalEncoding encoding = PhysicalEncoding::Int32LE;
    std::size_t width = 4;
    bool isNumeric() const { return encoding != PhysicalEncoding::FixedText; }
};
}
