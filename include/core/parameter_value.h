#pragma once
/** @file
 * @brief Exact host parameter values shared by frontend and runtime, without SQL/SDK dependencies.
 * @details Stage 2 extracts the stage-1 value model unchanged so the physical
 * interpreter need not include an operator tree, parser or storage I/O helpers.
 */
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <string>

namespace db {
/** @brief Agreed stage-0 parameter types; validity is not another type. */
enum class ParameterType { NullOnly, Int64, Float64, Text };
/** @brief One value-free coercion rule; Text never implicitly becomes a number. */
ParameterType commonParameterType(ParameterType left, ParameterType right);
/** @brief Exact numeric alternative or typed/untyped NULL; no implicit double conversion. */
class ParameterValue {
public:
    /** @brief Construct an exact signed integer. */
    static ParameterValue integer(int64_t value);
    /** @brief Preserve an input double as Float64. */
    static ParameterValue floating(double value);
    /** @brief Own a validated UTF-8 parameter; no dictionary or SQL-source embedding. */
    static ParameterValue text(std::string value);
    /** @brief Construct NULL without discarding a known slot type. */
    static ParameterValue null(ParameterType type = ParameterType::NullOnly);
    /** @brief Read the numeric alternative independently of validity. */
    ParameterType type() const;
    /** @brief Test SQL invalidity for all alternatives. */
    bool isNull() const;
    /** @brief Read a valid Int64; reject NULL and other types. */
    int64_t asInt64() const;
    /** @brief Read a valid Float64; reject NULL and other types. */
    double asFloat64() const;
    /** @brief Borrow valid Text until this owning ParameterValue is destroyed. */
    const std::string& asText() const;
private:
    using Storage = std::variant<std::monostate, std::optional<int64_t>, std::optional<double>, std::optional<std::string>>;
    explicit ParameterValue(Storage value) : value_(value) {}
    Storage value_;
};
using ParameterSchema = std::vector<ParameterType>;
using ParameterValues = std::vector<ParameterValue>;
} // namespace db
