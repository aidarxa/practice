#include "core/parameter_value.h"
#include "storage/text_column.h"
#include <stdexcept>
#include <string>

namespace db {
namespace {
/** @brief One checked access rule for both exact numeric alternatives. */
template <typename T>
T numericValue(const std::optional<T>* value, const char* type) {
    if (!value) throw std::runtime_error(std::string("Parameter type is not ") + type);
    if (!value->has_value()) throw std::runtime_error("Cannot read a NULL parameter");
    return **value;
}
} // namespace
ParameterValue ParameterValue::integer(int64_t value) {
    return ParameterValue(std::optional<int64_t>{value});
}
ParameterValue ParameterValue::floating(double value) {
    return ParameterValue(std::optional<double>{value});
}
ParameterValue ParameterValue::text(std::string value) {
    validateText(value);
    return ParameterValue(std::optional<std::string>{std::move(value)});
}
ParameterType commonParameterType(ParameterType left, ParameterType right) {
    if (left == ParameterType::NullOnly) return right;
    if (right == ParameterType::NullOnly || left == right) return left;
    if (left == ParameterType::Text || right == ParameterType::Text)
        throw std::runtime_error("Text and numeric types cannot be mixed implicitly");
    return ParameterType::Float64;
}
ParameterValue ParameterValue::null(ParameterType type) {
    switch (type) {
        case ParameterType::NullOnly: return ParameterValue(std::monostate{});
        case ParameterType::Int64: return ParameterValue(std::optional<int64_t>{});
        case ParameterType::Float64: return ParameterValue(std::optional<double>{});
        case ParameterType::Text: return ParameterValue(std::optional<std::string>{});
    }
    throw std::runtime_error("Unknown parameter type");
}
ParameterType ParameterValue::type() const {
    if (std::holds_alternative<std::optional<int64_t>>(value_)) return ParameterType::Int64;
    if (std::holds_alternative<std::optional<double>>(value_)) return ParameterType::Float64;
    if (std::holds_alternative<std::optional<std::string>>(value_)) return ParameterType::Text;
    return ParameterType::NullOnly;
}
bool ParameterValue::isNull() const {
    if (const auto* integer = std::get_if<std::optional<int64_t>>(&value_)) return !integer->has_value();
    if (const auto* floating = std::get_if<std::optional<double>>(&value_)) return !floating->has_value();
    if (const auto* text = std::get_if<std::optional<std::string>>(&value_)) return !text->has_value();
    return true;
}
int64_t ParameterValue::asInt64() const {
    return numericValue(std::get_if<std::optional<int64_t>>(&value_), "Int64");
}
double ParameterValue::asFloat64() const {
    return numericValue(std::get_if<std::optional<double>>(&value_), "Float64");
}
const std::string& ParameterValue::asText() const {
    const auto* value = std::get_if<std::optional<std::string>>(&value_);
    if (!value || !*value) throw std::runtime_error("Parameter is not valid Text");
    return **value;
}
} // namespace db
