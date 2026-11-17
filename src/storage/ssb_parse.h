#pragma once
/** @file @brief Shared strict integer grammar for raw SSB fields and native row counts. */
#include <charconv>
#include <stdexcept>
#include <string_view>

namespace db { namespace ssb_detail {
/** @brief Preserve the caller's exact integer domain; reject overflow, signs on unsigned and trailing bytes. */
template<class Integer> Integer parseInteger(std::string_view value) {
    Integer result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::runtime_error("Invalid SSB integer field or row count");
    return result;
}
}} // namespace db::ssb_detail
