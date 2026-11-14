#pragma once
/** @file @brief Lossless presentation of UTF-8 Text in the pipe-delimited terminal result. */
#include <string>
#include <string_view>

namespace db { namespace cli {
/** @brief Quote Text as a JSON string; preserve Unicode/spaces and distinguish it from SQL NULL. */
inline std::string quoteText(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') { result += '\\'; result += static_cast<char>(byte); }
        else if (byte < 0x20) {
            result += "\\u00";
            result += hex[byte >> 4]; result += hex[byte & 15];
        } else result += static_cast<char>(byte);
    }
    result += '"';
    return result;
}
}} // namespace db::cli
