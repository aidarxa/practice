#pragma once
/** @file @brief Identifier operations on generated C++ (ordinary escaped string/character literals). */
#include <map>
#include <set>
#include <string>
namespace db {
/** @brief Discover actual symbols, excluding comments and quoted diagnostics. */
std::set<std::string> codeIdentifiers(const std::string& code);
/** @brief Canonicalize whole symbols; never alter quoted bytes or comments. */
std::string replaceCodeIdentifiers(const std::string& code, const std::map<std::string,std::string>& replacements);
}
