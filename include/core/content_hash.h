#pragma once
/** @file @brief SHA-256 identities for module source and prebuilt toolchain inputs. */
#include <filesystem>
#include <string>
#include <string_view>
namespace db {
/** @brief Hash exact bytes; cache lookup still compares the full structural key. */
std::string contentHash(std::string_view bytes);
/** @brief Hash a binary input; missing/unreadable files are errors, never cache hits. */
std::string fileContentHash(const std::filesystem::path& path);
}
