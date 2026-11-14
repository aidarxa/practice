#pragma once
/** @file @brief Host-only SSB numeric/validity loading; device headers stay separate. */
#include "storage/ssb_schema.h"
#include "storage/column_loader.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#ifndef BASE_PATH
#define BASE_PATH "data/"
#endif
#define SF 10
#if SF == 10
#define DATA_DIR BASE_PATH "10/"
#define LO_LEN 59986217
#define P_LEN 800000
#define S_LEN 20000
#define C_LEN 300000
#define D_LEN 2557
#else // 20
#define DATA_DIR BASE_PATH "20/"
#define LO_LEN 119994746
#define P_LEN 1000000
#define S_LEN 40000
#define C_LEN 600000
#define D_LEN 2556
#endif

inline std::string getTableName(std::string col_name) {
  if (col_name.empty()) return "";
  if(col_name[0] == 'l') return "LINEORDER";
  if(col_name[0] == 's') return "SUPPLIER";
  if(col_name[0] == 'c') return "CUSTOMER";
  if(col_name[0] == 'p') return "PART";
  if(col_name[0] == 'd') return "DDATE";
  return "";
}

inline std::string lookup(std::string col_name) {
  return db::ssbColumn(col_name).fileName();
}

inline std::size_t nullBitmapWordCount(std::size_t num_entries) {
  return num_entries / 64U + (num_entries % 64U != 0);
}

inline uint64_t nullBitmapLastWordMask(std::size_t num_entries) {
  const std::size_t rem = num_entries & 63U;
  return rem == 0U ? ~0ULL : ((1ULL << rem) - 1ULL);
}

inline void maskUnusedValidityBits(std::vector<uint64_t>& bitmap, std::size_t num_entries) {
  if (!bitmap.empty()) {
    bitmap.back() &= nullBitmapLastWordMask(num_entries);
  }
}


inline std::vector<uint64_t> decodeValidityWordBitmap(std::vector<uint64_t> words,
                                                      std::size_t num_entries,
                                                      bool input_bits_are_valid) {
  const std::size_t word_count = nullBitmapWordCount(num_entries);
  words.resize(word_count, 0ULL);
  if (!input_bits_are_valid) {
    for (auto& word : words) word = ~word;
  }
  maskUnusedValidityBits(words, num_entries);
  return words;
}

inline std::vector<uint64_t> decodeValidityByteBitmap(const std::vector<unsigned char>& bytes,
                                                      std::size_t num_entries,
                                                      bool input_bytes_are_valid) {
  if (bytes.size() != num_entries) {
    throw std::runtime_error("decodeValidityByteBitmap: byte count does not match row count");
  }
  std::vector<uint64_t> validity(nullBitmapWordCount(num_entries), 0ULL);
  for (std::size_t row = 0; row < num_entries; ++row) {
    const bool bit = bytes[row] != 0;
    const bool valid = input_bytes_are_valid ? bit : !bit;
    if (valid) validity[row >> 6U] |= (1ULL << (row & 63U));
  }
  maskUnusedValidityBits(validity, num_entries);
  return validity;
}

enum class NullBitmapFileKind {
  ValidBits64,
  NullBits64,
  ValidBytes,
  NullBytes,
};

inline std::vector<std::pair<std::string, NullBitmapFileKind>> nullBitmapCandidateFiles(
    const std::string& col_name, const std::filesystem::path& directory = DATA_DIR) {
  const std::string physical = (directory / lookup(col_name)).string();
  const std::string logical = (directory / col_name).string();

  std::vector<std::pair<std::string, NullBitmapFileKind>> candidates;
  auto add_all = [&](const std::string& base) {
    candidates.emplace_back(base + ".valid64", NullBitmapFileKind::ValidBits64);
    candidates.emplace_back(base + ".valid", NullBitmapFileKind::ValidBits64);
    candidates.emplace_back(base + ".validity", NullBitmapFileKind::ValidBits64);
    candidates.emplace_back(base + ".valid_bitmap", NullBitmapFileKind::ValidBits64);
    candidates.emplace_back(base + ".valid8", NullBitmapFileKind::ValidBytes);

    candidates.emplace_back(base + ".null64", NullBitmapFileKind::NullBits64);
    candidates.emplace_back(base + ".null", NullBitmapFileKind::NullBits64);
    candidates.emplace_back(base + ".nulls", NullBitmapFileKind::NullBits64);
    candidates.emplace_back(base + ".null_bitmap", NullBitmapFileKind::NullBits64);
    candidates.emplace_back(base + ".null8", NullBitmapFileKind::NullBytes);
  };

  add_all(physical);
  if (logical != physical) {
    add_all(logical);
  }
  return candidates;
}

inline bool hasNullBitmapFile(const std::string& col_name, const std::filesystem::path& directory = DATA_DIR) {
  for (const auto& [path, kind] : nullBitmapCandidateFiles(col_name, directory)) {
    (void)kind;
    if (std::filesystem::exists(std::filesystem::path(path))) {
      return true;
    }
  }
  return false;
}

inline std::vector<uint64_t> loadValidityBitmapIfExists(const std::string& col_name,
                                                        std::size_t num_entries) {
  const std::size_t word_count = nullBitmapWordCount(num_entries);
  if (word_count == 0U) return {};

  for (const auto& [path, kind] : nullBitmapCandidateFiles(col_name)) {
    std::filesystem::path fs_path(path);
    if (!std::filesystem::exists(fs_path)) {
      continue;
    }

    const std::uintmax_t size = std::filesystem::file_size(fs_path);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Cannot open null bitmap file: " + path);
    }

    std::vector<uint64_t> validity(word_count, 0ULL);

    if (kind == NullBitmapFileKind::ValidBits64 ||
        kind == NullBitmapFileKind::NullBits64) {
      const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(word_count) * sizeof(uint64_t);
      if (size != expected_bytes) {
        throw std::runtime_error(
            "Invalid null bitmap size for " + col_name + " at " + path +
            ": expected " + std::to_string(expected_bytes) +
            " bytes for uint64 bitset, got " + std::to_string(size));
      }
      in.read(reinterpret_cast<char*>(validity.data()), static_cast<std::streamsize>(expected_bytes));
      if (!in) {
        throw std::runtime_error("Failed to read null bitmap file: " + path);
      }
      return decodeValidityWordBitmap(std::move(validity), num_entries, kind == NullBitmapFileKind::ValidBits64);
    }

    if (kind == NullBitmapFileKind::ValidBytes ||
        kind == NullBitmapFileKind::NullBytes) {
      if (size != static_cast<std::uintmax_t>(num_entries)) {
        throw std::runtime_error(
            "Invalid null bitmap size for " + col_name + " at " + path +
            ": expected " + std::to_string(num_entries) +
            " bytes for byte bitmap, got " + std::to_string(size));
      }
      std::vector<unsigned char> bytes(num_entries);
      in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(num_entries));
      if (!in) {
        throw std::runtime_error("Failed to read null bitmap file: " + path);
      }
      return decodeValidityByteBitmap(bytes, num_entries, kind == NullBitmapFileKind::ValidBytes);
    }
  }

  return {};
}


/** @brief Resolve the physical name and strictly decode numeric rows, including ASCII flags.
 * @details The old raw-prefix helper is isolated in the historical stage-3 test.
 * Engine and manual queries use this strict entry, never its raw view.
 */
template <typename T>
inline std::vector<T> loadColumn(std::string col_name, std::size_t num_entries) {
  static_assert(std::is_same<T, int32_t>::value, "SSB numeric storage decodes to Int32");
  const auto& column = db::ssbColumn(col_name);
  return db::loadNumericColumn(std::filesystem::path(DATA_DIR) / column.fileName(), num_entries, column.storage);
}
