#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define BASE_PATH "/home/aidar/practice/data/" // edit base path to data
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


inline int index_of(std::string *arr, int len, std::string val) {
  for (int i = 0; i < len; i++)
    if (arr[i] == val)
      return i;

  return -1;
}

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
  std::string lineorder[] = {
      "lo_orderkey", "lo_linenumber",    "lo_custkey",       "lo_partkey",
      "lo_suppkey",  "lo_orderdate",     "lo_orderpriority", "lo_shippriority",
      "lo_quantity", "lo_extendedprice", "lo_ordtotalprice", "lo_discount",
      "lo_revenue",  "lo_supplycost",    "lo_tax",           "lo_commitdate",
      "lo_shipmode"};
  std::string part[] = {"p_partkey",  "p_name",   "p_mfgr",
                        "p_category", "p_brand1", "p_color",
                        "p_type",     "p_size",   "p_container"};
  std::string supplier[] = {"s_suppkey", "s_name",   "s_address", "s_city",
                            "s_nation",  "s_region", "s_phone"};
  std::string customer[] = {"c_custkey", "c_name",      "c_address",
                            "c_city",    "c_nation",    "c_region",
                            "c_phone",   "c_mktsegment"};
  std::string date[] = {"d_datekey",
                        "d_date",
                        "d_dayofweek",
                        "d_month",
                        "d_year",
                        "d_yearmonthnum",
                        "d_yearmonth",
                        "d_daynuminweek",
                        "d_daynuminmonth",
                        "d_daynuminyear",
                        "d_sellingseason",
                        "d_lastdayinweekfl",
                        "d_lastdayinmonthfl",
                        "d_holidayfl",
                        "d_weekdayfl"};

  if (col_name[0] == 'l') {
    int index = index_of(lineorder, 17, col_name);
    return "LINEORDER" + std::to_string(index);
  } else if (col_name[0] == 's') {
    int index = index_of(supplier, 7, col_name);
    return "SUPPLIER" + std::to_string(index);
  } else if (col_name[0] == 'c') {
    int index = index_of(customer, 8, col_name);
    return "CUSTOMER" + std::to_string(index);
  } else if (col_name[0] == 'p') {
    int index = index_of(part, 9, col_name);
    return "PART" + std::to_string(index);
  } else if (col_name[0] == 'd') {
    int index = index_of(date, 15, col_name);
    return "DDATE" + std::to_string(index);
  }

  return "";
}

inline std::size_t nullBitmapWordCount(std::size_t num_entries) {
  return (num_entries + 63U) / 64U;
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

enum class NullBitmapFileKind {
  ValidBits64,
  NullBits64,
  ValidBytes,
  NullBytes,
};

inline std::vector<std::pair<std::string, NullBitmapFileKind>> nullBitmapCandidateFiles(
    const std::string& col_name) {
  const std::string physical = std::string(DATA_DIR) + lookup(col_name);
  const std::string logical = std::string(DATA_DIR) + col_name;

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

inline bool hasNullBitmapFile(const std::string& col_name) {
  for (const auto& [path, kind] : nullBitmapCandidateFiles(col_name)) {
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
      if (kind == NullBitmapFileKind::NullBits64) {
        for (auto& word : validity) word = ~word;
      }
      maskUnusedValidityBits(validity, num_entries);
      return validity;
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
      for (std::size_t row = 0; row < num_entries; ++row) {
        const bool bit = bytes[row] != 0;
        const bool valid = (kind == NullBitmapFileKind::ValidBytes) ? bit : !bit;
        if (valid) {
          validity[row >> 6U] |= (1ULL << (row & 63U));
        }
      }
      maskUnusedValidityBits(validity, num_entries);
      return validity;
    }
  }

  return {};
}


template <typename T>
inline std::vector<T> loadColumn(std::string col_name, int num_entries) {
  std::vector<T> ret(num_entries);

  std::string filename = DATA_DIR + lookup(col_name);
  std::ifstream colData(filename.c_str(), std::ios::in | std::ios::binary);
  if (!colData) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  colData.read((char *)ret.data(), num_entries * sizeof(T));
  return ret;
}