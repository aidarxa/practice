#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
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