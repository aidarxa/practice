#pragma once
/** @file
 * @brief Single physical schema for the converted SSB files.
 * @details City/nation/region and manufacturer/category/brand are already Int32
 * codes in these files. Other text remains text, never an invented dictionary ID.
 * DDATE has 17 columns; monthnum/weeknum precede sellingseason and the four flags.
 */
#include "storage/column.h"
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace db {
/** @brief Stable logical name, table and numbered physical file with its encoding. */
struct SsbColumn {
    const char* name;
    const char* table;
    unsigned index;
    ColumnStorage storage;
    std::string fileName() const { return std::string(table) + std::to_string(index); }
};

inline constexpr ColumnStorage ssbInt{PhysicalEncoding::Int32LE, 4};
inline constexpr ColumnStorage ssbFlag{PhysicalEncoding::AsciiBoolean, 1};
inline constexpr std::array<SsbColumn, 58> ssbColumns{{
    {"lo_orderkey", "LINEORDER", 0, ssbInt},
    {"lo_linenumber", "LINEORDER", 1, ssbInt},
    {"lo_custkey", "LINEORDER", 2, ssbInt},
    {"lo_partkey", "LINEORDER", 3, ssbInt},
    {"lo_suppkey", "LINEORDER", 4, ssbInt},
    {"lo_orderdate", "LINEORDER", 5, ssbInt},
    {"lo_orderpriority", "LINEORDER", 6, {PhysicalEncoding::FixedText, 16}},
    {"lo_shippriority", "LINEORDER", 7, ssbFlag},
    {"lo_quantity", "LINEORDER", 8, ssbInt},
    {"lo_extendedprice", "LINEORDER", 9, ssbInt},
    {"lo_ordtotalprice", "LINEORDER", 10, ssbInt},
    {"lo_discount", "LINEORDER", 11, ssbInt},
    {"lo_revenue", "LINEORDER", 12, ssbInt},
    {"lo_supplycost", "LINEORDER", 13, ssbInt},
    {"lo_tax", "LINEORDER", 14, ssbInt},
    {"lo_commitdate", "LINEORDER", 15, ssbInt},
    {"lo_shipmode", "LINEORDER", 16, {PhysicalEncoding::FixedText, 10}},
    {"s_suppkey", "SUPPLIER", 0, ssbInt},
    {"s_name", "SUPPLIER", 1, {PhysicalEncoding::FixedText, 25}},
    {"s_address", "SUPPLIER", 2, {PhysicalEncoding::FixedText, 25}},
    {"s_city", "SUPPLIER", 3, ssbInt},
    {"s_nation", "SUPPLIER", 4, ssbInt},
    {"s_region", "SUPPLIER", 5, ssbInt},
    {"s_phone", "SUPPLIER", 6, {PhysicalEncoding::FixedText, 15}},
    {"c_custkey", "CUSTOMER", 0, ssbInt},
    {"c_name", "CUSTOMER", 1, {PhysicalEncoding::FixedText, 25}},
    {"c_address", "CUSTOMER", 2, {PhysicalEncoding::FixedText, 25}},
    {"c_city", "CUSTOMER", 3, ssbInt},
    {"c_nation", "CUSTOMER", 4, ssbInt},
    {"c_region", "CUSTOMER", 5, ssbInt},
    {"c_phone", "CUSTOMER", 6, {PhysicalEncoding::FixedText, 15}},
    {"c_mktsegment", "CUSTOMER", 7, {PhysicalEncoding::FixedText, 10}},
    {"p_partkey", "PART", 0, ssbInt},
    {"p_name", "PART", 1, {PhysicalEncoding::FixedText, 22}},
    {"p_mfgr", "PART", 2, ssbInt},
    {"p_category", "PART", 3, ssbInt},
    {"p_brand1", "PART", 4, ssbInt},
    {"p_color", "PART", 5, {PhysicalEncoding::FixedText, 11}},
    {"p_type", "PART", 6, {PhysicalEncoding::FixedText, 25}},
    {"p_size", "PART", 7, ssbInt},
    {"p_container", "PART", 8, {PhysicalEncoding::FixedText, 10}},
    {"d_datekey", "DDATE", 0, ssbInt},
    {"d_date", "DDATE", 1, {PhysicalEncoding::FixedText, 18}},
    {"d_dayofweek", "DDATE", 2, {PhysicalEncoding::FixedText, 8}},
    {"d_month", "DDATE", 3, {PhysicalEncoding::FixedText, 9}},
    {"d_year", "DDATE", 4, ssbInt},
    {"d_yearmonthnum", "DDATE", 5, ssbInt},
    {"d_yearmonth", "DDATE", 6, {PhysicalEncoding::FixedText, 7}},
    {"d_daynuminweek", "DDATE", 7, ssbInt},
    {"d_daynuminmonth", "DDATE", 8, ssbInt},
    {"d_daynuminyear", "DDATE", 9, ssbInt},
    {"d_monthnuminyear", "DDATE", 10, ssbInt},
    {"d_weeknuminyear", "DDATE", 11, ssbInt},
    {"d_sellingseason", "DDATE", 12, {PhysicalEncoding::FixedText, 12}},
    {"d_lastdayinweekfl", "DDATE", 13, ssbFlag},
    {"d_lastdayinmonthfl", "DDATE", 14, ssbFlag},
    {"d_holidayfl", "DDATE", 15, ssbFlag},
    {"d_weekdayfl", "DDATE", 16, ssbFlag}
}};

/** @brief Reject unknown names instead of manufacturing a TABLE-1 filename. */
inline const SsbColumn& ssbColumn(std::string_view name) {
    for (const auto& column : ssbColumns) if (name == column.name) return column;
    throw std::invalid_argument("Unknown SSB physical column: " + std::string(name));
}
/** @brief Obtain catalog order from the same physical schema used by the loader. */
inline std::vector<std::string> ssbColumnNames(std::string_view table) {
    std::vector<std::string> names;
    for (const auto& column : ssbColumns) if (table == column.table) names.emplace_back(column.name);
    if (names.empty()) throw std::invalid_argument("Unknown SSB table: " + std::string(table));
    return names;
}
}
