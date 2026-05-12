#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ColumnStatistics {
  int64_t min_value_ = 0;    // e.g. 19920101 for d_datekey
  int64_t max_value_ = 0;    // e.g. 19981230 for d_datekey
  uint64_t cardinality_ = 0; // number of unique values (e.g. 25 for s_nation)
  bool is_unique_ = false;
  bool is_primary_key_ = false;
  bool is_nullable_ = false;
};

class TableMetadata {

public:
  TableMetadata(const std::string &name,
                const std::vector<std::string> &column_names, uint64_t size,
                bool is_fact_table)
      : name_(name), column_names_(column_names), size_(size),
        is_fact_table_(is_fact_table) {}

  TableMetadata(const TableMetadata &other) {
    name_ = other.name_;
    column_names_ = other.column_names_;
    size_ = other.size_;
    is_fact_table_ = other.is_fact_table_;
    column_stats_ = other.column_stats_;
  };
  TableMetadata(TableMetadata &&other) {
    name_ = std::move(other.name_);
    column_names_ = std::move(other.column_names_);
    size_ = std::move(other.size_);
    is_fact_table_ = std::move(other.is_fact_table_);
    column_stats_ = std::move(other.column_stats_);
  }
  TableMetadata &operator=(const TableMetadata &rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      column_names_ = rhs.column_names_;
      size_ = rhs.size_;
      is_fact_table_ = rhs.is_fact_table_;
      column_stats_ = rhs.column_stats_;
    }
    return *this;
  }
  TableMetadata &operator=(TableMetadata &&rhs) {
    if (this != &rhs) {
      name_ = std::move(rhs.name_);
      column_names_ = std::move(rhs.column_names_);
      size_ = std::move(rhs.size_);
      is_fact_table_ = std::move(rhs.is_fact_table_);
      column_stats_ = std::move(rhs.column_stats_);
    }
    return *this;
  }
  ~TableMetadata() = default;

  inline const std::string &getName() const { return name_; }
  inline const std::vector<std::string> &getColumnNames() const {
    return column_names_;
  }
  inline uint64_t getSize() const { return size_; }
  inline uint64_t getColumnCount() const { return column_names_.size(); }
  inline bool isFactTable() const { return is_fact_table_; }

  inline void setColumnStats(const std::string &col,
                             const ColumnStatistics &stats) {
    auto &dst = column_stats_[col];
    const bool was_unique = dst.is_unique_;
    const bool was_primary_key = dst.is_primary_key_;
    const bool was_nullable = dst.is_nullable_;
    dst = stats;
    dst.is_unique_ = stats.is_unique_ || was_unique || stats.is_primary_key_ || was_primary_key;
    dst.is_primary_key_ = stats.is_primary_key_ || was_primary_key;
    dst.is_nullable_ = stats.is_nullable_ || was_nullable;
  }
  inline const ColumnStatistics &
  getColumnStats(const std::string &col) const {
    auto it = column_stats_.find(col);
    if (it == column_stats_.end())
      throw std::runtime_error("Column stats not found: " + col);
    return it->second;
  }
  inline bool hasColumnStats(const std::string &col) const {
    return column_stats_.count(col) > 0;
  }
  inline const std::unordered_map<std::string, ColumnStatistics> &
  getAllColumnStats() const {
    return column_stats_;
  }

  inline void setColumnUnique(const std::string &col, bool unique = true) {
    auto &stats = column_stats_[col];
    stats.is_unique_ = unique;
    if (!unique) stats.is_primary_key_ = false;
  }

  inline void setColumnPrimaryKey(const std::string &col, bool primary_key = true) {
    auto &stats = column_stats_[col];
    stats.is_primary_key_ = primary_key;
    if (primary_key) stats.is_unique_ = true;
  }

  inline void setColumnNullable(const std::string &col, bool nullable = true) {
    column_stats_[col].is_nullable_ = nullable;
  }

  inline bool isColumnUnique(const std::string &col) const {
    auto it = column_stats_.find(col);
    return it != column_stats_.end() && it->second.is_unique_;
  }

  inline bool isColumnPrimaryKey(const std::string &col) const {
    auto it = column_stats_.find(col);
    return it != column_stats_.end() && it->second.is_primary_key_;
  }

  inline bool isColumnNullable(const std::string &col) const {
    auto it = column_stats_.find(col);
    return it != column_stats_.end() && it->second.is_nullable_;
  }

private:
  std::string name_;
  std::vector<std::string> column_names_;
  uint64_t size_;
  bool is_fact_table_;
  std::unordered_map<std::string, ColumnStatistics> column_stats_;
};

class Catalog {
public:
  Catalog() = default;
  Catalog(std::vector<TableMetadata> tables) : tables_(tables){};
  Catalog(const Catalog &other) { tables_ = other.tables_; }
  Catalog(Catalog &&other) { tables_ = std::move(other.tables_); }
  Catalog &operator=(Catalog &rhs) {
    if (this != &rhs) {
      tables_ = rhs.tables_;
    }
    return *this;
  }
  Catalog &operator=(Catalog &&rhs) {
    if (this != &rhs) {
      tables_ = std::move(rhs.tables_);
    }
    return *this;
  }
  ~Catalog() = default;

  inline void pushTableMetadata(const TableMetadata &tb) {
    tables_.push_back(tb);
  }
  inline void pushTableMetadata(TableMetadata &&tb) {
    tables_.push_back(std::move(tb));
  }
  template <typename... Args> inline void emplaceTableMetadata(Args &&...args) {
    tables_.emplace_back(std::forward<Args>(args)...);
  }
  inline const std::vector<TableMetadata> &getTablesMetadata() const {
    return tables_;
  }
  inline uint64_t getTablesMetadataCount() const { return tables_.size(); }
  inline const TableMetadata &getTableMetadata(const std::string &name) const {
    for (const auto &tb : tables_) {
      if (tb.getName() == name)
        return tb;
    }
    throw std::runtime_error("Table not found: " + name);
  }

private:
  std::vector<TableMetadata> tables_;
};