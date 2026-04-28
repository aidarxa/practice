#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
  };
  TableMetadata(TableMetadata &&other) {
    name_ = std::move(other.name_);
    column_names_ = std::move(other.column_names_);
    size_ = std::move(other.size_);
    is_fact_table_ = std::move(other.is_fact_table_);
  }
  TableMetadata &operator=(TableMetadata &rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      column_names_ = rhs.column_names_;
      size_ = rhs.size_;
      is_fact_table_ = rhs.is_fact_table_;
    }
    return *this;
  }
  TableMetadata &operator=(TableMetadata &&rhs) {
    if (this != &rhs) {
      name_ = std::move(rhs.name_);
      column_names_ = std::move(rhs.column_names_);
      size_ = std::move(rhs.size_);
      is_fact_table_ = std::move(rhs.is_fact_table_);
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

private:
  std::string name_;
  std::vector<std::string> column_names_;
  uint64_t size_;
  bool is_fact_table_;
};

class Catalog {
  Catalog() = default;
  Catalog(std::vector<TableMetadata> tables) : tables_(tables) {};
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