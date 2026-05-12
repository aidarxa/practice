#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "core/memory.h"
#include "core/device_buffer.h"

namespace db {

enum class LogicalType : uint8_t {
    Int64,
    UInt64,
    Float64,
};

struct ResultColumnDesc {
    LogicalType type = LogicalType::UInt64;
    uint32_t scale = 0;
    bool nullable = false;
};

template <typename T>
struct ColumnView {
    T* data = nullptr;
    uint64_t* null_bitmap = nullptr;
    bool nullable = false;
};

struct QueryResult {
    // Legacy row-wise host buffer. Kept for compatibility with sparse aggregate
    // internals and existing tests. The external host result is now also exposed
    // in columnar form below.
    std::vector<unsigned long long> data;
    std::vector<uint64_t> cell_validity_bitmap; // row-wise bitset: 1 = valid, 0 = NULL
    bool has_cell_validity = false;

    // Typed columnar host result. column_data[c][r] stores column c, row r.
    // column_validity_bitmap[c] uses one bit per row: 1 = valid, 0 = NULL.
    std::vector<std::vector<unsigned long long>> column_data;
    std::vector<std::vector<uint64_t>> column_validity_bitmap;
    bool has_columnar_result = false;

    size_t tuple_size = 1;
    std::vector<ResultColumnDesc> columns;

    // Number of logical rows returned by the query. This is not necessarily
    // data.size() / tuple_size for sparse hash-aggregate result buffers where
    // empty groups are represented by all-zero tuples.
    size_t row_count = 0;

    // Dense results, such as projection materialization, contain only returned
    // rows. Sparse results, such as fixed-domain grouped aggregates, may contain
    // empty buckets that must be skipped by the CLI.
    bool dense_result = false;
};

struct ExecutionContext {
    sycl::queue* q_ = nullptr;
    std::unique_ptr<DynamicDeviceBuffer<unsigned long long>> result_buffer_;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> result_validity_buffer_;
    std::unordered_map<std::string, void*> buffers_;
    std::unordered_map<std::string, uint64_t*> null_bitmaps_;
    size_t tuple_size_ = 1;
    std::vector<ResultColumnDesc> result_columns_;
    size_t result_row_count_{0};
    bool result_is_dense_{false};
    bool result_is_columnar_{false};
    size_t result_column_count_{0};
    size_t result_column_row_capacity_{0};
    std::vector<std::unique_ptr<DynamicDeviceBuffer<unsigned long long>>> result_column_buffers_;
    std::vector<std::unique_ptr<DynamicDeviceBuffer<uint64_t>>> result_column_validity_buffers_;
    size_t expected_result_validity_words_{0};

    // Размер буфера результатов в элементах (устанавливается QueryEngine перед выполнением).
    // DatabaseInstance использует это значение для copyToHost вместо хардкода 21000.
    size_t expected_result_size_{0};

    // Bytes permanently allocated for loaded base-table columns.
    // QueryEngine uses this for conservative preflight memory checks before
    // allocating large result buffers or launching JIT kernels with temporary
    // hash tables.
    size_t loaded_device_bytes_{0};

    template<typename T>
    T* getBuffer(const std::string& name) {
        auto it = buffers_.find(name);
        if (it != buffers_.end()) {
            return static_cast<T*>(it->second);
        }
        return nullptr;
    }


    uint64_t* getNullBitmap(const std::string& name) {
        auto it = null_bitmaps_.find(name);
        if (it == null_bitmaps_.end()) return nullptr;
        return it->second;
    }

    template<typename T>
    ColumnView<T> getColumnView(const std::string& name) {
        return ColumnView<T>{getBuffer<T>(name), getNullBitmap(name), getNullBitmap(name) != nullptr};
    }

    unsigned long long* getResultPointer() const {
        return result_buffer_ ? result_buffer_->data() : nullptr;
    }

    uint64_t* getResultValidityPointer() const {
        return result_validity_buffer_ ? result_validity_buffer_->data() : nullptr;
    }

    void ensureResultValidityCapacity(size_t value_count) {
        expected_result_validity_words_ = (value_count + 63ULL) / 64ULL;
        if (result_validity_buffer_) {
            result_validity_buffer_->ensureCapacity(expected_result_validity_words_ == 0 ? 1 : expected_result_validity_words_);
            result_validity_buffer_->zero();
        }
    }

    void ensureColumnarResultCapacity(size_t column_count, size_t row_count) {
        result_is_columnar_ = true;
        result_column_count_ = column_count;
        result_column_row_capacity_ = row_count;
        result_column_buffers_.resize(column_count);
        result_column_validity_buffers_.resize(column_count);
        const size_t value_capacity = row_count == 0 ? 1 : row_count;
        const size_t validity_words = ((row_count + 63ULL) / 64ULL) == 0 ? 1 : ((row_count + 63ULL) / 64ULL);
        for (size_t col = 0; col < column_count; ++col) {
            if (!result_column_buffers_[col]) {
                result_column_buffers_[col] = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(*q_, 0);
            }
            result_column_buffers_[col]->ensureCapacity(value_capacity);
            result_column_buffers_[col]->zero();
            if (!result_column_validity_buffers_[col]) {
                result_column_validity_buffers_[col] = std::make_unique<DynamicDeviceBuffer<uint64_t>>(*q_, 0);
            }
            result_column_validity_buffers_[col]->ensureCapacity(validity_words);
            result_column_validity_buffers_[col]->zero();
        }
    }

    unsigned long long* getResultColumnPointer(size_t col) const {
        if (col >= result_column_buffers_.size() || !result_column_buffers_[col]) {
            throw std::out_of_range("result column buffer is not allocated");
        }
        return result_column_buffers_[col]->data();
    }

    uint64_t* getResultColumnValidityPointer(size_t col) const {
        if (col >= result_column_validity_buffers_.size() || !result_column_validity_buffers_[col]) {
            throw std::out_of_range("result column validity buffer is not allocated");
        }
        return result_column_validity_buffers_[col]->data();
    }

    void resetResultShapeFlags() {
        result_row_count_ = 0;
        result_is_dense_ = false;
        result_is_columnar_ = false;
        result_column_count_ = 0;
        result_column_row_capacity_ = 0;
    }
};

class IQueryCache {
public:
    virtual ~IQueryCache() = default;
    virtual std::optional<std::string> get(const std::string& query_hash) = 0;
    virtual void put(const std::string& query_hash, const std::string& lib_path) = 0;
};

class ICompiler {
public:
    virtual ~ICompiler() = default;
    virtual std::string compile(const std::string& source_code, const std::string& query_hash) = 0;
};

class IExecutor {
public:
    virtual ~IExecutor() = default;
    virtual void execute(const std::string& lib_path, ExecutionContext* ctx) = 0;
};

class FileBasedQueryCache : public IQueryCache {
public:
    std::optional<std::string> get(const std::string& query_hash) override;
    void put(const std::string& query_hash, const std::string& lib_path) override;
private:
    std::unordered_map<std::string, std::string> cache_;
};

class AdaptiveCppCompiler : public ICompiler {
public:
    AdaptiveCppCompiler(std::string include_dir = {},
                        std::string deps_include_dir = {});
    std::string compile(const std::string& source_code, const std::string& query_hash) override;
private:
    std::string include_dir_;
    std::string deps_include_dir_;
};

class DynamicLibraryExecutor : public IExecutor {
public:
    void execute(const std::string& lib_path, ExecutionContext* ctx) override;
};

class QueryEngine {
public:
    QueryEngine(std::shared_ptr<Catalog> catalog,
                std::unique_ptr<IQueryCache> cache,
                std::unique_ptr<ICompiler> compiler,
                std::unique_ptr<IExecutor> executor);

    void executeQuery(const std::string& sql, ExecutionContext* ctx);
    std::string generateQueryCode(const std::string& sql);

private:
    std::shared_ptr<Catalog> catalog_;
    std::unique_ptr<IQueryCache> cache_;
    std::unique_ptr<ICompiler> compiler_;
    std::unique_ptr<IExecutor> executor_;
};

} // namespace db
