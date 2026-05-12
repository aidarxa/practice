#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <cstddef>
#include <unordered_map>
#include <vector>
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

struct QueryResult {
    std::vector<unsigned long long> data;
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
    std::unordered_map<std::string, void*> buffers_;
    size_t tuple_size_ = 1;
    std::vector<ResultColumnDesc> result_columns_;
    size_t result_row_count_{0};
    bool result_is_dense_{false};

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

    unsigned long long* getResultPointer() const {
        return result_buffer_ ? result_buffer_->data() : nullptr;
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
