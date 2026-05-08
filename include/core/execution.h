#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <sycl/sycl.hpp>

#include "core/memory.h"
#include "core/device_buffer.h"

namespace db {

struct ExecutionContext {
    sycl::queue* q_ = nullptr;
    std::unique_ptr<DynamicDeviceBuffer<unsigned long long>> result_buffer_;
    std::unordered_map<std::string, void*> buffers_;
    size_t tuple_size_ = 1;

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
    std::string compile(const std::string& source_code, const std::string& query_hash) override;
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
