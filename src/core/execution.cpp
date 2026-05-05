#include "core/execution.h"
#include "core/optimizer.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace db {

// --- FileBasedQueryCache ---

std::optional<std::string> FileBasedQueryCache::get(const std::string& query_hash) {
    auto it = cache_.find(query_hash);
    if (it != cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void FileBasedQueryCache::put(const std::string& query_hash, const std::string& lib_path) {
    cache_[query_hash] = lib_path;
}

// --- AdaptiveCppCompiler ---

std::string AdaptiveCppCompiler::compile(const std::string& source_code, const std::string& query_hash) {
    std::string cpp_path = "/tmp/" + query_hash + ".cpp";
    std::string so_path = "/tmp/" + query_hash + ".so";

    // Write source code to file
    std::ofstream ofs(cpp_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + cpp_path);
    }
    ofs << source_code;
    ofs.close();

    // TODO: Read INCLUDE path from config/env
    std::string include_path = "/home/aidar/practice/include";
    std::string deps_include = "/home/aidar/practice/deps/include";

    std::string compile_cmd = "acpp -O3 -fPIC -shared -I" + include_path + " -I" + deps_include + " " + cpp_path + " -o " + so_path;

    int ret = std::system(compile_cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("AdaptiveCPP compilation failed with code " + std::to_string(ret));
    }

    return so_path;
}

// --- DynamicLibraryExecutor ---

void DynamicLibraryExecutor::execute(const std::string& lib_path, ExecutionContext* ctx) {
    void* handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error("dlopen failed: " + std::string(dlerror()));
    }

    // RAII for dlclose
    struct DlCloser {
        void* h;
        ~DlCloser() { if (h) dlclose(h); }
    } closer{handle};

    dlerror(); // Clear any existing errors
    void* sym = dlsym(handle, "execute_query");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        throw std::runtime_error("dlsym failed: " + std::string(dlsym_error));
    }

    typedef void (*JitFunc)(db::ExecutionContext*);
    JitFunc func = reinterpret_cast<JitFunc>(sym);

    // Execute the kernel
    func(ctx);
}

// --- QueryEngine ---

QueryEngine::QueryEngine(std::shared_ptr<Catalog> catalog,
                         std::unique_ptr<IQueryCache> cache,
                         std::unique_ptr<ICompiler> compiler,
                         std::unique_ptr<IExecutor> executor)
    : catalog_(std::move(catalog)),
      cache_(std::move(cache)),
      compiler_(std::move(compiler)),
      executor_(std::move(executor)) {}

std::string QueryEngine::generateQueryCode(const std::string& sql) {
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sql, &result);
    if (!result.isValid() || result.size() == 0) {
        throw std::runtime_error("SQL syntax error: " + std::string(result.errorMsg()));
    }

    auto* ast = const_cast<hsql::SelectStatement*>(
        static_cast<const hsql::SelectStatement*>(result.getStatement(0)));

    Planner planner(catalog_);
    auto lp = planner.buildLogicalPlan(ast);

    QueryOptimizer optimizer(catalog_);
    optimizer.optimize(lp);

    auto pp = planner.buildPhysicalPlan(lp);

    CodeGenerator cg;
    return cg.generate(*pp);
}

void QueryEngine::executeQuery(const std::string& sql, ExecutionContext* ctx) {
    if (sql.empty()) {
        throw std::runtime_error("Empty query");
    }

    // Hash the query for caching
    std::string query_hash = "query_" + std::to_string(std::hash<std::string>{}(sql));

    auto cached_lib = cache_->get(query_hash);
    if (cached_lib.has_value()) {
        executor_->execute(cached_lib.value(), ctx);
        return;
    }

    // Cache MISS: generate, compile, then execute
    std::string source_code = generateQueryCode(sql);
    std::string lib_path = compiler_->compile(source_code, query_hash);
    cache_->put(query_hash, lib_path);

    executor_->execute(lib_path, ctx);
}

} // namespace db
