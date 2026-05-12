#include "core/execution.h"
#include "core/optimizer_rules.h"
#include "core/translator.h"
#include "core/visitor.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

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
AdaptiveCppCompiler::AdaptiveCppCompiler(std::string include_dir,
                                         std::string deps_include_dir)
    : include_dir_(std::move(include_dir)),
      deps_include_dir_(std::move(deps_include_dir)) {
    if (include_dir_.empty()) {
#ifdef CRYSTAL_INCLUDE_DIR_DEFAULT
        include_dir_ = CRYSTAL_INCLUDE_DIR_DEFAULT;
#endif
    }
    if (deps_include_dir_.empty()) {
#ifdef CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT
        deps_include_dir_ = CRYSTAL_DEPS_INCLUDE_DIR_DEFAULT;
#endif
    }
}

static std::string escapeShellArg(const std::string& value) {
    std::string escaped = "'";
    for (char c : value) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    escaped += "'";
    return escaped;
}

static std::string getRequiredPath(const char* env_name,
                                   const std::string& configured_value) {
    if (!configured_value.empty()) {
        return configured_value;
    }
    const char* env_value = std::getenv(env_name);
    if (!env_value) {
        throw std::runtime_error(
            std::string("Missing required path. Set environment variable ") +
            env_name + " or pass it into AdaptiveCppCompiler constructor.");
    }
    std::string path(env_value);
    if (path.empty()) {
        throw std::runtime_error(
            std::string("Environment variable ") + env_name + " is empty.");
    }
    return path;
}

static void validateExistingDir(const std::string& path, const std::string& name) {
    if (path.empty()) {
        throw std::runtime_error(name + " path is empty.");
    }
    std::filesystem::path fs_path(path);
    if (!std::filesystem::exists(fs_path)) {
        throw std::runtime_error(name + " path does not exist: " + path);
    }
    if (!std::filesystem::is_directory(fs_path)) {
        throw std::runtime_error(name + " path is not a directory: " + path);
    }
}

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

    const std::string include_path = getRequiredPath("CRYSTAL_INCLUDE_DIR", include_dir_);
    const std::string deps_include = getRequiredPath("CRYSTAL_DEPS_INCLUDE_DIR", deps_include_dir_);
    validateExistingDir(include_path, "CRYSTAL_INCLUDE_DIR");
    validateExistingDir(deps_include, "CRYSTAL_DEPS_INCLUDE_DIR");

    std::vector<std::string> args = {
        "acpp", "-O3", "-fPIC", "-shared",
        "-I" + include_path,
        "-I" + deps_include,
        cpp_path,
        "-o", so_path
    };

    std::string compile_cmd;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i != 0) compile_cmd += " ";
        compile_cmd += escapeShellArg(args[i]);
    }

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

// ============================================================================
// Helpers
// ============================================================================

// Рекурсивный поиск AggregateNode в корне дерева (глубина ≤ 2, он всегда сверху).
static const AggregateNode* findAggregateNode(const OperatorNode* node) {
    if (!node) return nullptr;
    if (node->getType() == OperatorType::AGGREGATE) {
        return static_cast<const AggregateNode*>(node);
    }
    for (const auto& child : node->getChildren()) {
        const AggregateNode* found = findAggregateNode(child.get());
        if (found) return found;
    }
    return nullptr;
}

static const ProjectionNode* findProjectionNode(const OperatorNode* node) {
    if (!node) return nullptr;
    if (node->getType() == OperatorType::PROJECTION) {
        return static_cast<const ProjectionNode*>(node);
    }
    for (const auto& child : node->getChildren()) {
        const ProjectionNode* found = findProjectionNode(child.get());
        if (found) return found;
    }
    return nullptr;
}

// Парсинг SQL → SelectStatement. Бросает runtime_error при ошибке.
static hsql::SelectStatement* parseSql(const std::string& sql,
                                        hsql::SQLParserResult& result) {
    hsql::SQLParser::parse(sql, &result);
    if (!result.isValid() || result.size() == 0) {
        throw std::runtime_error("SQL syntax error: " + std::string(result.errorMsg()));
    }
    return const_cast<hsql::SelectStatement*>(
        static_cast<const hsql::SelectStatement*>(result.getStatement(0)));
}

// Трансляция + оптимизация AST → оптимизированное дерево операторов.
static std::unique_ptr<OperatorNode> buildOptimizedTree(
        const hsql::SelectStatement* ast) {
    QueryTranslator translator;
    auto naive_tree = translator.translate(ast);
    Optimizer optimizer;
    return optimizer.optimize(std::move(naive_tree));
}

// ============================================================================
// generateQueryCode — новый конвейер (Translator → Optimizer → JITVisitor)
// ============================================================================
std::string QueryEngine::generateQueryCode(const std::string& sql) {
    hsql::SQLParserResult parse_result;
    auto* ast = parseSql(sql, parse_result);

    auto optimized_tree = buildOptimizedTree(ast);

    JITContext jit_ctx;
    JITOperatorVisitor visitor(jit_ctx, *catalog_);
    optimized_tree->accept(visitor);

    return visitor.generateCode();
}

// ============================================================================
// executeQuery — новый конвейер
// ============================================================================
void QueryEngine::executeQuery(const std::string& sql, ExecutionContext* ctx) {
    if (sql.empty()) {
        throw std::runtime_error("Empty query");
    }

    // ШАГ 1: Парсинг SQL
    hsql::SQLParserResult parse_result;
    auto* ast = parseSql(sql, parse_result);

    // ШАГ 2-3: Трансляция + оптимизация (всегда, включая cache hit — дёшево)
    auto optimized_tree = buildOptimizedTree(ast);

    // ШАГ 4: Расчёт expected_result_size (нужен до выполнения для ensureCapacity)
    const AggregateNode* agg_node = findAggregateNode(optimized_tree.get());
    if (agg_node) {
        ctx->expected_result_size_ = agg_node->calculateResultSize(*catalog_);
    } else if (const ProjectionNode* proj_node = findProjectionNode(optimized_tree.get())) {
        ctx->expected_result_size_ = proj_node->calculateResultSize(*catalog_);
    } else {
        ctx->expected_result_size_ = 1;
    }

    // Гарантируем достаточную ёмкость буфера и обнуляем его перед запуском ядра.
    // DynamicDeviceBuffer::ensureCapacity реаллоцирует только при нехватке места.
    ctx->result_buffer_->ensureCapacity(ctx->expected_result_size_);
    ctx->result_buffer_->zero();            // ctx->result_buffer_ почему-то имеет capacity 31500 при expected_result_size_ = 21000


    // ШАГ 5: Проверка кеша
    std::string query_hash = "query_" + std::to_string(std::hash<std::string>{}(sql));
    auto cached_lib = cache_->get(query_hash);
    if (cached_lib.has_value()) {
        // Cache HIT: размер уже рассчитан, буфер подготовлен — просто выполняем
        executor_->execute(cached_lib.value(), ctx);
        return;
    }

    // ШАГ 6: Cache MISS — JIT генерация кода
    JITContext jit_ctx;
    JITOperatorVisitor visitor(jit_ctx, *catalog_);
    optimized_tree->accept(visitor);
    std::string source_code = visitor.generateCode();

    // ШАГ 7: Компиляция → .so
    std::string lib_path = compiler_->compile(source_code, query_hash);
    cache_->put(query_hash, lib_path);

    // ШАГ 8: Выполнение
    executor_->execute(lib_path, ctx);
}

} // namespace db
