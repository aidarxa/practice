#include "core/execution.h"
#include "core/optimizer_rules.h"
#include "core/translator.h"
#include "core/visitor.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
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



static void collectHashJoinTables(const OperatorNode* node,
                                  const Catalog& catalog,
                                  std::unordered_set<std::string>& out) {
    if (!node) return;
    if (node->getType() == OperatorType::HASH_JOIN) {
        std::vector<const TableScanNode*> scans;
        std::function<void(const OperatorNode*)> collect_scans = [&](const OperatorNode* n) {
            if (!n) return;
            if (n->getType() == OperatorType::TABLE_SCAN) {
                scans.push_back(static_cast<const TableScanNode*>(n));
            }
            for (const auto& child : n->getChildren()) collect_scans(child.get());
        };
        collect_scans(node);
        for (const auto* scan : scans) {
            try {
                const auto& meta = catalog.getTableMetadata(scan->table_name);
                if (!meta.isFactTable()) out.insert(scan->table_name);
            } catch (...) {}
        }
    }
    for (const auto& child : node->getChildren()) {
        collectHashJoinTables(child.get(), catalog, out);
    }
}

static uint64_t estimatePhtSlotsForTable(const std::string& table,
                                         const Catalog& catalog) {
    if (table == "DDATE") return 61131ULL;
    return catalog.getTableMetadata(table).getSize();
}

static uint64_t estimateProjectionInputRows(const OperatorNode* root,
                                            const Catalog& catalog) {
    const ProjectionNode* proj = findProjectionNode(root);
    if (!proj || proj->getChildren().empty()) return 0;

    std::vector<const TableScanNode*> scans;
    std::function<void(const OperatorNode*)> collect = [&](const OperatorNode* node) {
        if (!node) return;
        if (node->getType() == OperatorType::TABLE_SCAN) {
            scans.push_back(static_cast<const TableScanNode*>(node));
        }
        for (const auto& child : node->getChildren()) collect(child.get());
    };
    collect(proj->getChildren()[0].get());

    uint64_t row_count = 1;
    for (const auto* scan : scans) {
        const auto& meta = catalog.getTableMetadata(scan->table_name);
        if (row_count == 1 || meta.isFactTable()) row_count = meta.getSize();
    }
    return row_count;
}

static size_t checkedMulSize(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string(label) + ": size_t multiplication overflow");
    }
    return a * b;
}

static size_t estimateJitTemporaryBytes(const OperatorNode* root,
                                        const Catalog& catalog) {
    std::unordered_set<std::string> hash_tables;
    collectHashJoinTables(root, catalog, hash_tables);

    size_t bytes = 0;
    for (const auto& table : hash_tables) {
        // PHT_1/PHT_2 are represented as int arrays.  The row-id projection
        // path still uses PHT_2, i.e. two int slots per hash slot.
        size_t slots = static_cast<size_t>(estimatePhtSlotsForTable(table, catalog));
        size_t table_bytes = checkedMulSize(checkedMulSize(2, slots, "PHT slots"),
                                            sizeof(int), "PHT bytes");
        if (bytes > std::numeric_limits<size_t>::max() - table_bytes) {
            throw std::overflow_error("temporary memory estimate overflow");
        }
        bytes += table_bytes;
    }

    if (findProjectionNode(root)) {
        const uint64_t input_rows = estimateProjectionInputRows(root, catalog);
        const size_t projection_tiles = static_cast<size_t>((input_rows + 511ULL) / 512ULL);
        // Exact materialization uses three per-tile ULL arrays:
        // counts, exclusive offsets, write-local counters, plus one block-sums
        // array for the GPU prefix scan over tile counts.
        const size_t projection_scan_blocks = (projection_tiles + 255ULL) / 256ULL;
        const size_t projection_tile_bytes = checkedMulSize(
            checkedMulSize(3, projection_tiles, "projection tile buffers"),
            sizeof(unsigned long long), "projection tile bytes");
        const size_t projection_scan_bytes = checkedMulSize(
            projection_scan_blocks, sizeof(unsigned long long),
            "projection scan block bytes");
        if (projection_tile_bytes > std::numeric_limits<size_t>::max() - projection_scan_bytes) {
            throw std::overflow_error("projection temporary memory estimate overflow");
        }
        const size_t projection_temp = projection_tile_bytes + projection_scan_bytes;
        if (bytes > std::numeric_limits<size_t>::max() - projection_temp) {
            throw std::overflow_error("temporary memory estimate overflow");
        }
        bytes += projection_temp;
    }
    return bytes;
}

static std::string formatBytes(size_t bytes) {
    std::ostringstream out;
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double MiB = 1024.0 * 1024.0;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<size_t>(GiB)) out << (bytes / GiB) << " GiB";
    else out << (bytes / MiB) << " MiB";
    return out.str();
}

static void preflightDeviceMemoryOrThrow(const ExecutionContext* ctx,
                                         const OperatorNode* root,
                                         const Catalog& catalog) {
    if (!ctx || !ctx->q_) return;
    const auto dev = ctx->q_->get_device();
    const size_t total = static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());
    if (total == 0) return;

    const size_t result_bytes = checkedMulSize(ctx->expected_result_size_,
                                               sizeof(unsigned long long),
                                               "result buffer bytes");
    const size_t temporary_bytes = estimateJitTemporaryBytes(root, catalog);
    const size_t loaded_bytes = ctx->loaded_device_bytes_;
    const size_t existing_result_bytes = ctx->result_buffer_
        ? checkedMulSize(ctx->result_buffer_->capacity(), sizeof(unsigned long long),
                         "existing result buffer bytes")
        : 0;

    // DynamicDeviceBuffer frees the old result buffer before allocating a larger
    // one.  If the existing buffer is already large enough, it stays resident.
    const size_t resident_result_bytes = existing_result_bytes >= result_bytes
        ? existing_result_bytes
        : result_bytes;

    size_t required = loaded_bytes;
    auto add = [&](size_t v, const char* label) {
        if (required > std::numeric_limits<size_t>::max() - v) {
            throw std::overflow_error(std::string(label) + ": memory estimate overflow");
        }
        required += v;
    };
    add(resident_result_bytes, "resident result bytes");
    add(temporary_bytes, "JIT temporary bytes");

    // Keep a conservative reserve for runtime allocations, code objects, queues,
    // driver bookkeeping, and fragmentation.  A hard allocation failure in ROCm
    // may destabilize the graphics session, so reject before malloc_device.
    const size_t reserve = std::max(total / 10, static_cast<size_t>(512ULL * 1024ULL * 1024ULL));
    const size_t budget = total > reserve ? total - reserve : total / 2;

    if (required > budget) {
        std::ostringstream msg;
        msg << "Insufficient GPU memory for query before JIT execution. "
            << "Estimated resident requirement: " << formatBytes(required)
            << " (loaded columns: " << formatBytes(loaded_bytes)
            << ", result upper bound: " << formatBytes(result_bytes)
            << ", JIT temporaries: " << formatBytes(temporary_bytes)
            << "). Device memory: " << formatBytes(total)
            << ", safety budget: " << formatBytes(budget)
            << ". Reduce projection width, add selective predicates, or lower scale factor.";
        throw std::runtime_error(msg.str());
    }
}

static void collectTableScansForResultLayout(const OperatorNode* node,
                                             std::vector<const TableScanNode*>& out) {
    if (!node) return;
    if (node->getType() == OperatorType::TABLE_SCAN) {
        out.push_back(static_cast<const TableScanNode*>(node));
    }
    for (const auto& child : node->getChildren()) {
        collectTableScansForResultLayout(child.get(), out);
    }
}

static std::vector<ResultColumnDesc> inferResultColumns(const OperatorNode* node,
                                                        const Catalog& catalog) {
    std::vector<ResultColumnDesc> descs;
    if (!node) return descs;

    if (const auto* agg = findAggregateNode(node)) {
        for (std::size_t i = 0; i < agg->group_by_exprs.size(); ++i) {
            descs.push_back({LogicalType::Int64, 0, false});
        }
        for (const auto& agg_def : agg->aggregates) {
            if (agg_def.isAvg()) {
                descs.push_back({LogicalType::Float64, 0, true});
            } else if (agg_def.isCount()) {
                descs.push_back({LogicalType::UInt64, 0, false});
            } else {
                descs.push_back({LogicalType::UInt64, 0, true});
            }
        }
        return descs;
    }

    if (const auto* proj = findProjectionNode(node)) {
        std::vector<const TableScanNode*> scans;
        if (!proj->getChildren().empty()) {
            collectTableScansForResultLayout(proj->getChildren()[0].get(), scans);
        }
        for (const auto& expr : proj->select_exprs) {
            if (!expr) continue;
            if (expr->getType() == ExprType::STAR) {
                for (const auto* scan : scans) {
                    const auto& meta = catalog.getTableMetadata(scan->table_name);
                    for (uint64_t i = 0; i < meta.getColumnCount(); ++i) {
                        descs.push_back({LogicalType::Int64, 0, false});
                    }
                }
            } else {
                descs.push_back({LogicalType::Int64, 0, false});
            }
        }
    }
    return descs;
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
    } else if (findProjectionNode(optimized_tree.get())) {
        // Projection uses generated two-pass exact materialization:
        // Count pass -> host prefix scan -> exact ensureCapacity() -> write pass.
        // Do not allocate the upper-bound result buffer before JIT execution.
        ctx->expected_result_size_ = 1;
    } else {
        ctx->expected_result_size_ = 1;
    }

    ctx->result_columns_ = inferResultColumns(optimized_tree.get(), *catalog_);
    ctx->result_row_count_ = 0;
    ctx->result_is_dense_ = findProjectionNode(optimized_tree.get()) != nullptr;

    preflightDeviceMemoryOrThrow(ctx, optimized_tree.get(), *catalog_);

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
