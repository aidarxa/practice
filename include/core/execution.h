#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <cstddef>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#include "core/memory.h"
#include "core/device_buffer.h"
#include "core/config.h"
#include "core/execution_plan.h"
#include "core/module_cache.h"
#include "core/text.h"
#include "core/text_storage.h"

namespace db {

class LegacyQueue;
class PlanPrefixService;
class TextFetch;

using ::LogicalType;

struct ResultColumnDesc {
    LogicalType type = LogicalType::UInt64;
    uint32_t scale = 0;
    bool nullable = false;
};


struct QueryTimingStats {
    /** @brief Execution scheduling changes interval relationships, never the direct summary boundary. */
    const char* model = "pcuda-modules-v1";
    /** @brief Stage-5 native command sums, direct build elapsed and actual per-query event deltas. */
    ModuleBuildTiming module_build;
    ModuleCounters module_events;
    /**
     * @brief Direct SQL parse-to-ready-PCUDA-source elapsed, including resolve/parameters,
     * plan/schema/cache lookup, missing-plan optimization/lowering and module wrapper emission.
     * @details Stops before compiler/file/build-script work, loading or runtime/GPU allocations.
     * Ready plans stop at plan lookup. This overlaps prepare/codegen: never add it to components.
     */
    double sql_to_pcuda_ms = 0.0;
    // Query preparation before code generation or cached-library execution:
    // SQL parsing, logical/physical preparation, result-shape inference,
    // memory preflight, result-buffer preparation, cache-key build and lookup.
    double prepare_ms = 0.0;
    double codegen_ms = 0.0;
    double compile_ms = 0.0;
    double library_load_ms = 0.0;
    // Stage 2 measures the host plan interpreter, including submissions,
    // runtime allocation/init, kernels, required waits and service copies.
    // The legacy field name is retained; no whole-query execute_query remains.
    double generated_execute_ms = 0.0;
    // Backward-compatible alias for old reports. This is not pure GPU-kernel
    // time; reporting keeps it equal to generated_execute_ms.
    double gpu_execute_ms = 0.0;
    // Backward-compatible alias for old code paths; kept in sync with
    // generated_execute_ms at reporting boundaries.
    double jit_execute_ms = 0.0;
    double host_fetch_ms = 0.0;
    // Full engine time before host result fetch:
    // prepare + codegen + compile + library_load + generated_execute.
    double engine_ms = 0.0;
    // Full SQL query path through the prototype, from DatabaseInstance entry
    // to a fully populated CPU QueryResult.
    double total_query_ms = 0.0;
    // Legacy aggregate field. QueryResult finalization populates it with
    // total_query_ms so old consumers get the complete query time.
    double total_engine_ms = 0.0;
};

struct QueryFetchOptions {
    bool limit_enabled = false;
    std::size_t row_limit = 0;
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

    // Typed columnar host result. Exactly one typed vector is populated per
    // column according to columns[c].type. column_data is retained as a raw
    // UInt64-compatible view for legacy tests and diagnostics only.
    std::vector<std::vector<unsigned long long>> column_data;
    std::vector<std::vector<std::int64_t>> column_i64;
    std::vector<std::vector<std::uint64_t>> column_u64;
    std::vector<std::vector<double>> column_f64;
    std::vector<std::vector<std::string>> column_text; ///< Owned UTF-8 values, never device addresses or IDs.
    std::vector<std::vector<uint64_t>> column_validity_bitmap;
    bool has_columnar_result = false;

    // If only a display window was fetched from GPU, row_count still stores
    // the full logical result size, while materialized_row_offset/count
    // identify the host-resident row interval [offset, offset+count).
    bool partial_result = false;
    size_t materialized_row_offset = 0;
    size_t materialized_row_count = 0;

    size_t tuple_size = 1;
    std::vector<ResultColumnDesc> columns;
    std::vector<std::string> column_names;

    // Number of logical rows returned by the query. This is not necessarily
    // data.size() / tuple_size for sparse hash-aggregate result buffers where
    // empty groups are represented by all-zero tuples.
    size_t row_count = 0;

    // Dense results, such as projection materialization, contain only returned
    // rows. Sparse results, such as fixed-domain grouped aggregates, may contain
    // empty buckets that must be skipped by the CLI.
    bool dense_result = false;

    QueryTimingStats timing;
};


struct ResultColumnDeviceBuffer {
    LogicalType type = LogicalType::UInt64;
    std::unique_ptr<DynamicDeviceBuffer<std::int64_t>> i64;
    std::unique_ptr<DynamicDeviceBuffer<std::uint64_t>> u64;
    std::unique_ptr<DynamicDeviceBuffer<double>> f64;
    std::unique_ptr<DynamicDeviceBuffer<TextView>> text;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> validity;

    /** @brief A column owns one logical payload; obsolete types are not an unbounded side cache. */
    void retainType(LogicalType requested_type) {
        if (requested_type!=LogicalType::Text) text.reset();
        if (requested_type!=LogicalType::Int64) i64.reset();
        if (requested_type!=LogicalType::Float64) f64.reset();
        if (requested_type!=LogicalType::UInt64) u64.reset();
        type=requested_type;
    }

    void ensure(const std::shared_ptr<DeviceRuntime>& runtime, LogicalType requested_type, size_t row_count) {
        retainType(requested_type);
        const size_t value_capacity = row_count == 0 ? 1 : row_count;
        const size_t words = row_count / 64 + (row_count % 64 != 0);
        const size_t validity_words = words == 0 ? 1 : words;

        if (type == LogicalType::Text) {
            if (!text) text = std::make_unique<DynamicDeviceBuffer<TextView>>(runtime, 0);
            text->ensureCapacity(value_capacity);
            text->zero();
        } else if (type == LogicalType::Int64) {
            if (!i64) i64 = std::make_unique<DynamicDeviceBuffer<std::int64_t>>(runtime, 0);
            i64->ensureCapacity(value_capacity);
            i64->zero();
        } else if (type == LogicalType::Float64) {
            if (!f64) f64 = std::make_unique<DynamicDeviceBuffer<double>>(runtime, 0);
            f64->ensureCapacity(value_capacity);
            f64->zero();
        } else {
            if (!u64) u64 = std::make_unique<DynamicDeviceBuffer<std::uint64_t>>(runtime, 0);
            u64->ensureCapacity(value_capacity);
            u64->zero();
        }

        if (!validity) validity = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, 0);
        validity->ensureCapacity(validity_words);
        validity->zero();
    }
};

/** @brief A base column owns its data and optional validity independently of borrowed views. */
struct LoadedDeviceColumn {
    std::unique_ptr<DynamicDeviceBuffer<int>> values;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> validity;
};

struct ExecutionContext {
    /** @brief Owners are declared before buffers so their stream outlives all allocations. */
    std::shared_ptr<DeviceRuntime> runtime_;
    std::shared_ptr<LegacyQueue> legacy_queue_;
    /** @brief Prebuilt scan and conversion workspace, never owned by a generated module. */
    std::shared_ptr<PlanPrefixService> prefix_service_;
    std::unordered_map<std::string, LoadedDeviceColumn> base_columns_;
    std::unordered_map<std::string, LoadedTextColumn> text_columns_;
    std::unique_ptr<DynamicDeviceBuffer<uint8_t>> text_parameter_arena_;
    std::shared_ptr<TextFetch> text_fetch_;
    std::unique_ptr<DynamicDeviceBuffer<unsigned long long>> result_buffer_;
    std::unique_ptr<DynamicDeviceBuffer<uint64_t>> result_validity_buffer_;
    std::unordered_map<std::string, void*> buffers_;
    std::unordered_map<std::string, uint64_t*> null_bitmaps_;
    size_t tuple_size_ = 1;
    std::vector<ResultColumnDesc> result_columns_;
    std::vector<std::string> result_column_names_;
    size_t result_row_count_{0};
    bool result_is_dense_{false};
    bool result_is_columnar_{false};
    size_t result_column_count_{0};
    size_t result_column_row_capacity_{0};
    std::vector<ResultColumnDeviceBuffer> result_column_storage_;
    std::unordered_map<std::string, std::unique_ptr<DynamicDeviceBuffer<int>>> scratch_i32_buffers_;
    std::unordered_map<std::string, std::unique_ptr<DynamicDeviceBuffer<unsigned long long>>> scratch_u64_buffers_;
    /** @brief Temporary legacy adapter's runtime-owned, reusable byte allocations. */
    std::unordered_map<std::string, std::unique_ptr<DynamicDeviceBuffer<uint8_t>>> plan_scratch_buffers_;
    std::string active_plan_identity_; ///< Value-free owner of the retained query workspace, not a module cache key.
    size_t expected_result_validity_words_{0};
    QueryTimingStats timing_;
    CrystalConfig config_;

    // Размер буфера результатов в элементах (устанавливается QueryEngine перед выполнением).
    // DatabaseInstance использует это значение для copyToHost вместо хардкода 21000.
    size_t expected_result_size_{0};

    // Diagnostic base-column bytes only. DeviceRuntime's actual allocation ledger
    // guards memory, including validity, retained typed storage, staging and growth peaks.
    size_t loaded_device_bytes_{0};

    template<typename T>
    T* getBuffer(const std::string& name) {
        auto owned = base_columns_.find(name);
        if (owned != base_columns_.end()) return reinterpret_cast<T*>(owned->second.values->data());
        auto it = buffers_.find(name);
        if (it != buffers_.end()) {
            return static_cast<T*>(it->second);
        }
        return nullptr;
    }


    uint64_t* getNullBitmap(const std::string& name) {
        auto owned = base_columns_.find(name);
        if (owned != base_columns_.end()) return owned->second.validity ? owned->second.validity->data() : nullptr;
        auto it = null_bitmaps_.find(name);
        if (it == null_bitmaps_.end()) return nullptr;
        return it->second;
    }

    template<typename T>
    ColumnView<T> getColumnView(const std::string& name) {
        return ColumnView<T>{getBuffer<T>(name), getNullBitmap(name), getNullBitmap(name) != nullptr};
    }

    int* getScratchIntBuffer(const std::string& name, size_t element_count) {
        auto& buf = scratch_i32_buffers_[name];
        if (!buf) buf = std::make_unique<DynamicDeviceBuffer<int>>(runtime_, 0);
        buf->ensureCapacity(element_count == 0 ? 1 : element_count);
        return buf->data();
    }

    unsigned long long* getScratchUInt64Buffer(const std::string& name, size_t element_count) {
        auto& buf = scratch_u64_buffers_[name];
        if (!buf) buf = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime_, 0);
        buf->ensureCapacity(element_count == 0 ? 1 : element_count);
        return buf->data();
    }

    unsigned long long* getResultPointer() const {
        return result_buffer_ ? result_buffer_->data() : nullptr;
    }

    uint64_t* getResultValidityPointer() const {
        return result_validity_buffer_ ? result_validity_buffer_->data() : nullptr;
    }

    /** @brief Legacy capacity helper; the physical plan initializes explicitly when initialize=false. */
    void ensureResultValidityCapacity(size_t value_count, bool initialize = true) {
        expected_result_validity_words_ = value_count / 64 + (value_count % 64 != 0);
        if (result_validity_buffer_) {
            result_validity_buffer_->ensureCapacity(expected_result_validity_words_ == 0 ? 1 : expected_result_validity_words_);
            if (initialize) result_validity_buffer_->zero();
        }
    }

    void ensureColumnarResultCapacity(size_t column_count, size_t row_count) {
        result_is_columnar_ = true;
        result_column_count_ = column_count;
        result_column_row_capacity_ = row_count;
        result_column_storage_.resize(column_count);
        for (size_t col = 0; col < column_count; ++col) {
            LogicalType type = LogicalType::UInt64;
            if (col < result_columns_.size()) type = result_columns_[col].type;
            result_column_storage_[col].ensure(runtime_, type, row_count);
        }
    }

    std::uint64_t* getResultColumnUInt64Pointer(size_t col) const {
        if (col >= result_column_storage_.size() || !result_column_storage_[col].u64) {
            throw std::out_of_range("UInt64 result column buffer is not allocated");
        }
        return result_column_storage_[col].u64->data();
    }

    std::int64_t* getResultColumnInt64Pointer(size_t col) const {
        if (col >= result_column_storage_.size() || !result_column_storage_[col].i64) {
            throw std::out_of_range("Int64 result column buffer is not allocated");
        }
        return result_column_storage_[col].i64->data();
    }

    double* getResultColumnFloat64Pointer(size_t col) const {
        if (col >= result_column_storage_.size() || !result_column_storage_[col].f64) {
            throw std::out_of_range("Float64 result column buffer is not allocated");
        }
        return result_column_storage_[col].f64->data();
    }

    // Backward-compatible name for old generated libraries. New codegen emits
    // the typed accessors above.
    unsigned long long* getResultColumnPointer(size_t col) const {
        return reinterpret_cast<unsigned long long*>(getResultColumnUInt64Pointer(col));
    }

    uint64_t* getResultColumnValidityPointer(size_t col) const {
        if (col >= result_column_storage_.size() || !result_column_storage_[col].validity) {
            throw std::out_of_range("result column validity buffer is not allocated");
        }
        return result_column_storage_[col].validity->data();
    }

    void resetForQuery() {
        resetResultShapeFlags();
        timing_ = QueryTimingStats{};
    }

    void resetResultShapeFlags() {
        result_row_count_ = 0;
        result_is_dense_ = false;
        result_is_columnar_ = false;
        result_column_count_ = 0;
        result_column_row_capacity_ = 0;
    }
};

class IExecutor {
public:
    virtual ~IExecutor() = default;
    virtual void execute(const LoadedKernels& kernels, const ExecutionPlan& plan,
                         const ParameterValues& parameters, ExecutionContext* ctx) = 0;
    /** @brief Strategies without staged execution remain correct by resolving the full set first. */
    virtual void execute(const KernelLookup&,const ExecutionPlan&,const ParameterValues&,ExecutionContext*);
};

class DynamicLibraryExecutor : public IExecutor {
public:
    void execute(const LoadedKernels& kernels, const ExecutionPlan& plan,
                   const ParameterValues& parameters, ExecutionContext* ctx) override;
    /** @brief Experimental readiness is consumed only at a kernel's first actual launch. */
    void execute(const KernelLookup&,const ExecutionPlan&,const ParameterValues&,ExecutionContext*) override;
};

class QueryEngine {
public:
    QueryEngine(std::shared_ptr<Catalog> catalog,
                std::unique_ptr<ProcessModuleCache> cache,
                std::unique_ptr<ICompiler> compiler,
                std::unique_ptr<IExecutor> executor);

    void executeQuery(const std::string& sql, ExecutionContext* ctx);
    std::string generateQueryCode(const std::string& sql);
    /** @brief Actual process-lifetime preparation/load events; device JIT is a separate layer. */
    const ModuleCounters& moduleCounters() const { return counters_; }

private:
    std::shared_ptr<Catalog> catalog_;
    std::unique_ptr<ProcessModuleCache> cache_;
    std::unique_ptr<ICompiler> compiler_;
    std::unique_ptr<IExecutor> executor_;
    struct CachedQuery;
    std::unordered_map<std::string, std::shared_ptr<CachedQuery>> plans_;
    /** @brief Last prepared slot types per logical/catalog shape, so SQL NULL retains a known numeric type. */
    std::unordered_map<std::string, ParameterSchema> known_parameter_types_;
    ModuleCounters counters_;
};

} // namespace db
