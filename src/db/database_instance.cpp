#include "db/database_instance.h"
#include "core/catalog_cache.h"
#include "crystal/utils.h"
#include "core/pcuda_services.h"
#include "core/text_fetch.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>

namespace db {

namespace {
using Clock = std::chrono::steady_clock;
static double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}


DatabaseInstance::DatabaseInstance()
    : config_(loadCrystalConfig()),
      dataset_(SsbDataset::open(config_.data_dir)),
      runtime_(std::make_shared<DeviceRuntime>(config_)) {
    ctx_ = std::make_unique<ExecutionContext>();
    ctx_->runtime_ = runtime_;
    ctx_->config_ = config_;

    initCatalog();

    engine_ = std::make_unique<QueryEngine>(
        catalog_,
        std::make_unique<FileBasedQueryCache>(),
        std::make_unique<AdaptiveCppCompiler>(config_),
        std::make_unique<DynamicLibraryExecutor>()
    );
    // Создаём буфер с нулевой начальной ёмкостью.
    // QueryEngine::executeQuery вызовет ensureCapacity с точным размером
    // перед запуском каждого ядра (через expected_result_size_).
    ctx_->result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime_, 0);
    ctx_->result_validity_buffer_ = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime_, 0);
}

DatabaseInstance::~DatabaseInstance() = default;

void DatabaseInstance::initCatalog() {
    catalog_ = std::make_shared<Catalog>();

    db::CatalogCacheManager cache_manager(dataset_);

    // Лямбда: берет список колонок и размер таблицы, получает стату и записывает
    auto applyStats = [&](TableMetadata& table) {
        const auto table_size = table.getSize();
        for (const auto& col_name : table.getColumnNames()) {
            const auto storage = ssbColumn(col_name).storage;
            table.setColumnStorage(col_name, storage);
            if (!storage.isNumeric()) continue;
            db::CachedStats stats = cache_manager.getStatsForColumn(col_name, table_size);
            if (stats.cardinality > 0) {
                table.setColumnStats(col_name, {stats.min_val, stats.max_val, static_cast<uint64_t>(stats.cardinality)});
            }
        }
    };

    auto applyNullabilityFromStorage = [&](TableMetadata& table) {
        for (const auto& col_name : table.getColumnNames()) {
            if (hasNullBitmapFile(col_name, dataset_.directory())) {
                table.setColumnNullable(col_name, true);
            }
        }
    };

    for (const auto& definition : ssbTables) {
        TableMetadata table(definition.name, ssbColumnNames(definition.name), dataset_.rowCount(definition.name), !*definition.primary_key);
        applyStats(table);
        applyNullabilityFromStorage(table);
        if (*definition.primary_key) table.setColumnPrimaryKey(definition.primary_key);
        catalog_->pushTableMetadata(table);
    }
}

void DatabaseInstance::loadData() {
    if (data_loaded_) return;
    std::cout << "Loading data to GPU..." << std::endl;
    int total_columns = 0;
    for (const auto& table : catalog_->getTablesMetadata()) {
        total_columns += table.getColumnCount();
    }

    // Commit owners only after the entire load succeeds. A failed load is retryable
    // and releases its partial device/pinned storage through the same runtime.
    std::unordered_map<std::string, LoadedDeviceColumn> loaded;
    std::unordered_map<std::string, LoadedTextColumn> text_loaded;
    size_t loaded_bytes = 0;
    int current_column = 0;
    for (const auto& table : catalog_->getTablesMetadata()) {
        uint64_t size = table.getSize();
        for (const auto& col_name : table.getColumnNames()) {
            current_column++;
            const auto storage = table.getColumnStorage(col_name);
            if (!storage.isNumeric()) {
                auto validity = loadValidityBitmapIfExists(col_name, size, dataset_.directory());
                const auto data = loadTextColumn(dataset_.columnPath(col_name), size, storage, validity);
                const bool wide = checkedSizeMultiply(size, storage.width) > INT32_MAX;
                loaded_bytes = checkedSizeAdd(loaded_bytes, checkedSizeAdd(data.bytes.size(), checkedSizeMultiply(data.offsets.size(), wide?8:4)));
                loaded_bytes = checkedSizeAdd(loaded_bytes, checkedSizeMultiply(validity.size(), sizeof(uint64_t)));
                text_loaded.emplace(col_name, uploadTextColumn(runtime_, data, wide, validity));
                continue;
            }
            std::cout << "\r[" << current_column << "/" << total_columns << "] Loading " << std::left << std::setw(20) << col_name << std::flush;

            // Load from disk
            std::vector<int> data = loadColumn<int>(col_name, size, dataset_.directory());

            // Allocate and copy to GPU
            LoadedDeviceColumn column;
            column.values = std::make_unique<DynamicDeviceBuffer<int>>(runtime_, size);
            column.values->copyFromHost(data);
            loaded_bytes = checkedSizeAdd(loaded_bytes, checkedSizeMultiply(size, sizeof(int)));

            std::string buf_name = "d_" + col_name;

            std::vector<uint64_t> validity = loadValidityBitmapIfExists(col_name, size, dataset_.directory());
            if (!validity.empty()) {
                column.validity = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime_, validity.size());
                column.validity->copyFromHost(validity);
                loaded_bytes = checkedSizeAdd(loaded_bytes, checkedSizeMultiply(validity.size(), sizeof(uint64_t)));
            }
            loaded.emplace(buf_name, std::move(column));
        }
    }
    ctx_->base_columns_.swap(loaded);
    ctx_->text_columns_.swap(text_loaded);
    ctx_->loaded_device_bytes_ = loaded_bytes; // diagnostic only; allocation ledger is authoritative
    data_loaded_ = true;
    std::cout << "\nData loading complete!" << std::endl;
}

QueryResult DatabaseInstance::executeQuery(const std::string& sql) {
    return executeQuery(sql, QueryFetchOptions{});
}

QueryResult DatabaseInstance::executeQuery(const std::string& sql, const QueryFetchOptions& fetch_options) {
    const auto total_query_start = Clock::now();

    // QueryEngine::executeQuery самостоятельно вызовет ensureCapacity и zero
    // на основе рассчитанного expected_result_size_.
    engine_->executeQuery(sql, ctx_.get());

    // DynamicLibraryExecutor ставит финальный барьер и относит его к
    // generated_execute_ms. Поэтому host_fetch_ms начинается уже после того,
    // как результат готов к копированию на CPU.
    const auto fetch_start = Clock::now();

    QueryResult result;
    result.tuple_size = ctx_->tuple_size_ == 0 ? 1 : ctx_->tuple_size_;
    result.columns = ctx_->result_columns_;
    result.column_names = ctx_->result_column_names_;
    result.dense_result = ctx_->result_is_dense_;

    auto finalize_timing = [&](QueryResult& out) {
        const auto total_query_end = Clock::now();
        out.timing = ctx_->timing_;
        if (out.timing.generated_execute_ms == 0.0) {
            out.timing.generated_execute_ms = out.timing.gpu_execute_ms != 0.0
                ? out.timing.gpu_execute_ms
                : out.timing.jit_execute_ms;
        }
        out.timing.gpu_execute_ms = out.timing.generated_execute_ms;
        out.timing.jit_execute_ms = out.timing.generated_execute_ms;
        out.timing.host_fetch_ms = elapsedMs(fetch_start, total_query_end);
        out.timing.total_query_ms = elapsedMs(total_query_start, total_query_end);
        out.timing.total_engine_ms = out.timing.total_query_ms;
    };

    if (ctx_->result_is_columnar_) {
        result.row_count = ctx_->result_row_count_;
        result.dense_result = true;
        result.has_columnar_result = true;
        result.column_data.resize(ctx_->result_column_count_);
        result.column_i64.resize(ctx_->result_column_count_);
        result.column_u64.resize(ctx_->result_column_count_);
        result.column_f64.resize(ctx_->result_column_count_);
        result.column_text.resize(ctx_->result_column_count_);
        result.column_validity_bitmap.resize(ctx_->result_column_count_);
        size_t row_offset = 0;
        size_t rows_to_copy = result.row_count;
        if (fetch_options.limit_enabled && fetch_options.row_limit > 0 && result.row_count > fetch_options.row_limit) {
            row_offset = result.row_count - fetch_options.row_limit;
            rows_to_copy = fetch_options.row_limit;
            result.partial_result = true;
        }
        result.materialized_row_offset = row_offset;
        result.materialized_row_count = rows_to_copy;
        const size_t validity_words = rows_to_copy / 64 + (rows_to_copy % 64 != 0);
        for (size_t col = 0; col < ctx_->result_column_count_; ++col) {
            const db::LogicalType type = (col < result.columns.size()) ? result.columns[col].type : db::LogicalType::UInt64;
            if (col < ctx_->result_column_storage_.size()) {
                const auto& storage = ctx_->result_column_storage_[col];
                if (type == db::LogicalType::Text && storage.text) {
                    if (!ctx_->text_fetch_) ctx_->text_fetch_ = std::make_shared<TextFetch>(runtime_);
                    result.column_text[col] = ctx_->text_fetch_->read(storage.text->data(), storage.validity->data(),
                                                                      result.row_count, row_offset, rows_to_copy);
                } else if (type == db::LogicalType::Float64 && storage.f64) {
                    storage.f64->copyRangeToHost(result.column_f64[col], row_offset, rows_to_copy);
                    result.column_data[col].resize(rows_to_copy);
                    for (size_t r = 0; r < rows_to_copy; ++r) {
                        unsigned long long bits = 0ULL;
                        static_assert(sizeof(bits) == sizeof(double), "double/ULL size mismatch");
                        std::memcpy(&bits, &result.column_f64[col][r], sizeof(double));
                        result.column_data[col][r] = bits;
                    }
                } else if (type == db::LogicalType::Int64 && storage.i64) {
                    storage.i64->copyRangeToHost(result.column_i64[col], row_offset, rows_to_copy);
                    result.column_data[col].resize(rows_to_copy);
                    for (size_t r = 0; r < rows_to_copy; ++r) {
                        result.column_data[col][r] = static_cast<unsigned long long>(result.column_i64[col][r]);
                    }
                } else if (storage.u64) {
                    storage.u64->copyRangeToHost(result.column_u64[col], row_offset, rows_to_copy);
                    result.column_data[col].resize(rows_to_copy);
                    for (size_t r = 0; r < rows_to_copy; ++r) {
                        result.column_data[col][r] = static_cast<unsigned long long>(result.column_u64[col][r]);
                    }
                }
                if (storage.validity && col < result.columns.size() && result.columns[col].nullable) {
                    if (row_offset == 0) {
                        storage.validity->copyToHost(result.column_validity_bitmap[col], validity_words);
                    } else {
                        // Copy the minimal source bitmap word range and rebase bits to row 0.
                        const size_t first_word = row_offset >> 6U;
                        const size_t last_bit = checkedSizeAdd(row_offset, rows_to_copy);
                        const size_t source_words = last_bit / 64 + (last_bit % 64 != 0) - first_word;
                        std::vector<uint64_t> source_bitmap;
                        storage.validity->copyRangeToHost(source_bitmap, first_word, source_words);
                        result.column_validity_bitmap[col].assign(validity_words, 0ULL);
                        for (size_t r = 0; r < rows_to_copy; ++r) {
                            const size_t src_row = row_offset + r;
                            const size_t src_local_bit = src_row - first_word * 64ULL;
                            const bool valid = ((source_bitmap[src_local_bit >> 6U] >> (src_local_bit & 63U)) & 1ULL) != 0ULL;
                            if (valid) result.column_validity_bitmap[col][r >> 6U] |= (1ULL << (r & 63U));
                        }
                    }
                }
            }
        }
        finalize_timing(result);
        return result;
    }

    // Legacy row-wise path used by hash/group aggregate kernels.
    size_t copy_size = ctx_->expected_result_size_;
    if (copy_size == 0) copy_size = 1; // защита от нулевого размера

    if (!ctx_->result_is_dense_ && ctx_->expected_result_validity_words_ == 0 && result.tuple_size > 1 && copy_size >= result.tuple_size) {
        // Sparse aggregate buffers are internal GPU state. Do not scan them on the
        // host to find non-empty buckets. Compact the sparse rows on GPU into a
        // dense scratch buffer, then copy only the requested display window.
        const size_t physical_rows = copy_size / result.tuple_size;
        unsigned long long* d_sparse_values = ctx_->getScratchUInt64Buffer("__sparse_result_values", copy_size);
        unsigned long long* d_sparse_count = ctx_->getScratchUInt64Buffer("__sparse_result_count", 1);
        unsigned long long* d_source = ctx_->result_buffer_->data();
        compactSparseRows(*runtime_, d_source, d_sparse_values, d_sparse_count, physical_rows, result.tuple_size);
        std::vector<unsigned long long> sparse_count;
        ctx_->scratch_u64_buffers_.at("__sparse_result_count")->copyToHost(sparse_count, 1);
        if (sparse_count.front() > physical_rows) throw std::runtime_error("sparse result count exceeds source rows");
        result.row_count = static_cast<size_t>(sparse_count.front());
        result.dense_result = true;
        result.partial_result = false;
        size_t row_offset = 0;
        size_t rows_to_copy = result.row_count;
        if (fetch_options.limit_enabled && fetch_options.row_limit > 0 && result.row_count > fetch_options.row_limit) {
            row_offset = result.row_count - fetch_options.row_limit;
            rows_to_copy = fetch_options.row_limit;
            result.partial_result = true;
        }
        result.materialized_row_offset = row_offset;
        result.materialized_row_count = rows_to_copy;
        const size_t values_to_copy = checkedSizeMultiply(rows_to_copy, result.tuple_size);
        if (values_to_copy != 0) {
            auto& sparse_value_buffer = ctx_->scratch_u64_buffers_.at("__sparse_result_values");
            sparse_value_buffer->copyRangeToHost(result.data, checkedSizeMultiply(row_offset, result.tuple_size), values_to_copy);
        }
        // Validity for fast sparse SSB aggregates is absent by design: all copied
        // cells are non-null. Nullable sparse paths use dense/columnar finalization.
        result.has_cell_validity = false;
        finalize_timing(result);
        return result;
    }

    std::vector<unsigned long long> h_result;
    ctx_->result_buffer_->copyToHost(h_result, copy_size);

    std::vector<uint64_t> h_validity;
    if (ctx_->expected_result_validity_words_ > 0 && ctx_->result_validity_buffer_) {
        ctx_->result_validity_buffer_->copyToHost(h_validity, ctx_->expected_result_validity_words_);
    }

    result.data = std::move(h_result);
    result.cell_validity_bitmap = std::move(h_validity);
    result.has_cell_validity = !result.cell_validity_bitmap.empty();

    auto row_valid_at = [&](size_t value_idx) -> bool {
        if (!result.has_cell_validity || result.cell_validity_bitmap.empty()) return true;
        const size_t word = value_idx >> 6U;
        if (word >= result.cell_validity_bitmap.size()) return false;
        return ((result.cell_validity_bitmap[word] >> (value_idx & 63U)) & 1ULL) != 0ULL;
    };

    auto build_columnar_result = [&]() {
        if (result.tuple_size == 0 || result.data.empty()) return;
        const size_t physical_rows = result.data.size() / result.tuple_size;
        if (physical_rows == 0) return;
        result.column_data.assign(result.tuple_size, std::vector<unsigned long long>(physical_rows, 0ULL));
        result.column_i64.assign(result.tuple_size, std::vector<std::int64_t>());
        result.column_u64.assign(result.tuple_size, std::vector<std::uint64_t>());
        result.column_f64.assign(result.tuple_size, std::vector<double>());
        for (size_t col = 0; col < result.tuple_size; ++col) {
            const auto type = (col < result.columns.size()) ? result.columns[col].type : db::LogicalType::UInt64;
            if (type == db::LogicalType::Float64) result.column_f64[col].assign(physical_rows, 0.0);
            else if (type == db::LogicalType::Int64) result.column_i64[col].assign(physical_rows, 0);
            else result.column_u64[col].assign(physical_rows, 0);
        }
        const size_t words_per_col = physical_rows / 64 + (physical_rows % 64 != 0);
        result.column_validity_bitmap.assign(result.tuple_size, std::vector<uint64_t>(words_per_col, 0ULL));
        for (size_t row = 0; row < physical_rows; ++row) {
            const size_t base = row * result.tuple_size;
            for (size_t col = 0; col < result.tuple_size; ++col) {
                const size_t src_idx = base + col;
                result.column_data[col][row] = result.data[src_idx];
                const auto type = (col < result.columns.size()) ? result.columns[col].type : db::LogicalType::UInt64;
                if (type == db::LogicalType::Float64) {
                    double value = 0.0;
                    unsigned long long bits = result.data[src_idx];
                    std::memcpy(&value, &bits, sizeof(double));
                    result.column_f64[col][row] = value;
                } else if (type == db::LogicalType::Int64) {
                    result.column_i64[col][row] = static_cast<std::int64_t>(result.data[src_idx]);
                } else {
                    result.column_u64[col][row] = static_cast<std::uint64_t>(result.data[src_idx]);
                }
                if (row_valid_at(src_idx)) {
                    result.column_validity_bitmap[col][row >> 6U] |= (1ULL << (row & 63U));
                }
            }
        }
        result.has_columnar_result = true;
    };

    // Do not eagerly transpose legacy sparse buffers to columnar host vectors.
    // Sparse aggregate output is usually small to print but can have a large physical
    // hash domain; eager transpose was dominating Host result fetch time.

    if (ctx_->result_is_dense_) {
        result.row_count = ctx_->result_row_count_;
        if (result.row_count == 0 && !result.data.empty() && ctx_->expected_result_size_ != 0) {
            result.row_count = result.data.size() / result.tuple_size;
        }
    } else if (result.tuple_size <= 1) {
        result.row_count = result.data.empty() ? 0 : 1;
    } else {
        const size_t physical_rows = result.data.size() / result.tuple_size;
        size_t non_empty_rows = 0;
        auto valid_at = [&](size_t value_idx) -> bool {
            if (!result.has_cell_validity || result.cell_validity_bitmap.empty()) return false;
            const size_t word = value_idx >> 6U;
            if (word >= result.cell_validity_bitmap.size()) return false;
            return ((result.cell_validity_bitmap[word] >> (value_idx & 63U)) & 1ULL) != 0ULL;
        };
        for (size_t row = 0; row < physical_rows; ++row) {
            const size_t base = row * result.tuple_size;
            bool present = false;
            for (size_t col = 0; col < result.tuple_size; ++col) {
                const size_t idx = base + col;
                if (result.data[idx] != 0ULL || valid_at(idx)) {
                    present = true;
                    break;
                }
            }
            if (present) ++non_empty_rows;
        }
        result.row_count = non_empty_rows;
    }
    result.materialized_row_offset = 0;
    result.materialized_row_count = result.row_count;
    finalize_timing(result);
    return result;
}

std::string DatabaseInstance::generateQueryCode(const std::string& sql) {
    runtime_->wait();
    ctx_->resetForQuery();
    return engine_->generateQueryCode(sql);
}

} // namespace db
