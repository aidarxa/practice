#include "db/database_instance.h"
#include "core/catalog_cache.h"
#include "crystal/utils.h"
#include "core/inline_math.h"

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
      q_(sycl::gpu_selector_v,sycl::property::queue::in_order()) {
    ctx_ = std::make_unique<ExecutionContext>();
    ctx_->q_ = &q_;
    ctx_->config_ = config_;

    initCatalog();

    engine_ = std::make_unique<QueryEngine>(
        catalog_,
        std::make_unique<FileBasedQueryCache>(),
        std::make_unique<AdaptiveCppCompiler>(config_.include_dir, config_.deps_include_dir),
        std::make_unique<DynamicLibraryExecutor>()
    );

    // Создаём буфер с нулевой начальной ёмкостью.
    // QueryEngine::executeQuery вызовет ensureCapacity с точным размером
    // перед запуском каждого ядра (через expected_result_size_).
    ctx_->result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(q_, 0);
    ctx_->result_validity_buffer_ = std::make_unique<DynamicDeviceBuffer<uint64_t>>(q_, 0);
}

DatabaseInstance::~DatabaseInstance() {
    for (auto& pair : ctx_->buffers_) {
        if (pair.second) {
            sycl::free(pair.second, q_);
        }
    }
    for (auto& pair : ctx_->null_bitmaps_) {
        if (pair.second) {
            sycl::free(pair.second, q_);
        }
    }
}

void DatabaseInstance::initCatalog() {
    catalog_ = std::make_shared<Catalog>();

    // Инициализируем кэш (создаст папку meta рядом с исполняемым файлом)
    db::CatalogCacheManager cache_manager("meta/catalog_cache.txt");

    // Лямбда: берет список колонок и размер таблицы, получает стату и записывает
    auto applyStats = [&](TableMetadata& table) {
        int table_size = table.getSize(); // Возвращает LO_LEN, S_LEN и т.д.
        for (const auto& col_name : table.getColumnNames()) {
            db::CachedStats stats = cache_manager.getStatsForColumn(col_name, table_size);
            if (stats.cardinality > 0) {
                table.setColumnStats(col_name, {stats.min_val, stats.max_val, static_cast<uint64_t>(stats.cardinality)});
            }
        }
    };

    auto applyNullabilityFromStorage = [](TableMetadata& table) {
        for (const auto& col_name : table.getColumnNames()) {
            if (hasNullBitmapFile(col_name)) {
                table.setColumnNullable(col_name, true);
            }
        }
    };

    // LINEORDER (fact)
    TableMetadata lo("LINEORDER",
                     {"lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey",
                      "lo_suppkey", "lo_orderdate", "lo_orderpriority",
                      "lo_shippriority", "lo_quantity", "lo_extendedprice",
                      "lo_ordtotalprice", "lo_discount", "lo_revenue",
                      "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"},
                     LO_LEN, true);
    applyStats(lo);
    applyNullabilityFromStorage(lo);
    catalog_->pushTableMetadata(lo);

    // SUPPLIER (dimension)
    TableMetadata s("SUPPLIER",
                    {"s_suppkey", "s_name", "s_address", "s_city", "s_nation",
                     "s_region", "s_phone"},
                    S_LEN, false);
    applyStats(s);
    applyNullabilityFromStorage(s);
    s.setColumnPrimaryKey("s_suppkey");
    catalog_->pushTableMetadata(s);

    // CUSTOMER (dimension)
    TableMetadata c("CUSTOMER",
                    {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                     "c_region", "c_phone", "c_mktsegment"},
                    C_LEN, false);
    applyStats(c);
    applyNullabilityFromStorage(c);
    c.setColumnPrimaryKey("c_custkey");
    catalog_->pushTableMetadata(c);

    // PART (dimension)
    TableMetadata p("PART",
                    {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                     "p_color", "p_type", "p_size", "p_container"},
                    P_LEN, false);
    applyStats(p);
    applyNullabilityFromStorage(p);
    p.setColumnPrimaryKey("p_partkey");
    catalog_->pushTableMetadata(p);

    // DDATE (dimension)
    TableMetadata d("DDATE",
                    {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year",
                     "d_yearmonthnum", "d_yearmonth", "d_daynuminweek",
                     "d_daynuminmonth", "d_daynuminyear", "d_sellingseason",
                     "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl",
                     "d_weekdayfl"},
                    D_LEN, false);
    applyStats(d);
    applyNullabilityFromStorage(d);
    d.setColumnPrimaryKey("d_datekey");
    catalog_->pushTableMetadata(d);
}

void DatabaseInstance::loadData() {
    std::cout << "Loading data to GPU..." << std::endl;
    int total_columns = 0;
    for (const auto& table : catalog_->getTablesMetadata()) {
        total_columns += table.getColumnCount();
    }

    ctx_->loaded_device_bytes_ = 0;
    int current_column = 0;
    for (const auto& table : catalog_->getTablesMetadata()) {
        uint64_t size = table.getSize();
        for (const auto& col_name : table.getColumnNames()) {
            current_column++;
            std::cout << "\r[" << current_column << "/" << total_columns << "] Loading " << std::left << std::setw(20) << col_name << std::flush;

            // Load from disk
            std::vector<int> data = loadColumn<int>(col_name, size);

            // Allocate and copy to GPU
            int* d_ptr = sycl::malloc_device<int>(size, q_);
            ctx_->loaded_device_bytes_ += static_cast<size_t>(size) * sizeof(int);
            q_.copy(data.data(), d_ptr, size).wait();

            std::string buf_name = "d_" + col_name;
            ctx_->buffers_[buf_name] = d_ptr;

            std::vector<uint64_t> validity = loadValidityBitmapIfExists(col_name, size);
            if (!validity.empty()) {
                uint64_t* d_null_bitmap = sycl::malloc_device<uint64_t>(validity.size(), q_);
                ctx_->loaded_device_bytes_ += validity.size() * sizeof(uint64_t);
                q_.copy(validity.data(), d_null_bitmap, validity.size()).wait();
                ctx_->null_bitmaps_[buf_name] = d_null_bitmap;
            }
        }
    }
    std::cout << "\nData loading complete!" << std::endl;
}

QueryResult DatabaseInstance::executeQuery(const std::string& sql) {
    return executeQuery(sql, QueryFetchOptions{});
}

QueryResult DatabaseInstance::executeQuery(const std::string& sql, const QueryFetchOptions& fetch_options) {
    // QueryEngine::executeQuery самостоятельно вызовет ensureCapacity и zero
    // на основе рассчитанного expected_result_size_.
    engine_->executeQuery(sql, ctx_.get());

    // Ждём завершения всех GPU операций перед копированием на хост. Обычно generated
    // function уже сделал q.wait(); если нет, ожидание относится к GPU execution,
    // а не к объёму результата. Host fetch timing начинается после этого барьера.
    q_.wait();
    const auto fetch_start = Clock::now();

    QueryResult result;
    result.tuple_size = ctx_->tuple_size_ == 0 ? 1 : ctx_->tuple_size_;
    result.columns = ctx_->result_columns_;
    result.dense_result = ctx_->result_is_dense_;

    if (ctx_->result_is_columnar_) {
        result.row_count = ctx_->result_row_count_;
        result.dense_result = true;
        result.has_columnar_result = true;
        result.column_data.resize(ctx_->result_column_count_);
        result.column_i64.resize(ctx_->result_column_count_);
        result.column_u64.resize(ctx_->result_column_count_);
        result.column_f64.resize(ctx_->result_column_count_);
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
        const size_t validity_words = (rows_to_copy + 63ULL) / 64ULL;
        for (size_t col = 0; col < ctx_->result_column_count_; ++col) {
            const db::LogicalType type = (col < result.columns.size()) ? result.columns[col].type : db::LogicalType::UInt64;
            if (col < ctx_->result_column_storage_.size()) {
                const auto& storage = ctx_->result_column_storage_[col];
                if (type == db::LogicalType::Float64 && storage.f64) {
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
                        const size_t last_bit = row_offset + rows_to_copy;
                        const size_t source_words = ((last_bit + 63ULL) >> 6U) - first_word;
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
        result.timing = ctx_->timing_;
        result.timing.host_fetch_ms = elapsedMs(fetch_start, Clock::now());
        result.timing.gpu_execute_ms = result.timing.gpu_execute_ms == 0.0 ? result.timing.jit_execute_ms : result.timing.gpu_execute_ms;
        result.timing.jit_execute_ms = result.timing.gpu_execute_ms;
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
        q_.memset(d_sparse_values, 0, copy_size * sizeof(unsigned long long));
        q_.memset(d_sparse_count, 0, sizeof(unsigned long long));

        unsigned long long* d_source = ctx_->result_buffer_->data();
        const size_t tuple_size_device = result.tuple_size;
        q_.submit([&](sycl::handler& h) {
            h.parallel_for<class CompactSparseAggregateResult>(sycl::range<1>(physical_rows), [=](sycl::id<1> gid) {
                const unsigned long long row = static_cast<unsigned long long>(gid[0]);
                const unsigned long long base = row * static_cast<unsigned long long>(tuple_size_device);
                bool present = false;
                for (unsigned long long c = 0; c < static_cast<unsigned long long>(tuple_size_device); ++c) {
                    const unsigned long long idx = base + c;
                    if (d_source[idx] != 0ULL) {
                        present = true;
                        break;
                    }
                }
                if (!present) return;
                const unsigned long long out_row = db::atomic_fetch_add_ull(d_sparse_count[0], 1ULL);
                const unsigned long long out_base = out_row * static_cast<unsigned long long>(tuple_size_device);
                for (unsigned long long c = 0; c < static_cast<unsigned long long>(tuple_size_device); ++c) {
                    const unsigned long long src_idx = base + c;
                    const unsigned long long dst_idx = out_base + c;
                    d_sparse_values[dst_idx] = d_source[src_idx];
                }
            });
        });
        q_.wait();

        unsigned long long sparse_row_count = 0ULL;
        q_.memcpy(&sparse_row_count, d_sparse_count, sizeof(unsigned long long)).wait();
        result.row_count = static_cast<size_t>(sparse_row_count);
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
        const size_t values_to_copy = rows_to_copy * result.tuple_size;
        if (values_to_copy != 0) {
            auto& sparse_value_buffer = ctx_->scratch_u64_buffers_.at("__sparse_result_values");
            sparse_value_buffer->copyRangeToHost(result.data, row_offset * result.tuple_size, values_to_copy);
        }
        // Validity for fast sparse SSB aggregates is absent by design: all copied
        // cells are non-null. Nullable sparse paths use dense/columnar finalization.
        result.has_cell_validity = false;
        result.timing = ctx_->timing_;
        result.timing.host_fetch_ms = elapsedMs(fetch_start, Clock::now());
        result.timing.gpu_execute_ms = result.timing.gpu_execute_ms == 0.0 ? result.timing.jit_execute_ms : result.timing.gpu_execute_ms;
        result.timing.jit_execute_ms = result.timing.gpu_execute_ms;
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
        const size_t words_per_col = (physical_rows + 63ULL) / 64ULL;
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
    result.timing = ctx_->timing_;
    result.timing.host_fetch_ms = elapsedMs(fetch_start, Clock::now());
    result.timing.gpu_execute_ms = result.timing.gpu_execute_ms == 0.0 ? result.timing.jit_execute_ms : result.timing.gpu_execute_ms;
    result.timing.jit_execute_ms = result.timing.gpu_execute_ms;
    return result;
}

std::string DatabaseInstance::generateQueryCode(const std::string& sql) {
    return engine_->generateQueryCode(sql);
}

} // namespace db
