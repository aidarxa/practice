#include "db/database_instance.h"
#include "core/catalog_cache.h"
#include "crystal/utils.h"

#include <iostream>
#include <iomanip>

namespace db {

DatabaseInstance::DatabaseInstance()
    : q_(sycl::gpu_selector_v,sycl::property::queue::in_order()) {
    ctx_ = std::make_unique<ExecutionContext>();
    ctx_->q_ = &q_;

    initCatalog();

    engine_ = std::make_unique<QueryEngine>(
        catalog_,
        std::make_unique<FileBasedQueryCache>(),
        std::make_unique<AdaptiveCppCompiler>(),
        std::make_unique<DynamicLibraryExecutor>()
    );

    // Создаём буфер с нулевой начальной ёмкостью.
    // QueryEngine::executeQuery вызовет ensureCapacity с точным размером
    // перед запуском каждого ядра (через expected_result_size_).
    ctx_->result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(q_, 0);
}

DatabaseInstance::~DatabaseInstance() {
    for (auto& pair : ctx_->buffers_) {
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

    // LINEORDER (fact)
    TableMetadata lo("LINEORDER",
                     {"lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey",
                      "lo_suppkey", "lo_orderdate", "lo_orderpriority",
                      "lo_shippriority", "lo_quantity", "lo_extendedprice",
                      "lo_ordtotalprice", "lo_discount", "lo_revenue",
                      "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"},
                     LO_LEN, true);
    applyStats(lo);
    catalog_->pushTableMetadata(lo);

    // SUPPLIER (dimension)
    TableMetadata s("SUPPLIER",
                    {"s_suppkey", "s_name", "s_address", "s_city", "s_nation",
                     "s_region", "s_phone"},
                    S_LEN, false);
    applyStats(s);
    catalog_->pushTableMetadata(s);

    // CUSTOMER (dimension)
    TableMetadata c("CUSTOMER",
                    {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                     "c_region", "c_phone", "c_mktsegment"},
                    C_LEN, false);
    applyStats(c);
    catalog_->pushTableMetadata(c);

    // PART (dimension)
    TableMetadata p("PART",
                    {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                     "p_color", "p_type", "p_size", "p_container"},
                    P_LEN, false);
    applyStats(p);
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
    catalog_->pushTableMetadata(d);
}

void DatabaseInstance::loadData() {
    std::cout << "Loading data to GPU..." << std::endl;
    int total_columns = 0;
    for (const auto& table : catalog_->getTablesMetadata()) {
        total_columns += table.getColumnCount();
    }

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
            q_.copy(data.data(), d_ptr, size).wait();

            std::string buf_name = "d_" + col_name;
            ctx_->buffers_[buf_name] = d_ptr;
        }
    }
    std::cout << "\nData loading complete!" << std::endl;
}

std::pair<std::vector<unsigned long long>, size_t> DatabaseInstance::executeQuery(const std::string& sql) {
    // QueryEngine::executeQuery самостоятельно вызовет ensureCapacity и zero
    // на основе рассчитанного expected_result_size_.
    engine_->executeQuery(sql, ctx_.get());

    // Ждём завершения всех GPU операций перед копированием на хост.
    q_.wait();

    // Определяем фактическое количество элементов для копирования.
    // expected_result_size_ устанавливается QueryEngine до запуска ядра.
    size_t copy_size = ctx_->expected_result_size_;
    if (copy_size == 0) copy_size = 1; // защита от нулевого размера

    // Копируем результаты на хост
    std::vector<unsigned long long> h_result;
    ctx_->result_buffer_->copyToHost(h_result, copy_size);

    return {h_result, ctx_->tuple_size_};
}

std::string DatabaseInstance::generateQueryCode(const std::string& sql) {
    return engine_->generateQueryCode(sql);
}

} // namespace db
