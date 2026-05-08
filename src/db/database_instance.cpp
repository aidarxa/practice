#include "db/database_instance.h"
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

    // Выделяем память для результатов
    // 21000 - достаточный размер для SSB (7000 групп * 3 элемента)
    ctx_->result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(q_, 21000);
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

    // LINEORDER (fact)
    TableMetadata lo("LINEORDER",
                     {"lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey",
                      "lo_suppkey", "lo_orderdate", "lo_orderpriority",
                      "lo_shippriority", "lo_quantity", "lo_extendedprice",
                      "lo_ordtotalprice", "lo_discount", "lo_revenue",
                      "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"},
                     LO_LEN, true);
    catalog_->pushTableMetadata(lo);

    // SUPPLIER (dimension)
    TableMetadata s("SUPPLIER",
                    {"s_suppkey", "s_name", "s_address", "s_city", "s_nation",
                     "s_region", "s_phone"},
                    S_LEN, false);
    s.setColumnStats("s_suppkey", {1, S_LEN, S_LEN});
    s.setColumnStats("s_region", {0, 4, 5});
    s.setColumnStats("s_nation", {0, 24, 25});
    catalog_->pushTableMetadata(s);

    // CUSTOMER (dimension)
    TableMetadata c("CUSTOMER",
                    {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                     "c_region", "c_phone", "c_mktsegment"},
                    C_LEN, false);
    c.setColumnStats("c_custkey", {1, C_LEN, C_LEN});
    c.setColumnStats("c_region", {0, 4, 5});
    c.setColumnStats("c_nation", {0, 24, 25});
    catalog_->pushTableMetadata(c);

    // PART (dimension)
    TableMetadata p("PART",
                    {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                     "p_color", "p_type", "p_size", "p_container"},
                    P_LEN, false);
    p.setColumnStats("p_partkey", {1, P_LEN, P_LEN});
    p.setColumnStats("p_category", {1, 25, 25});
    p.setColumnStats("p_brand1", {1, 1000, 1000});
    catalog_->pushTableMetadata(p);

    // DDATE (dimension)
    TableMetadata d("DDATE",
                    {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year",
                     "d_yearmonthnum", "d_yearmonth", "d_daynuminweek",
                     "d_daynuminmonth", "d_daynuminyear", "d_sellingseason",
                     "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl",
                     "d_weekdayfl"},
                    D_LEN, false);
    d.setColumnStats("d_datekey", {19920101, 19981230, D_LEN});
    d.setColumnStats("d_year", {1992, 1998, 7});
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
    // Очищаем буфер результатов перед каждым запуском
    ctx_->result_buffer_->ensureCapacity(21000);
    ctx_->result_buffer_->zero();

    // Запускаем JIT и ядро
    engine_->executeQuery(sql, ctx_.get());

    // Ждем завершения всех GPU операций
    q_.wait();

    // Копируем результаты на хост
    std::vector<unsigned long long> h_result;
    ctx_->result_buffer_->copyToHost(h_result, 21000);

    return {h_result, ctx_->tuple_size_};
}

std::string DatabaseInstance::generateQueryCode(const std::string& sql) {
    return engine_->generateQueryCode(sql);
}

} // namespace db
