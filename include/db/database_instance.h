#pragma once

#include <memory>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

#include "core/execution.h"
#include "core/memory.h"

namespace db {

class DatabaseInstance {
public:
    DatabaseInstance();
    ~DatabaseInstance();

    /// Выполняет загрузку реальных данных с диска на GPU
    void loadData();

    /// Выполняет JIT-компиляцию и запуск на GPU, возвращает сырые данные ответа и tuple_size
    QueryResult executeQuery(const std::string& sql);

    /// Возвращает сгенерированный JIT-код без его компиляции
    std::string generateQueryCode(const std::string& sql);

private:
    void initCatalog();

    sycl::queue q_;
    std::shared_ptr<Catalog> catalog_;
    std::unique_ptr<ExecutionContext> ctx_;
    std::unique_ptr<QueryEngine> engine_;
};

} // namespace db
