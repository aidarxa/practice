#include "db/query_engine.h"

#include <stdexcept>

namespace db {

std::string QueryEngine::processQuery(const std::string& sql) {
    // Заглушка: будет заменена на полный пайплайн
    // Parser → Planner → Optimizer → CodeGenerator
    if (sql.empty()) {
        throw std::runtime_error("Empty query");
    }
    return "-- [STUB] Generated SYCL code for:\n-- " + sql + "\n";
}

} // namespace db
