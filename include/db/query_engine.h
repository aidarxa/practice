#pragma once

#include <string>

namespace db {

/// Фасад для бэкенда Crystal-SYCL.
/// Принимает SQL-строку, возвращает сгенерированный SYCL C++ код.
/// Текущая реализация — заглушка; будет заменена на полный пайплайн
/// (Parser → Planner → Optimizer → CodeGenerator).
class QueryEngine {
public:
    QueryEngine() = default;

    /// @throws std::runtime_error при ошибке парсинга / кодогенерации
    std::string processQuery(const std::string& sql);
};

} // namespace db
