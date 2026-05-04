#pragma once

#include "session_context.h"
#include "db/query_engine.h"

namespace db::cli {

class TerminalApp {
public:
    TerminalApp() = default;

    /// Запускает REPL-цикл. Блокирующий вызов — возвращает управление
    /// только после команды выхода (.exit / \q) или EOF (Ctrl-D).
    void run();

private:
    SessionContext ctx_;
    db::QueryEngine engine_;

    /// Выполняет SQL-запрос через engine_ и выводит результат
    /// согласно текущему ctx_ (stdout или файл).
    void executeQuery(const std::string& sql);

    /// Определяет, является ли строка мета-командой.
    static bool isMetaCommand(const std::string& line);
};

} // namespace db::cli
