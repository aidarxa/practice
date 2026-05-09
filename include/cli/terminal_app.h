#pragma once

#include "session_context.h"
#include "db/database_instance.h"
#include <memory>

namespace db::cli {

class TerminalApp {
public:
    TerminalApp(std::shared_ptr<db::DatabaseInstance> db);

    /// Запускает REPL-цикл. Блокирующий вызов — возвращает управление
    /// только после команды выхода (.exit / \q) или EOF (Ctrl-D).
    void run();

private:
    SessionContext ctx_;
    std::shared_ptr<db::DatabaseInstance> db_;

    /// Выполняет SQL-запрос через db_ и выводит результат
    /// согласно текущему ctx_ (stdout или файл).
    void executeQuery(const std::string& sql);

    /// Определяет, является ли строка мета-командой.
    static bool isMetaCommand(const std::string& line);
};

} // namespace db::cli
