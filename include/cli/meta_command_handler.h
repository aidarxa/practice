#pragma once

#include "session_context.h"

#include <string>

namespace db::cli {

class MetaCommandHandler {
public:
    /// Обрабатывает мета-команду (начинается с '\', '.' или "help").
    /// Возвращает true, если пользователь хочет выйти из программы.
    static bool handle(const std::string& command, SessionContext& ctx);

private:
    static void printHelp();
};

} // namespace db::cli
