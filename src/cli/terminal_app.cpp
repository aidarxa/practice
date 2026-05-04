#include "cli/terminal_app.h"
#include "cli/meta_command_handler.h"

#include <linenoise.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

namespace db::cli {

namespace {

/// Удаляет пробельные символы с обоих концов строки.
std::string trim(const std::string& s) {
    auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

/// Проверяет, содержит ли строка символ ';' (завершение SQL-запроса).
bool containsSemicolon(const std::string& s) {
    return s.find(';') != std::string::npos;
}

} // anonymous namespace

void TerminalApp::run() {
    // Настраиваем linenoise
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(256);

    std::cout << "Crystal-SYCL interactive console. Type \"help\" for help.\n\n";

    while (true) {
        char* raw = linenoise("crystal> ");
        if (raw == nullptr) {
            // EOF (Ctrl-D)
            std::cout << "\n";
            break;
        }

        std::string line = trim(std::string(raw));
        linenoiseFree(raw);

        if (line.empty()) {
            continue;
        }

        // --- Мета-команда: обрабатываем сразу, без ожидания ';' ---
        if (isMetaCommand(line)) {
            linenoiseHistoryAdd(line.c_str());
            if (MetaCommandHandler::handle(line, ctx_)) {
                break;  // .exit или \q
            }
            continue;
        }

        // --- SQL-запрос: накапливаем строки до ';' ---
        std::string buffer = line;

        while (!containsSemicolon(buffer)) {
            raw = linenoise("      -> ");
            if (raw == nullptr) {
                // EOF в середине ввода — отменяем неполный запрос
                std::cout << "\nQuery cancelled.\n";
                buffer.clear();
                break;
            }

            std::string continuation = trim(std::string(raw));
            linenoiseFree(raw);

            if (!continuation.empty()) {
                buffer += " " + continuation;
            }
        }

        if (buffer.empty()) {
            continue;
        }

        // Добавляем полную склеенную команду в историю
        linenoiseHistoryAdd(buffer.c_str());

        executeQuery(buffer);
    }
}

void TerminalApp::executeQuery(const std::string& sql) {
    try {
        std::string code = engine_.processQuery(sql);

        if (ctx_.output_file.empty()) {
            // Вывод в консоль с разделителями
            std::cout << "--- Generated Code ---\n"
                      << code
                      << "--- End ---\n";
        } else {
            // Дописываем в файл
            std::ofstream ofs(ctx_.output_file, std::ios::app);
            if (!ofs.is_open()) {
                std::cerr << "[ERROR] Cannot open file: " << ctx_.output_file << "\n";
                return;
            }
            ofs << code;
            std::cout << "Code saved to " << ctx_.output_file << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
    }
}

bool TerminalApp::isMetaCommand(const std::string& line) {
    if (line.empty()) return false;
    if (line[0] == '\\' || line[0] == '.') return true;
    if (line == "help") return true;
    return false;
}

} // namespace db::cli
