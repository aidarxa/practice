#include "cli/meta_command_handler.h"

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace db::cli {

bool MetaCommandHandler::handle(const std::string& command, SessionContext& ctx) {
    // Разбиваем команду на токены для парсинга аргументов
    std::istringstream iss(command);
    std::string cmd;
    iss >> cmd;

    // --- Выход ---
    if (cmd == ".exit" || cmd == "\\q") {
        std::cout << "Bye!\n";
        return true;
    }

    // --- Перенаправление вывода ---
    if (cmd == "\\o") {
        std::string filename;
        if (iss >> filename) {
            ctx.output_file = filename;
            std::cout << "Output redirected to: " << filename << "\n";
        } else {
            ctx.output_file.clear();
            std::cout << "Output redirected to: stdout\n";
        }
        return false;
    }

    // --- AST echo ---
    if (cmd == "\\ast") {
        std::string arg;
        if (iss >> arg) {
            if (arg == "on") {
                ctx.echo_ast = true;
                std::cout << "AST echo: ON\n";
            } else if (arg == "off") {
                ctx.echo_ast = false;
                std::cout << "AST echo: OFF\n";
            } else {
                std::cerr << "Usage: \\ast on|off\n";
            }
        } else {
            std::cerr << "Usage: \\ast on|off\n";
        }
        return false;
    }

    // --- Logical plan echo ---
    if (cmd == "\\logical") {
        std::string arg;
        if (iss >> arg) {
            if (arg == "on") {
                ctx.echo_logical = true;
                std::cout << "Logical plan echo: ON\n";
            } else if (arg == "off") {
                ctx.echo_logical = false;
                std::cout << "Logical plan echo: OFF\n";
            } else {
                std::cerr << "Usage: \\logical on|off\n";
            }
        } else {
            std::cerr << "Usage: \\logical on|off\n";
        }
        return false;
    }

    // --- Dump Code echo ---
    if (cmd == "\\dump") {
        std::string arg;
        if (iss >> arg) {
            if (arg == "on") {
                ctx.dump_code = true;
                std::cout << "Dump code: ON\n";
            } else if (arg == "off") {
                ctx.dump_code = false;
                std::cout << "Dump code: OFF\n";
            } else {
                std::cerr << "Usage: \\dump on|off\n";
            }
        } else {
            std::cerr << "Usage: \\dump on|off\n";
        }
        return false;
    }



    // --- Extended timing output ---
    if (cmd == "\\timing") {
        std::string arg;
        if (iss >> arg) {
            if (arg == "on") {
                ctx.extended_timing = true;
                std::cout << "Extended timing: ON\n";
            } else if (arg == "off") {
                ctx.extended_timing = false;
                std::cout << "Extended timing: OFF\n";
            } else {
                std::cerr << "Usage: \\timing on|off\n";
            }
        } else {
            std::cout << "Extended timing: " << (ctx.extended_timing ? "ON" : "OFF") << "\n";
        }
        return false;
    }

    // --- Result output row limit ---
    if (cmd == "\\limit") {
        std::string arg;
        if (!(iss >> arg)) {
            if (ctx.output_row_limit_enabled) {
                std::cout << "Output row limit: " << ctx.output_row_limit << " last rows\n";
            } else {
                std::cout << "Output row limit: unlimited\n";
            }
            return false;
        }

        if (arg == "off" || arg == "all" || arg == "unlimited") {
            ctx.output_row_limit_enabled = false;
            std::cout << "Output row limit: unlimited\n";
            return false;
        }

        if (arg == "on") {
            ctx.output_row_limit_enabled = true;
            if (ctx.output_row_limit == 0) ctx.output_row_limit = 1000;
            std::cout << "Output row limit: " << ctx.output_row_limit << " last rows\n";
            return false;
        }

        try {
            std::size_t consumed = 0;
            unsigned long long value = std::stoull(arg, &consumed, 10);
            if (consumed != arg.size() || value == 0ULL) {
                std::cerr << "Usage: \\limit N|all|off (N must be positive)\n";
                return false;
            }
            ctx.output_row_limit_enabled = true;
            ctx.output_row_limit = static_cast<std::size_t>(value);
            std::cout << "Output row limit: " << ctx.output_row_limit << " last rows\n";
        } catch (const std::exception&) {
            std::cerr << "Usage: \\limit N|all|off\n";
        }
        return false;
    }

    // --- Справка ---
    if (cmd == "help") {
        printHelp();
        return false;
    }

    // --- Неизвестная команда ---
    std::cerr << "Unknown command: " << command
              << "\nType \"help\" for a list of commands.\n";
    return false;
}

void MetaCommandHandler::printHelp() {
    std::cout <<
        "Crystal-SYCL interactive console\n"
        "\n"
        "Meta-commands:\n"
        "  help              Show this help message\n"
        "  .exit  | \\q       Exit the program\n"
        "  \\o [filename]     Redirect generated code to file (no arg = stdout)\n"
        "  \\dump on|off      Toggle generated code output instead of execution\n"
        "  \\limit [N|all|off] Show only the last N result rows (default: 1000)\n"
        "  \\ast on|off       Toggle AST echo\n"
        "  \\logical on|off   Toggle logical plan echo\n"
        "\n"
        "SQL queries:\n"
        "  Enter SQL terminated by ';'. Multi-line input is supported.\n"
        "  Press Ctrl-D (EOF) to exit.\n";
}

} // namespace db::cli
