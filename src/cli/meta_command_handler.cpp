#include "cli/meta_command_handler.h"

#include <iostream>
#include <sstream>

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
        "  \\ast on|off       Toggle AST echo\n"
        "  \\logical on|off   Toggle logical plan echo\n"
        "\n"
        "SQL queries:\n"
        "  Enter SQL terminated by ';'. Multi-line input is supported.\n"
        "  Press Ctrl-D (EOF) to exit.\n";
}

} // namespace db::cli
