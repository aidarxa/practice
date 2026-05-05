#include "cli/terminal_app.h"
#include "db/database_instance.h"
#include <memory>
#include <iostream>

int main() {
    try {
        auto db = std::make_shared<db::DatabaseInstance>();
        db->loadData();

        db::cli::TerminalApp app(db);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}