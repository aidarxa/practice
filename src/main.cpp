#include "cli/terminal_app.h"
#include "db/database_instance.h"
#include <memory>
#include <iostream>

int main() {
    try {
        db::cli::StartupTiming startup;
        auto db = std::make_shared<db::DatabaseInstance>();
        startup.databaseReady();
        db->loadData();
        startup.dataReady();

        db::cli::TerminalApp app(db, &startup);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
