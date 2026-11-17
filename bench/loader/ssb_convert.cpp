/** @file @brief Small host-only entry point used by prepare_ssb.sh. */
#include "storage/ssb_import.h"
#include <iostream>

namespace {
void show(const db::SsbDataset& dataset) {
    std::cout << "Native SSB dataset: " << dataset.directory() << "\nScale: " << dataset.scale() << '\n';
    for (const auto& table : db::ssbTables) std::cout << table.name << ": " << dataset.rowCount(table.name) << " rows\n";
}
}
int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "import" && argc == 6) show(db::importSsbTables(argv[2], argv[3], argv[4], argv[5]));
        else if (mode == "verify" && argc == 3) show(db::SsbDataset::open(argv[2]));
        else if (mode == "scale" && argc == 3) db::validateSsbScale(argv[2]);
        else throw std::invalid_argument("Usage: ssb_convert import INPUT OUTPUT SCALE GENERATOR | verify DATASET | scale SCALE");
        return 0;
    } catch (const std::exception& error) { std::cerr << "SSB: " << error.what() << '\n'; return 1; }
}
