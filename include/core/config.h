#pragma once

#include <cstddef>
#include <string>

namespace db {

struct CrystalConfig {
    std::string include_dir;
    std::string deps_include_dir;

    bool dump_generated_code = false;
    bool extended_timing = false;
    bool memory_guard_enabled = true;

    std::size_t output_row_limit = 1000;
    bool output_row_limit_enabled = true;

    std::size_t memory_guard_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
    double memory_guard_reserve_fraction = 0.10;

    std::string source_path;
};

CrystalConfig loadCrystalConfig();

} // namespace db
