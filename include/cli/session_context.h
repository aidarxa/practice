#pragma once

#include <cstddef>
#include <string>

namespace db::cli {

struct SessionContext {
    bool echo_ast     = false;
    bool echo_logical = false;
    bool dump_code    = false;
    std::string output_file;

    // Result output limit. By default the CLI prints only the last 1000
    // returned rows, matching practical DB console behaviour for large
    // analytical result sets. The total returned-row count is still reported
    // separately.
    bool output_row_limit_enabled = true;
    std::size_t output_row_limit = 1000;
};

} // namespace db::cli
