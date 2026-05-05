#pragma once

#include <string>

namespace db::cli {

struct SessionContext {
    bool echo_ast     = false;
    bool echo_logical = false;
    bool dump_code    = false;
    std::string output_file;
};

} // namespace db::cli
