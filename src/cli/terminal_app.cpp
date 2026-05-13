#include "cli/terminal_app.h"
#include "cli/meta_command_handler.h"

#include <linenoise.h>

#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

namespace db::cli {

namespace {
using Clock = std::chrono::steady_clock;
static double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
static void printTimingLine(const char* label, double ms) {
    std::cout << label << ": " << std::fixed << std::setprecision(3) << ms << " ms\n";
}
}


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

static bool resultCellIsValid(const db::QueryResult& result, size_t value_idx) {
    if (!result.has_cell_validity || result.cell_validity_bitmap.empty()) return true;
    const size_t word = value_idx >> 6U;
    if (word >= result.cell_validity_bitmap.size()) return true;
    return ((result.cell_validity_bitmap[word] >> (value_idx & 63U)) & 1ULL) != 0ULL;
}

static bool resultColumnCellIsValid(const db::QueryResult& result, size_t row, size_t col) {
    if (!result.has_columnar_result || col >= result.column_validity_bitmap.size()) {
        return resultCellIsValid(result, row * result.tuple_size + col);
    }
    const auto& bitmap = result.column_validity_bitmap[col];
    if (bitmap.empty()) return true;
    const size_t word = row >> 6U;
    if (word >= bitmap.size()) return false;
    return ((bitmap[word] >> (row & 63U)) & 1ULL) != 0ULL;
}

static unsigned long long resultCellRaw(const db::QueryResult& result, size_t row, size_t col) {
    if (result.has_columnar_result && col < result.column_data.size() && row < result.column_data[col].size()) {
        return result.column_data[col][row];
    }
    return result.data[row * result.tuple_size + col];
}

std::string formatResultValue(unsigned long long raw,
                              const std::vector<db::ResultColumnDesc>& cols,
                              size_t col_idx,
                              bool valid) {
    if (!valid) return "NULL";
    if (col_idx < cols.size() && cols[col_idx].type == db::LogicalType::Float64) {
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(raw), "double and ULL must have equal size");
        std::memcpy(&value, &raw, sizeof(value));
        std::ostringstream oss;
        oss << std::setprecision(15) << value;
        return oss.str();
    }
    return std::to_string(raw);
}


std::string formatResultCell(const db::QueryResult& result, size_t row, size_t col) {
    const bool valid = resultColumnCellIsValid(result, row, col);
    if (!valid) return "NULL";
    const auto type = (col < result.columns.size()) ? result.columns[col].type : db::LogicalType::UInt64;
    std::ostringstream oss;
    if (type == db::LogicalType::Float64) {
        if (col < result.column_f64.size() && row < result.column_f64[col].size()) {
            oss << std::setprecision(15) << result.column_f64[col][row];
            return oss.str();
        }
    } else if (type == db::LogicalType::Int64) {
        if (col < result.column_i64.size() && row < result.column_i64[col].size()) {
            return std::to_string(result.column_i64[col][row]);
        }
    } else {
        if (col < result.column_u64.size() && row < result.column_u64[col].size()) {
            return std::to_string(result.column_u64[col][row]);
        }
    }
    return formatResultValue(resultCellRaw(result, row, col), result.columns, col, valid);
}

} // anonymous namespace

TerminalApp::TerminalApp(std::shared_ptr<db::DatabaseInstance> db)
    : db_(std::move(db)) {
    if (db_) ctx_.applyConfig(db_->config());
}

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
    const auto total_start = Clock::now();
    try {
        if (ctx_.dump_code) {
            std::string code = db_->generateQueryCode(sql);
            if (ctx_.output_file.empty()) {
                std::cout << "--- Generated Code ---\n"
                          << code
                          << "--- End ---\n";
            } else {
                std::ofstream ofs(ctx_.output_file, std::ios::app);
                if (!ofs.is_open()) {
                    std::cerr << "[ERROR] Cannot open file: " << ctx_.output_file << "\n";
                    return;
                }
                ofs << code;
                std::cout << "Code saved to " << ctx_.output_file << "\n";
            }
        } else {
            // Выполняем реальный запрос
            auto query_result = db_->executeQuery(sql);
            const std::vector<unsigned long long>& result = query_result.data;
            size_t tuple_size = query_result.tuple_size;
            const auto& columns = query_result.columns;

            auto is_non_empty_tuple = [&](size_t row) -> bool {
                if (query_result.dense_result) return row < query_result.row_count;
                if (tuple_size <= 1) return row == 0 && !result.empty();
                const size_t base = row * tuple_size;
                if (base + tuple_size > result.size()) return false;
                for (size_t j = 0; j < tuple_size; ++j) {
                    const size_t value_idx = base + j;
                    if (result[value_idx] != 0ULL || resultCellIsValid(query_result, value_idx)) return true;
                }
                return false;
            };

            std::vector<size_t> rows_to_print;
            const size_t limit = ctx_.output_row_limit_enabled ? ctx_.output_row_limit
                                                               : static_cast<size_t>(-1);
            if (query_result.dense_result) {
                const size_t total = query_result.row_count;
                const size_t begin = (ctx_.output_row_limit_enabled && total > limit) ? (total - limit) : 0;
                for (size_t row = begin; row < total; ++row) rows_to_print.push_back(row);
            } else if (tuple_size <= 1) {
                if (!result.empty() && query_result.row_count != 0) rows_to_print.push_back(0);
            } else {
                const size_t physical_rows = tuple_size == 0 ? 0 : result.size() / tuple_size;
                for (size_t row = 0; row < physical_rows; ++row) {
                    if (!is_non_empty_tuple(row)) continue;
                    if (ctx_.output_row_limit_enabled && rows_to_print.size() == limit) {
                        rows_to_print.erase(rows_to_print.begin());
                    }
                    rows_to_print.push_back(row);
                }
            }

            std::cout << "--- Result ---\n";
            if (rows_to_print.empty()) {
                std::cout << "0\n";
            } else {
                for (size_t row : rows_to_print) {
                    const size_t base = row * tuple_size;
                    if (tuple_size <= 1) {
                        std::cout << formatResultCell(query_result, row, 0) << "\n";
                        continue;
                    }
                    std::cout << "Row " << row << ": ";
                    for (size_t j = 0; j < tuple_size; ++j) {
                        std::cout << formatResultCell(query_result, row, j);
                        if (j < tuple_size - 1) std::cout << " | ";
                    }
                    std::cout << "\n";
                }
            }
            std::cout << "Rows returned: " << query_result.row_count << "\n";
            if (ctx_.output_row_limit_enabled && query_result.row_count > ctx_.output_row_limit) {
                std::cout << "Rows shown: " << rows_to_print.size() << "\n";
            }
            printTimingLine("GPU execution + host fetch time", query_result.timing.gpu_and_host_fetch_ms);
            if (ctx_.extended_timing) {
                printTimingLine("Code generation time", query_result.timing.codegen_ms);
                printTimingLine("ACPP compilation time", query_result.timing.compile_ms);
                printTimingLine("Library load + execution start time", query_result.timing.library_load_ms);
                printTimingLine("Generated function execution time", query_result.timing.jit_execute_ms);
                printTimingLine("Host result fetch time", query_result.timing.host_fetch_ms);
                printTimingLine("Engine processing time", query_result.timing.total_engine_ms);
                printTimingLine("Total query processing time", elapsedMs(total_start, Clock::now()));
            }
            std::cout << "--- End ---\n";
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
