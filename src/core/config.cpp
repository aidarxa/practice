#include "core/config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace db {

namespace {

static std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool parseBool(const std::string& value, const std::string& key) {
    const std::string v = lower(trim(value));
    if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
    if (v == "0" || v == "false" || v == "off" || v == "no") return false;
    throw std::runtime_error("Invalid boolean value for config key '" + key + "': " + value);
}

static std::size_t parseSize(const std::string& value, const std::string& key) {
    std::string v = lower(trim(value));
    std::size_t mult = 1;
    auto ends_with = [&](const std::string& suffix) {
        return v.size() >= suffix.size() && v.compare(v.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (ends_with("gib")) { mult = 1024ULL * 1024ULL * 1024ULL; v.resize(v.size() - 3); }
    else if (ends_with("gb")) { mult = 1000ULL * 1000ULL * 1000ULL; v.resize(v.size() - 2); }
    else if (ends_with("mib")) { mult = 1024ULL * 1024ULL; v.resize(v.size() - 3); }
    else if (ends_with("mb")) { mult = 1000ULL * 1000ULL; v.resize(v.size() - 2); }
    else if (ends_with("kib")) { mult = 1024ULL; v.resize(v.size() - 3); }
    else if (ends_with("kb")) { mult = 1000ULL; v.resize(v.size() - 2); }
    v = trim(v);
    std::size_t consumed = 0;
    unsigned long long base = std::stoull(v, &consumed, 10);
    if (consumed != v.size()) {
        throw std::runtime_error("Invalid size value for config key '" + key + "': " + value);
    }
    if (base > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max() / mult)) {
        throw std::overflow_error("Config size value overflow for key '" + key + "'");
    }
    return static_cast<std::size_t>(base) * mult;
}

static bool readConfigFile(const std::string& path, CrystalConfig& cfg) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    cfg.source_path = path;
    std::string line;
    unsigned line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::size_t comment = line.find_first_of("#;");
        if (comment != std::string::npos) line.resize(comment);
        line = trim(line);
        if (line.empty()) continue;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Invalid config line " + std::to_string(line_no) + " in " + path + ": expected key=value");
        }
        std::string key = lower(trim(line.substr(0, eq)));
        std::string value = trim(line.substr(eq + 1));
        if ((value.size() >= 2) && ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }

        if (key == "crystal_include_dir" || key == "include_dir") cfg.include_dir = value;
        else if (key == "crystal_deps_include_dir" || key == "deps_include_dir") cfg.deps_include_dir = value;
        else if (key == "dump_generated_code" || key == "dump_code") cfg.dump_generated_code = parseBool(value, key);
        else if (key == "extended_timing" || key == "timing") cfg.extended_timing = parseBool(value, key);
        else if (key == "memory_guard_enabled") cfg.memory_guard_enabled = parseBool(value, key);
        else if (key == "reuse_scratch_buffers" || key == "scratch_buffer_reuse") cfg.reuse_scratch_buffers = parseBool(value, key);
        else if (key == "output_limit" || key == "output_row_limit") {
            const std::string v = lower(value);
            if (v == "off" || v == "all" || v == "unlimited") {
                cfg.output_row_limit_enabled = false;
            } else {
                cfg.output_row_limit = parseSize(value, key);
                if (cfg.output_row_limit == 0) throw std::runtime_error("output_limit must be positive or off");
                cfg.output_row_limit_enabled = true;
            }
        }
        else if (key == "memory_guard_reserve_bytes") cfg.memory_guard_reserve_bytes = parseSize(value, key);
        else if (key == "memory_guard_reserve_fraction") {
            cfg.memory_guard_reserve_fraction = std::stod(value);
            if (cfg.memory_guard_reserve_fraction < 0.0 || cfg.memory_guard_reserve_fraction > 0.9) {
                throw std::runtime_error("memory_guard_reserve_fraction must be in [0.0, 0.9]");
            }
        }
        else {
            throw std::runtime_error("Unknown config key '" + key + "' in " + path);
        }
    }
    return true;
}

} // namespace

CrystalConfig loadCrystalConfig() {
    CrystalConfig cfg;
    std::vector<std::string> candidates;
    if (const char* p = std::getenv("CRYSTAL_CONFIG")) {
        if (*p) candidates.emplace_back(p);
    }
    candidates.emplace_back("crystal.conf");
    candidates.emplace_back("crystal-sycl.conf");
    if (const char* home = std::getenv("HOME")) {
        candidates.emplace_back(std::string(home) + "/.config/crystal-sycl/config.conf");
    }

    for (const auto& path : candidates) {
        if (readConfigFile(path, cfg)) break;
    }

    // Explicit environment variables override config-file values.
    if (const char* p = std::getenv("CRYSTAL_INCLUDE_DIR")) {
        if (*p) cfg.include_dir = p;
    }
    if (const char* p = std::getenv("CRYSTAL_DEPS_INCLUDE_DIR")) {
        if (*p) cfg.deps_include_dir = p;
    }
    return cfg;
}

} // namespace db
