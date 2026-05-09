#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <unordered_set>

// Подключаем твои утилиты для DATA_DIR, lookup и loadColumn
#include "crystal/utils.h"

namespace db {

struct CachedStats {
    uintmax_t file_size = 0;
    uintmax_t mtime = 0;
    int min_val = 0;
    int max_val = 0;
    int cardinality = 0;
};

class CatalogCacheManager {
private:
    std::string cache_file_path_;
    std::unordered_map<std::string, CachedStats> cache_;
    bool cache_modified_ = false;

    uintmax_t getFileMtime(const std::filesystem::path& p) {
        auto ftime = std::filesystem::last_write_time(p);
        return std::chrono::duration_cast<std::chrono::seconds>(
                   ftime.time_since_epoch()).count();
    }

public:
    CatalogCacheManager(const std::string& cache_file_path)
        : cache_file_path_(cache_file_path) {
        loadCache();
    }

    ~CatalogCacheManager() {
        if (cache_modified_) saveCache();
    }

    void loadCache() {
        std::ifstream file(cache_file_path_);
        if (!file.is_open()) return;

        std::string line, col_name;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            CachedStats stats;
            if (iss >> col_name >> stats.file_size >> stats.mtime 
                    >> stats.min_val >> stats.max_val >> stats.cardinality) {
                cache_[col_name] = stats;
            }
        }
    }

    void saveCache() {
        std::filesystem::path p(cache_file_path_);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(cache_file_path_);
        for (const auto& [col, stats] : cache_) {
            file << col << " " << stats.file_size << " " << stats.mtime << " " 
                 << stats.min_val << " " << stats.max_val << " " << stats.cardinality << "\n";
        }
    }

    // НОВОЕ: Принимаем num_entries напрямую из TableMetadata!
    CachedStats getStatsForColumn(const std::string& col_name, int num_entries) {
        // Формируем путь точно так же, как в твоем loadColumn
        std::string file_path_str = std::string(DATA_DIR) + lookup(col_name);
        std::filesystem::path file_path(file_path_str);
        
        if (!std::filesystem::exists(file_path)) {
            std::cerr << "Warning: File not found for column " << col_name << " at " << file_path_str << "\n";
            return CachedStats();
        }

        uintmax_t current_size = std::filesystem::file_size(file_path);
        uintmax_t current_mtime = getFileMtime(file_path);

        if (cache_.count(col_name)) {
            const auto& cached = cache_[col_name];
            if (cached.file_size == current_size && cached.mtime == current_mtime) {
                return cached;
            }
        }

        std::cout << "Analyzing data for column: " << col_name << "...\n";
        CachedStats new_stats;
        new_stats.file_size = current_size;
        new_stats.mtime = current_mtime;
        
        // ВЫЗЫВАЕМ ТВОЙ СТАНДАРТНЫЙ loadColumn!
        std::vector<int> data = loadColumn<int>(col_name, num_entries);

        if (!data.empty()) {
            int min_v = std::numeric_limits<int>::max();
            int max_v = std::numeric_limits<int>::lowest();
            std::unordered_set<int> unique_vals;

            for (int val : data) {
                if (val < min_v) min_v = val;
                if (val > max_v) max_v = val;
                unique_vals.insert(val);
            }

            new_stats.min_val = min_v;
            new_stats.max_val = max_v;
            new_stats.cardinality = unique_vals.size();
        }

        cache_[col_name] = new_stats;
        cache_modified_ = true;
        return new_stats;
    }
};

} // namespace db