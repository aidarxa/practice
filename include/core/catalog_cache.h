#pragma once
/** @file @brief Dataset-scoped numeric statistics; no global SF10 file/path assumptions. */
#include "storage/ssb_dataset.h"
#include "storage/column_loader.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace db {
struct CachedStats {
    std::uintmax_t file_size = 0;
    std::int64_t mtime = 0;
    int min_val = 0, max_val = 0;
    std::uint64_t cardinality = 0;
};
/** @brief Cache belongs to one validated native directory, with nanosecond file invalidation.
 * @details Cache persistence is optional. Readers see complete snapshots; concurrent
 * writers use private temporary files and atomic rename. No SQL module cache is added.
 */
class CatalogCacheManager {
public:
    explicit CatalogCacheManager(SsbDataset dataset) : dataset_(std::move(dataset)), path_(dataset_.directory() / ".crystal-numeric-stats-v1") { load(); }
    ~CatalogCacheManager() { if (modified_) save(); }
    CachedStats getStatsForColumn(const std::string& name, std::uint64_t rows) {
        const auto& column = ssbColumn(name);
        if (rows != dataset_.rowCount(column.table)) throw std::invalid_argument("Statistics row count differs from dataset");
        requireNumericStorage(column.storage);
        const auto file = dataset_.columnPath(name);
        const auto bytes = columnFileBytes(file, rows, column.storage.width);
        const auto time = std::filesystem::last_write_time(file).time_since_epoch().count();
        const auto found = cache_.find(name);
        if (found != cache_.end() && found->second.file_size == bytes && found->second.mtime == time) return found->second;
        std::cout << "Analyzing data for column: " << name << "...\n";
        auto stats = calculate(loadNumericColumn(file, rows, column.storage));
        stats.file_size = bytes; stats.mtime = time;
        cache_[name] = stats; modified_ = true;
        return stats;
    }
private:
    SsbDataset dataset_;
    std::filesystem::path path_;
    std::unordered_map<std::string, CachedStats> cache_;
    bool modified_ = false;
    static CachedStats calculate(const std::vector<std::int32_t>& values) {
        CachedStats stats;
        if (values.empty()) return stats;
        stats.min_val = stats.max_val = values.front();
        std::unordered_set<int> unique;
        for (const auto value : values) {
            stats.min_val = std::min(stats.min_val, value);
            stats.max_val = std::max(stats.max_val, value);
            unique.insert(value);
        }
        stats.cardinality = unique.size();
        return stats;
    }
    void load() {
        std::ifstream input(path_);
        std::string record;
        if (!std::getline(input, record) || record != "CRYSTAL_SSB_NUMERIC_STATS_1") return;
        while (std::getline(input, record)) {
            std::istringstream row(record); std::string name; CachedStats stats;
            if (!(row >> name >> stats.file_size >> stats.mtime >> stats.min_val >> stats.max_val >> stats.cardinality)) continue;
            try {
                if (ssbColumn(name).storage.isNumeric() && stats.min_val <= stats.max_val && stats.cardinality <= dataset_.rowCount(ssbColumn(name).table))
                    cache_[name] = stats;
            } catch (const std::invalid_argument&) {} // Cache is untrusted/optional, never schema input.
        }
    }
    void save() noexcept {
        try { persist(); } catch (const std::exception&) {} // Even path allocation is optional at destruction.
    }
    void persist() {
        auto pattern = (dataset_.directory() / ".crystal-stats-XXXXXX").string();
        const auto fd = ::mkstemp(pattern.data());
        if (fd < 0) return; // Read-only datasets work without a persistent statistics cache.
        ::close(fd);
        try {
            std::ofstream file(pattern);
            file << "CRYSTAL_SSB_NUMERIC_STATS_1\n";
            for (const auto& entry : cache_) {
                const auto& stats = entry.second;
                file << entry.first << ' ' << stats.file_size << ' ' << stats.mtime << ' ' << stats.min_val << ' ' << stats.max_val << ' ' << stats.cardinality << '\n';
            }
            file.close();
            if (file) std::filesystem::rename(pattern, path_);
        } catch (...) { ::unlink(pattern.c_str()); throw; }
        ::unlink(pattern.c_str());
    }
};
} // namespace db
