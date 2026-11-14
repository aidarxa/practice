/** @file @brief Independent CPU checks of the preserved storage view and runtime configuration. */
#include "core/config.h"
#include "crystal/utils.h"
#include "legacy_storage_view.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
class Files {
public:
    std::filesystem::path directory;
    Files() {
        const auto parent = std::filesystem::is_directory("/tmp/opencode") ? std::filesystem::path("/tmp/opencode")
                                                                         : std::filesystem::temp_directory_path();
        auto pattern = (parent / "pcuda-storage-XXXXXX").string();
        auto* created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("mkdtemp");
        directory = created;
    }
    ~Files() {
        std::filesystem::remove(directory / "column");
        std::filesystem::remove(directory / "config");
        std::filesystem::remove(directory);
    }
    void write(const char* name, const std::string& content) {
        std::ofstream file(directory / name, std::ios::binary);
        file.write(content.data(), content.size());
        if (!file) throw std::runtime_error("fixture file write");
    }
};
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
void storage(Files& files) {
    const int32_t values[] = {7,9,11};
    const std::string bytes(reinterpret_cast<const char*>(values), sizeof(values));
    files.write("column", bytes);
    assert(loadColumnFile<int32_t>(files.directory / "column", 2) == std::vector<int32_t>({7,9}));
    files.write("column", bytes.substr(0, 4));
    assert(loadColumnFile<int32_t>(files.directory / "column", 3) == std::vector<int32_t>({7,0,0}));
    files.write("column", bytes.substr(0, 5));
    std::vector<int32_t> expected(3);
    std::memcpy(expected.data(), bytes.data(), 5);
    assert(loadColumnFile<int32_t>(files.directory / "column", 3) == expected);
    rejects([&] { loadColumnFile<int32_t>(files.directory / "missing", 1); });
    rejects([&] { loadColumnFile<int32_t>(files.directory / "column", SIZE_MAX); });
    assert(nullBitmapWordCount(SIZE_MAX) == SIZE_MAX / 64 + 1);
}
void config(Files& files) {
    const auto path = (files.directory / "config").string();
    setenv("CRYSTAL_CONFIG", path.c_str(), 1);
    unsetenv("CRYSTAL_DEVICE_NAME");
    unsetenv("CRYSTAL_JIT_OVERLAP_EXPERIMENT");
    files.write("config", "device_name='Test GPU'\nmemory_guard_reserve_bytes=2MiB\nmemory_guard_reserve_fraction=0.25\n");
    auto config = db::loadCrystalConfig();
    assert(config.device_name == "Test GPU" && config.memory_guard_reserve_bytes == 2 * 1024 * 1024);
    assert(config.memory_guard_reserve_fraction == 0.25);
    assert(!config.jit_overlap_experiment);
    files.write("config","jit_overlap_experiment=on\n");
    assert(db::loadCrystalConfig().jit_overlap_experiment);
    setenv("CRYSTAL_JIT_OVERLAP_EXPERIMENT","off",1);
    assert(!db::loadCrystalConfig().jit_overlap_experiment);
    setenv("CRYSTAL_JIT_OVERLAP_EXPERIMENT","invalid",1);
    rejects([]{db::loadCrystalConfig();});
    unsetenv("CRYSTAL_JIT_OVERLAP_EXPERIMENT");
    setenv("CRYSTAL_DEVICE_NAME", "Override", 1);
    assert(db::loadCrystalConfig().device_name == "Override");
    unsetenv("CRYSTAL_DEVICE_NAME");
    for (const auto& invalid : {"memory_guard_reserve_bytes=-1", "memory_guard_reserve_bytes=18446744073709551615GiB",
                              "memory_guard_reserve_fraction=nan", "memory_guard_reserve_fraction=0.2junk",
                              "memory_guard_reserve_fraction=0.95", "jit_overlap_experiment=maybe"}) {
        files.write("config", invalid);
        rejects([] { db::loadCrystalConfig(); });
    }
}
}
int main() {
    Files files;
    storage(files);
    config(files);
    std::cout << "Storage compatibility and configuration checks passed\n";
}
