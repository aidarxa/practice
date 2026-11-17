/** @file @brief Host-only behavioral checks of native SSB conversion and dataset metadata. */
#include "storage/ssb_import.h"
#include "storage/text_column.h"
#include "storage/column_loader.h"
#include "core/catalog_cache.h"
#include "core/config.h"
#include "test/assertions_enabled.h"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

namespace {
struct Files {
    std::filesystem::path root;
    Files() {
        auto pattern = (std::filesystem::temp_directory_path() / "ssb-native-XXXXXX").string();
        const auto created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("mkdtemp");
        root = created;
    }
    ~Files() { std::error_code error; std::filesystem::remove_all(root, error); }
    auto input() const { return root / "tables"; }
    auto output() const { return root / "columns"; }
    void write(const std::filesystem::path& file, const std::string& data) const {
        std::ofstream out(file, std::ios::binary); out.write(data.data(), data.size());
        if (!out) throw std::runtime_error("fixture write");
    }
    std::string read(const std::filesystem::path& file) const {
        std::ifstream in(file, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }
    void tables() const {
        std::filesystem::create_directory(input());
        write(input() / "supplier.tbl", "1|Supplier#1|ул. A |UNITED KI1|UNITED KINGDOM|EUROPE|123|\n");
        write(input() / "customer.tbl", "1|Customer#1|Addr|UNITED KI5|UNITED KINGDOM|EUROPE|123|BUILDING|\n");
        write(input() / "part.tbl", "1|A part|MFGR#1|MFGR#12|MFGR#1221|RED|TYPE|7|SM BOX|\n");
        write(input() / "date.tbl", "19930106|January 6 1993|Wednesday|January|1993|199301|Jan1993|3|6|6|1|1|Winter|0|0|1|1|\n");
        write(input() / "lineorder.tbl", "1|1|1|1|1|19930101|1-URGENT|0|5|100|100|2|90|-2147483648|1|19930102|AIR|\n");
    }
};
template<class F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
std::string text(const db::HostTextColumn& column) {
    return {column.bytes.begin(), column.bytes.end()};
}
void nativeConversion() {
    Files files; files.tables();
    const auto dataset = db::importSsbTables(files.input(), files.output(), "0.01", "fixture");
    assert(dataset.rowCount("LINEORDER") == 1 && dataset.rowCount("DDATE") == 1);
    assert(dataset.scale() == "0.01" && dataset.generator() == "fixture");
    const auto loaded = db::SsbDataset::open(files.output());
    assert(loaded.directory() == std::filesystem::canonical(files.output()));
    for (const auto& column : db::ssbColumns)
        assert(std::filesystem::file_size(loaded.columnPath(column.name)) == column.storage.width);
    for (const auto* name : {"s_city", "s_nation", "s_region", "c_city", "c_nation", "c_region", "p_mfgr", "p_category", "p_brand1"})
        assert(!db::ssbColumn(name).storage.isNumeric());
    const auto load = [&](const char* name) { return db::loadTextColumn(loaded.columnPath(name), 1, db::ssbColumn(name).storage); };
    assert(text(load("s_city")) == "UNITED KI1");
    assert(text(load("s_address")) == "ул. A ");
    assert(text(load("c_mktsegment")) == "BUILDING");
    assert(text(load("p_category")) == "MFGR#12");
    assert(text(load("p_brand1")) == "MFGR#1221");
    assert(text(load("d_dayofweek")) == "Wednesday");
    assert(db::loadNumericColumn(loaded.columnPath("lo_supplycost"), 1, db::ssbInt) == std::vector<int32_t>{INT32_MIN});
    assert(db::loadNumericColumn(loaded.columnPath("d_weekdayfl"), 1, db::ssbFlag) == std::vector<int32_t>{1});
    rejects([&] { loaded.rowCount("UNKNOWN"); });
    rejects([&] { loaded.columnPath("../outside"); });
    const auto original = files.read(files.output() / db::ssbManifestName);
    rejects([&] { loaded.writeManifest(); });
    assert(files.read(files.output() / db::ssbManifestName) == original);
    rejects([&] { db::importSsbTables(files.input(), files.output(), "0.01", "fixture"); });
    assert(files.read(files.output() / db::ssbManifestName) == original);
}
void malformedInput(const char* table, const std::string& input) {
    Files files; files.tables(); files.write(files.input() / table, input);
    rejects([&] { db::importSsbTables(files.input(), files.output(), "1", "fixture"); });
    assert(!std::filesystem::exists(files.output()));
    assert(std::distance(std::filesystem::directory_iterator(files.root), std::filesystem::directory_iterator{}) == 1);
}
void invalidInputs() {
    for (const auto* value : {"2147483648", "-2147483649", "1x", "", "+1"})
        malformedInput("lineorder.tbl", std::string(value) + "|1|1|1|1|19930101|X|0|5|100|100|2|90|50|1|19930102|AIR|\n");
    malformedInput("supplier.tbl", "1|too|few|\n");
    malformedInput("supplier.tbl", "1|Name|Addr|city longer than ten|Nation|EUROPE|123|\n");
    malformedInput("supplier.tbl", "1|Name|Addr|CITY|Nation|EUROPE|123|EXTRA|\n");
    malformedInput("supplier.tbl", std::string("1|Name|x") + std::string("\0", 1) + "y|CITY|Nation|EUROPE|123|\n");
    malformedInput("supplier.tbl", "1|Name|\xc0\x80|CITY|Nation|EUROPE|123|\n");
    malformedInput("lineorder.tbl", "1|1|1|1|1|19930101|X|2|5|100|100|2|90|50|1|19930102|AIR|\n");
    malformedInput("supplier.tbl", std::string(9000, 'x') + "\n");
    malformedInput("supplier.tbl", "\n");
    Files missing; missing.tables(); std::filesystem::remove(missing.input() / "part.tbl");
    rejects([&] { db::importSsbTables(missing.input(), missing.output(), "1", "fixture"); });
    assert(!std::filesystem::exists(missing.output()));
    for (const auto* table : {"supplier.tbl", "customer.tbl", "part.tbl", "date.tbl"}) {
        Files duplicate; duplicate.tables();
        const auto original = duplicate.read(duplicate.input() / table);
        duplicate.write(duplicate.input() / table, original + original);
        rejects([&] { db::importSsbTables(duplicate.input(), duplicate.output(), "1", "fixture"); });
        assert(!std::filesystem::exists(duplicate.output()));
    }
}
void manifestFailures() {
    Files files; files.tables();
    const auto dataset = db::importSsbTables(files.input(), files.output(), "1", "fixture");
    const auto manifest = files.output() / db::ssbManifestName;
    const auto original = files.read(manifest);
    const auto changed = [&](std::string before, std::string after) {
        auto value = original; const auto at = value.find(before); assert(at != std::string::npos);
        value.replace(at, before.size(), after); files.write(manifest, value);
        rejects([&] { db::SsbDataset::open(files.output()); });
    };
    changed("CRYSTAL_SSB_NATIVE_1", "CRYSTAL_SSB_OLD_1");
    changed("table\tLINEORDER\t1", "table\tLINEORDER\t18446744073709551615");
    for (const auto* invalid : {"", "-1", "+1", "1x"})
        changed("table\tLINEORDER\t1", std::string("table\tLINEORDER\t") + invalid);
    changed("CUSTOMER7", "../../outside");
    changed("scale\t1", "scale\tnan");
    changed("scale\t1", "scale\t-1");
    changed("scale\t1", std::string("scale\t1\0extra", 13));
    files.write(manifest, original + "unexpected\n");
    rejects([&] { db::SsbDataset::open(files.output()); });
    files.write(manifest, original);
    files.write(dataset.columnPath("c_mktsegment"), "1234");
    rejects([&] { db::SsbDataset::open(files.output()); });
}
void emptyAndMetadata() {
    Files files; files.tables();
    for (const auto* name : {"supplier.tbl", "customer.tbl", "part.tbl", "date.tbl", "lineorder.tbl"}) files.write(files.input() / name, "");
    const auto dataset = db::importSsbTables(files.input(), files.output(), "2", "fixture");
    assert(dataset.rowCount("LINEORDER") == 0);
    assert(db::SsbDataset::open(files.output()).rowCount("PART") == 0);
    for (const auto* scale : {"0", "-1", "nan", "inf", "1x", "1\t2", "1e999"}) {
        Files invalid; invalid.tables();
        rejects([&] { db::importSsbTables(invalid.input(), invalid.output(), scale, "fixture"); });
        assert(!std::filesystem::exists(invalid.output()));
    }
    Files generator; generator.tables();
    rejects([&] { db::importSsbTables(generator.input(), generator.output(), "1", "x\ty"); });
    assert(!std::filesystem::exists(generator.output()));
}
void datasetStatistics() {
    Files first; first.tables();
    Files second; second.tables();
    const auto one = db::importSsbTables(first.input(), first.output(), "1", "fixture");
    second.write(second.input() / "supplier.tbl", "9|Name|Addr|CITY|Nation|ASIA|123|\n");
    const auto two = db::importSsbTables(second.input(), second.output(), "1", "fixture");
    {
        db::CatalogCacheManager cache(one);
        const auto value = cache.getStatsForColumn("s_suppkey", 1);
        assert(value.min_val == 1 && value.max_val == 1 && value.cardinality == 1);
        rejects([&] { cache.getStatsForColumn("s_city", 1); });
        rejects([&] { cache.getStatsForColumn("s_suppkey", 2); });
    }
    db::CatalogCacheManager cache(two);
    const auto value = cache.getStatsForColumn("s_suppkey", 1);
    assert(value.min_val == 9 && value.max_val == 9 && value.cardinality == 1);
    const auto stamp = std::filesystem::last_write_time(one.columnPath("s_suppkey"));
    first.write(one.columnPath("s_suppkey"), std::string("\x05\0\0\0", 4));
    std::filesystem::last_write_time(one.columnPath("s_suppkey"), stamp + std::chrono::seconds(1));
    db::CatalogCacheManager refreshed(one);
    assert(refreshed.getStatsForColumn("s_suppkey", 1).min_val == 5);
    for (const auto* invalid : {"bad header\n", "CRYSTAL_SSB_NUMERIC_STATS_1\nbroken\nunknown 4 0 0 1 1\ns_suppkey 4 0 9 1 999\n"}) {
        first.write(one.directory() / ".crystal-numeric-stats-v1", invalid);
        db::CatalogCacheManager corrupt(one);
        assert(corrupt.getStatsForColumn("s_suppkey", 1).min_val == 5);
    }
    Files readonly; readonly.tables();
    const auto source = db::importSsbTables(readonly.input(), readonly.output(), "1", "fixture");
    const auto permissions = std::filesystem::status(source.directory()).permissions();
    std::filesystem::permissions(source.directory(), std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    {
        db::CatalogCacheManager borrowed_temporary(db::SsbDataset::open(source.directory()));
        assert(borrowed_temporary.getStatsForColumn("s_suppkey", 1).min_val == 1);
    }
    std::filesystem::permissions(source.directory(), permissions);
}
void dataConfiguration() {
    Files files;
    const auto config = files.root / "crystal.conf";
    files.write(config, "data_dir = \"/a path/dataset\"\n");
    // Preserve the invoking environment, including an existing user's dataset selection.
    const auto restore = [](const char* key, const char* value) { if (value) setenv(key, value, 1); else unsetenv(key); };
    const auto* old_config = std::getenv("CRYSTAL_CONFIG"); const bool has_config = old_config;
    const std::string config_value = old_config ? old_config : "";
    const auto* old_data = std::getenv("CRYSTAL_DATA_DIR"); const bool has_data = old_data;
    const std::string data_value = old_data ? old_data : "";
    setenv("CRYSTAL_CONFIG", config.c_str(), 1); unsetenv("CRYSTAL_DATA_DIR");
    assert(db::loadCrystalConfig().data_dir == "/a path/dataset");
    setenv("CRYSTAL_DATA_DIR", files.root.c_str(), 1);
    assert(db::loadCrystalConfig().data_dir == files.root.string());
    setenv("CRYSTAL_DATA_DIR", "", 1); rejects([] { db::loadCrystalConfig(); });
    restore("CRYSTAL_CONFIG", has_config ? config_value.c_str() : nullptr);
    restore("CRYSTAL_DATA_DIR", has_data ? data_value.c_str() : nullptr);
}
}
#ifdef SSB_COMBINED_TEST
int datasetChecks() {
#else
int main() {
#endif
    nativeConversion(); invalidInputs(); manifestFailures(); emptyAndMetadata(); datasetStatistics(); dataConfiguration();
    std::cout << "Native SSB dataset conversion and manifest checks passed\n";
    return 0;
}
