/** @file @brief Behavioral checks of the SSB physical column boundary. */
#include "crystal/utils.h"
#include "core/memory.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>

namespace {
/** @brief Own only this fixture's temporary files; no shared data directory is modified. */
struct Files {
    std::filesystem::path directory;
    Files() {
        auto pattern = (std::filesystem::temp_directory_path() / "ssb-stage7-XXXXXX").string();
        const auto created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("mkdtemp");
        directory = created;
    }
    ~Files() { std::filesystem::remove(directory / "column"); std::filesystem::remove(directory); }
    auto path() const { return directory / "column"; }
    void write(const std::string& bytes) {
        std::ofstream file(path(), std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!file) throw std::runtime_error("write fixture");
    }
};
template<class F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
void decoding(Files& files) {
    files.write(std::string("\x00\x00\x00\x80\xff\xff\xff\xff\x00\x00\x00\x00\xff\xff\xff\x7f", 16));
    assert(db::loadNumericColumn(files.path(), 4, db::ssbInt) == std::vector<int32_t>({INT32_MIN, -1, 0, INT32_MAX}));
    rejects([&] { db::loadNumericColumn(files.path(), 3, db::ssbInt); });
    rejects([&] { db::loadNumericColumn(files.path(), 5, db::ssbInt); });
    rejects([&] { db::loadNumericColumn(files.path(), SIZE_MAX, db::ssbInt); });
    files.write("01011");
    assert(db::loadNumericColumn(files.path(), 5, db::ssbFlag) == std::vector<int32_t>({0, 1, 0, 1, 1}));
    for (const auto invalid : {std::string("2"), std::string(" "), std::string("\0", 1), std::string("\1", 1)}) {
        files.write(invalid);
        rejects([&] { db::loadNumericColumn(files.path(), 1, db::ssbFlag); });
    }
    files.write("");
    assert(db::loadNumericColumn(files.path(), 0, db::ssbFlag).empty());
    assert(db::loadNumericColumn(files.path(), 0, db::ssbInt).empty());
    rejects([&] { db::loadNumericColumn(files.directory / "missing", 1, db::ssbInt); });
    rejects([&] { db::loadNumericColumn(files.path(), 0, {db::PhysicalEncoding::FixedText, 25}); });
    rejects([&] { db::loadNumericColumn(files.path(), 0, {db::PhysicalEncoding::Int32LE, 1}); });
    rejects([&] { db::loadNumericColumn(files.path(), 0, {db::PhysicalEncoding::AsciiBoolean, 4}); });
    rejects([&] { db::columnFileBytes(files.path(), 0, 0); });
}
void metadata() {
    TableMetadata source("SUPPLIER", db::ssbColumnNames("SUPPLIER"), 3, false);
    source.setColumnStorage("s_name", db::ssbColumn("s_name").storage);
    source.setColumnStats("s_suppkey", {0, 2, 3});
    source.setColumnPrimaryKey("s_suppkey");
    source.requireNumericColumn("s_suppkey");
    rejects([&] { source.requireNumericColumn("s_name"); });
    auto copy = source;
    auto moved = std::move(copy);
    copy = source;
    moved = std::move(copy);
    assert(moved.getColumnStorage("s_name").width == 25);
    assert(moved.isColumnPrimaryKey("s_suppkey"));
    rejects([&] { moved.requireNumericColumn("s_name"); });
}
void schema() {
    std::set<std::string> names, files;
    for (const auto& column : db::ssbColumns) {
        assert(names.insert(column.name).second);
        assert(files.insert(column.fileName()).second);
        assert(&column == &db::ssbColumn(column.name));
        assert(db::ssbColumnNames(column.table).at(column.index) == column.name);
        assert(column.storage.width > 0);
    }
    assert(!db::ssbColumn("s_city").storage.isNumeric());
    assert(!db::ssbColumn("s_name").storage.isNumeric());
    assert(db::ssbColumnNames("DDATE").size() == 17);
    rejects([] { lookup(""); });
    rejects([] { lookup("d_unknown"); });
    rejects([] { db::ssbColumnNames("missing"); });
}
}

#ifdef SSB_COMBINED_TEST
int storageChecks() {
#else
int main() {
#endif
    // The physical DDATE has two numeric fields before sellingseason and flags.
    assert(lookup("d_monthnuminyear") == "DDATE10");
    assert(lookup("d_weeknuminyear") == "DDATE11");
    assert(lookup("d_sellingseason") == "DDATE12");
    assert(lookup("d_lastdayinweekfl") == "DDATE13");
    assert(lookup("d_weekdayfl") == "DDATE16");
    Files files;
    decoding(files);
    metadata();
    schema();
    std::cout << "SSB physical storage checks passed\n";
    return 0;
}
