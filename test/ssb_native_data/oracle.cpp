/** @file @brief Independent streaming CPU oracle: no project, SQL, GPU or converter headers.
 * @details Reads physical columns using an independently written schema. Optional
 * original dbgen tables are checked byte-for-byte against every converted field.
 * Results use original Text, and preserve the SQL/manual Q1.1 date distinction.
 */
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {
using Cell = std::variant<std::int64_t, std::string>;
using Row = std::vector<Cell>;
using Groups = std::map<Row, std::int64_t>;
std::int64_t integer(const Cell& cell) { return std::get<std::int64_t>(cell); }
const std::string& text(const Cell& cell) { return std::get<std::string>(cell); }
void add(std::int64_t& target, std::int64_t value) {
    if ((value > 0 && target > INT64_MAX - value) || (value < 0 && target < INT64_MIN - value)) throw std::overflow_error("CPU oracle sum overflow");
    target += value;
}
std::int64_t number(std::string_view value) {
    std::int32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::runtime_error("Invalid independent source integer");
    return result;
}
std::string quote(const std::string& value) {
    const char* hex = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') { result += '\\'; result += byte; }
        else if (byte < 32) { result += "\\u00"; result += hex[byte >> 4]; result += hex[byte & 15]; }
        else result += byte;
    }
    return result + '"';
}
void printCell(const Cell& cell) {
    if (std::holds_alternative<std::string>(cell)) std::cout << quote(text(cell)); else std::cout << integer(cell);
}
/** @brief Zero means Int32LE; -1 means ASCII flag; positive widths are native Text. */
class Reader {
public:
    Reader(const std::filesystem::path& data, const std::filesystem::path& tables, std::string name,
           std::string source, std::vector<int> widths) : name_(std::move(name)), widths_(std::move(widths)) {
        if (!tables.empty()) { raw_.open(tables / source); if (!raw_) throw std::runtime_error("Missing raw source " + source); }
        for (std::size_t index = 0; index < widths_.size(); ++index) {
            const auto path = data / (name_ + std::to_string(index));
            const auto bytes = std::filesystem::file_size(path), width = physicalWidth(widths_[index]);
            if (bytes % width) throw std::runtime_error("Incomplete independent column");
            const auto count = bytes / width;
            if (index && count != rows) throw std::runtime_error("Independent row counts differ");
            rows = count; files_.emplace_back(path, std::ios::binary);
            if (!files_.back()) throw std::runtime_error("Missing independent column");
        }
    }
    bool next(Row& result) {
        if (position_ == rows) {
            if (raw_.is_open() && raw_.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Extra original source rows");
            return false;
        }
        std::string line;
        if (raw_.is_open() && !std::getline(raw_, line)) throw std::runtime_error("Missing original source row");
        std::string_view remaining(line);
        result.clear();
        for (std::size_t index = 0; index < widths_.size(); ++index) {
            const auto kind = widths_[index]; const auto width = physicalWidth(kind);
            std::string bytes(width, '\0');
            if (!files_[index].read(bytes.data(), width)) throw std::runtime_error("Independent column read failure");
            if (raw_.is_open()) {
                const auto pipe = remaining.find('|');
                if (pipe == std::string_view::npos) throw std::runtime_error("Invalid original source fields");
                verify(bytes, remaining.substr(0, pipe), kind); remaining.remove_prefix(pipe + 1);
            }
            result.push_back(decode(bytes, kind));
        }
        if (!remaining.empty()) throw std::runtime_error("Extra original source fields");
        ++position_; return true;
    }
    std::uint64_t rows = 0;
private:
    std::string name_;
    std::vector<int> widths_;
    std::vector<std::ifstream> files_;
    std::ifstream raw_;
    std::uint64_t position_ = 0;
    static std::size_t physicalWidth(int kind) { return kind > 0 ? kind : kind < 0 ? 1 : 4; }
    static Cell decode(const std::string& bytes, int kind) {
        if (kind > 0) return bytes.substr(0, bytes.find('\0'));
        if (kind < 0) {
            if (bytes[0] != '0' && bytes[0] != '1') throw std::runtime_error("Invalid independent flag");
            return std::int64_t(bytes[0] - '0');
        }
        std::uint64_t word = 0;
        for (unsigned byte = 0; byte < 4; ++byte) word |= std::uint64_t(static_cast<unsigned char>(bytes[byte])) << (8 * byte);
        return word > INT32_MAX ? static_cast<std::int64_t>(word) - 4294967296LL : static_cast<std::int64_t>(word);
    }
    static void verify(const std::string& bytes, std::string_view value, int kind) {
        std::string expected(bytes.size(), '\0');
        if (kind > 0) {
            if (value.size() > bytes.size()) throw std::runtime_error("Original source Text exceeds width");
            std::copy(value.begin(), value.end(), expected.begin());
        } else if (kind < 0) {
            if (value != "0" && value != "1") throw std::runtime_error("Invalid original source flag");
            expected[0] = value[0];
        } else {
            const auto word = static_cast<std::uint32_t>(number(value));
            for (unsigned byte = 0; byte < 4; ++byte) expected[byte] = static_cast<char>((word >> (8 * byte)) & 255);
        }
        if (bytes != expected) throw std::runtime_error("Original table/converted column byte mismatch");
    }
};
struct Person { std::string city, nation, region, segment; };
struct Part { std::string manufacturer, category, brand; };
struct Date { std::int64_t year, month; };
template<class T> const T* find(const std::unordered_map<std::int64_t, T>& values, std::int64_t key) {
    const auto found = values.find(key); return found == values.end() ? nullptr : &found->second;
}
bool city(const std::string& value) { return value == "UNITED KI1" || value == "UNITED KI5"; }
bool manufacturer(const Part& part) { return part.manufacturer == "MFGR#1" || part.manufacturer == "MFGR#2"; }
bool between(std::int64_t value, std::int64_t low, std::int64_t high) { return value >= low && value <= high; }
struct Oracle {
    std::unordered_map<std::int64_t, Person> customers, suppliers;
    std::unordered_map<std::int64_t, Part> parts;
    std::unordered_map<std::int64_t, Date> dates;
    std::map<std::string, Groups> groups;
    std::array<std::int64_t, 4> scalar{};
    std::array<bool, 4> scalar_present{};
    void consume(const Row& row) {
        const auto when = integer(row[5]), discount = integer(row[11]), quantity = integer(row[8]);
        const auto value = integer(row[9]) * discount;
        const std::array<bool, 4> selected{{between(when,19930101,19940101) && between(discount,1,3) && quantity<25,
            between(when,19930101,19931231) && between(discount,1,3) && between(quantity,0,24),
            between(when,19940101,19940131) && between(discount,4,6) && between(quantity,26,35),
            between(when,19940204,19940210) && between(discount,5,7) && between(quantity,26,35)}};
        for (unsigned i = 0; i < selected.size(); ++i) if (selected[i]) { add(scalar[i], value); scalar_present[i] = true; }
        const auto* date = find(dates, when);
        if (!date) return;
        const auto* customer = find(customers, integer(row[2])); const auto* supplier = find(suppliers, integer(row[4]));
        const auto* part = find(parts, integer(row[3]));
        const auto revenue = integer(row[12]), profit = revenue - integer(row[13]);
        const auto year = date->year;
        if (supplier && part) {
            if (part->category == "MFGR#12" && supplier->region == "AMERICA") add(groups["q21"][{year,part->brand}],revenue);
            if (part->brand >= "MFGR#2221" && part->brand <= "MFGR#2228" && supplier->region == "ASIA") add(groups["q22"][{year,part->brand}],revenue);
            if (part->brand == "MFGR#2221" && supplier->region == "EUROPE") add(groups["q23"][{year,part->brand}],revenue);
        }
        if (!customer || !supplier) return;
        if (between(year,1992,1997)) {
            if (customer->region == "ASIA" && supplier->region == "ASIA") add(groups["q31"][{year,customer->nation,supplier->nation}],revenue);
            if (customer->nation == "UNITED STATES" && supplier->nation == "UNITED STATES") add(groups["q32"][{year,customer->city,supplier->city}],revenue);
            if (city(customer->city) && city(supplier->city)) add(groups["q33"][{year,customer->city,supplier->city}],revenue);
        }
        if (date->month == 199712 && city(customer->city) && city(supplier->city)) add(groups["q34"][{year,customer->city,supplier->city}],revenue);
        if (!part || customer->region != "AMERICA") return;
        if (manufacturer(*part) && supplier->region == "AMERICA") {
            add(groups["q41"][{year,customer->nation}],profit);
            if (year == 1997 || year == 1998) add(groups["q42"][{year,supplier->nation,part->category}],profit);
        }
        if ((year == 1997 || year == 1998) && supplier->nation == "UNITED STATES" && part->category == "MFGR#14")
            add(groups["q43"][{year,supplier->city,part->brand}],profit);
    }
    void print(bool manual) const {
        std::cout << '{';
        for (unsigned index = 0; index < 3; ++index) {
            if (index) std::cout << ',';
            const auto slot = index ? index + 1 : manual ? 1 : 0;
            std::cout << quote("q1" + std::to_string(index + 1)) << ":[[";
            if (!manual && !scalar_present[slot]) std::cout << "null"; else std::cout << scalar[slot];
            std::cout << "]]";
        }
        for (const auto* query : {"q21","q22","q23","q31","q32","q33","q34","q41","q42","q43"}) {
            std::cout << ',' << quote(query) << ":["; bool first = true;
            const auto found = groups.find(query);
            if (found != groups.end()) for (const auto& group : found->second) {
                if (!first) std::cout << ','; first = false; std::cout << '[';
                for (const auto& cell : group.first) { printCell(cell); std::cout << ','; }
                std::cout << group.second << ']';
            }
            std::cout << ']';
        }
        std::cout << '}';
    }
};
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::invalid_argument("Usage: ssb_reference DATASET [ORIGINAL_TABLES]");
        const std::filesystem::path data(argv[1]), tables = argc == 3 ? argv[2] : "";
        Oracle oracle; Row row; std::map<std::string,std::uint64_t> counts;
        for (const auto* table : {"SUPPLIER", "CUSTOMER"}) {
            const bool customer = std::string(table) == "CUSTOMER";
            std::vector<int> widths{0,25,25,10,15,12,15}; if (customer) widths.push_back(10);
            Reader reader(data,tables,table,customer ? "customer.tbl" : "supplier.tbl",widths);
            auto& values = customer ? oracle.customers : oracle.suppliers;
            while (reader.next(row)) {
                if (!values.emplace(integer(row[0]), Person{text(row[3]),text(row[4]),text(row[5]),customer ? text(row[7]) : ""}).second)
                    throw std::runtime_error("Independent duplicate dimension key");
            }
            counts[table] = reader.rows;
        }
        Reader parts(data,tables,"PART","part.tbl",{0,22,6,7,10,11,25,0,10});
        while (parts.next(row)) if (!oracle.parts.emplace(integer(row[0]),Part{text(row[2]),text(row[3]),text(row[4])}).second) throw std::runtime_error("Duplicate part");
        counts["PART"] = parts.rows;
        Reader dates(data,tables,"DDATE","date.tbl",{0,18,9,9,0,0,7,0,0,0,0,0,12,-1,-1,-1,-1});
        while (dates.next(row)) if (!oracle.dates.emplace(integer(row[0]),Date{integer(row[4]),integer(row[5])}).second) throw std::runtime_error("Duplicate date");
        counts["DDATE"] = dates.rows;
        Reader facts(data,tables,"LINEORDER","lineorder.tbl",{0,0,0,0,0,0,16,-1,0,0,0,0,0,0,0,0,10});
        while (facts.next(row)) oracle.consume(row);
        counts["LINEORDER"] = facts.rows;
        std::cout << "{\"raw_columns_verified\":" << (tables.empty() ? "false" : "true") << ",\"row_counts\":{";
        bool first = true;
        for (const auto& count : counts) { if (!first) std::cout << ','; first = false; std::cout << quote(count.first) << ':' << count.second; }
        std::cout << "},\"sql\":"; oracle.print(false); std::cout << ",\"manual\":"; oracle.print(true); std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "Independent SSB oracle: " << error.what() << '\n'; return 1; }
}
