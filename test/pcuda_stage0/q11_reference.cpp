// Independent CPU oracle for the different Q1.1 date bounds in SQL and the SF10 reference.
// No project headers, SQL generator, GPU runtime or primitive implementation involved.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void add(std::uint64_t& sum, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - sum)
        throw std::overflow_error("oracle sum overflow");
    sum += value;
}
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(std::int32_t) == 4, "SSB binary columns require 32-bit integers");
        if (argc != 2) throw std::runtime_error("usage: q11_reference DATA_SF10_DIRECTORY");
        const std::filesystem::path directory(argv[1]);
        // Physical storage order: orderdate, discount, quantity, extendedprice.
        const std::array<const char*, 4> names = {"LINEORDER5", "LINEORDER11", "LINEORDER8", "LINEORDER9"};
        std::array<std::ifstream, 4> files;
        std::uintmax_t fileBytes = 0;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto path = directory / names[i];
            const auto bytes = std::filesystem::file_size(path);
            if (i == 0) fileBytes = bytes;
            if (!bytes || bytes != fileBytes || bytes % sizeof(std::int32_t))
                throw std::runtime_error("oracle column size mismatch");
            files[i].open(path, std::ios::binary);
            if (!files[i]) throw std::runtime_error("cannot open oracle column");
        }
        constexpr std::size_t chunk = 65536;
        std::array<std::vector<std::int32_t>, 4> data;
        for (auto& column : data) column.resize(chunk);
        std::uint64_t sql = 0, year1993 = 0, boundary = 0;
        const auto rows = fileBytes / sizeof(std::int32_t);
        for (std::uintmax_t offset = 0; offset < rows;) {
            const auto count = static_cast<std::size_t>(std::min<std::uintmax_t>(chunk, rows - offset));
            const auto bytes = static_cast<std::streamsize>(count * sizeof(std::int32_t));
            for (std::size_t column = 0; column < files.size(); ++column) {
                files[column].read(reinterpret_cast<char*>(data[column].data()), bytes);
                if (files[column].gcount() != bytes) throw std::runtime_error("oracle short read");
            }
            for (std::size_t i = 0; i < count; ++i) {
                const auto date = data[0][i], discount = data[1][i], quantity = data[2][i], price = data[3][i];
                if (date < 19930101 || date > 19940101 || discount < 1 || discount > 3 || quantity >= 25)
                    continue;
                if (price < 0 || quantity < 0) throw std::runtime_error("unexpected negative SSB input");
                const auto contribution = static_cast<std::uint64_t>(price) * discount;
                add(sql, contribution);
                if (date < 19940101) add(year1993, contribution);
                else add(boundary, contribution);
            }
            offset += count;
        }
        std::cout << "{\"rows\":" << rows << ",\"sql_q11\":" << sql
                  << ",\"year_1993\":" << year1993 << ",\"january_1_1994\":" << boundary << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
