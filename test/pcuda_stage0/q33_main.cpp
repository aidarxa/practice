#include "q33.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>

namespace {
using namespace stage0;

struct Column {
    std::size_t count;
    Buffer<int> host;
    Buffer<int> device;
    Column(pcudaStream_t stream, const std::filesystem::path& path, std::size_t n)
        : count(n), host(stream, n, true), device(stream, n) {
        const auto bytes = checkedBytes(n, sizeof(int));
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("column exceeds streamsize");
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char*>(host.data), static_cast<std::streamsize>(bytes));
        if (!file || file.gcount() != static_cast<std::streamsize>(bytes))
            throw std::runtime_error("column short read: " + path.string());
        check(pcudaMemcpyAsync(device.data, host.data, bytes, pcudaMemcpyHostToDevice, stream), "upload column");
    }
};

struct Domain { int min; std::size_t slots; };
Domain keyDomain(const Column& keys) {
    const auto bounds = std::minmax_element(keys.host.data, keys.host.data + keys.count);
    const auto width = static_cast<unsigned long long>(static_cast<long long>(*bounds.second) - *bounds.first) + 1;
    if (width > 10000000 || *bounds.first < 0)
        throw std::runtime_error("benchmark key domain too large or negative");
    std::vector<bool> seen(static_cast<std::size_t>(width), false);
    for (std::size_t i = 0; i < keys.count; ++i) {
        const auto index = static_cast<std::size_t>(static_cast<long long>(keys.host.data[i]) - *bounds.first);
        if (seen[index]) throw std::runtime_error("non-unique dimension key");
        seen[index] = true;
    }
    return {*bounds.first, static_cast<std::size_t>(width)};
}
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(int) == 4, "SSB columns are int32");
        if (argc != 8) throw std::runtime_error("usage: q33 DATA DEVICE date.so customer.so supplier.so probe.so RUNS");
        const std::filesystem::path data(argv[1]);
        const auto prop = selectDevice(argv[2]);
        const int runs = std::stoi(argv[7]);
        if (runs < 2 || runs > 50 || !prop.pcudaHasAtomic64 || prop.maxThreadsPerBlock < 128 || prop.maxThreadsDim[0] < 128)
            throw std::runtime_error("invalid run count or missing device capabilities");
        const char* names[] = {"DDATE0", "DDATE4", "CUSTOMER0", "CUSTOMER3", "SUPPLIER0", "SUPPLIER3",
                               "LINEORDER5", "LINEORDER2", "LINEORDER4", "LINEORDER12"};
        std::vector<std::size_t> counts;
        std::size_t resident = 0;
        for (const auto* name : names) {
            const auto bytes = std::filesystem::file_size(data / name);
            if (!bytes || bytes % sizeof(int) || bytes > std::numeric_limits<std::size_t>::max() - resident)
                throw std::runtime_error("invalid column size");
            resident += static_cast<std::size_t>(bytes);
            counts.push_back(static_cast<std::size_t>(bytes / sizeof(int)));
            if (!q33::validCount(counts.back()) || (counts.back() / 512 + (counts.back() % 512 != 0)) > static_cast<unsigned>(prop.maxGridSize[0]))
                throw std::runtime_error("column exceeds launch limits");
        }
        if (resident > prop.totalGlobalMem / 2) throw std::runtime_error("column memory budget exceeded");
        if (counts[0] != counts[1] || counts[2] != counts[3] || counts[4] != counts[5] ||
            counts[6] != counts[7] || counts[6] != counts[8] || counts[6] != counts[9])
            throw std::runtime_error("table column lengths differ");
        Stream stream;
        std::vector<std::unique_ptr<Column>> columns;
        for (std::size_t i = 0; i < counts.size(); ++i)
            columns.push_back(std::make_unique<Column>(stream.value, data / names[i], counts[i]));
        auto& dk = *columns[0]; auto& dy = *columns[1];
        auto& ck = *columns[2]; auto& cc = *columns[3];
        auto& sk = *columns[4]; auto& sc = *columns[5];
        const auto dd = keyDomain(dk), cd = keyDomain(ck), sd = keyDomain(sk);
        for (const auto slots : {dd.slots, cd.slots, sd.slots}) {
            const auto bytes = checkedBytes(slots, sizeof(int));
            if (bytes > std::numeric_limits<std::size_t>::max() - resident)
                throw std::overflow_error("hash memory overflow");
            resident += bytes;
        }
        constexpr std::size_t groups = 6 * 2 * 2;
        const auto resultBytes = checkedBytes(groups, sizeof(unsigned long long));
        const auto budget = prop.totalGlobalMem / 2;
        if (resultBytes > budget || resident > budget - resultBytes)
            throw std::runtime_error("hash/result memory budget exceeded");
        // Bound aggregate values before launch, including a possible all-rows group.
        unsigned long long maxRevenue = 0;
        for (std::size_t i = 0; i < counts[9]; ++i) {
            const int revenue = columns[9]->host.data[i];
            if (revenue < 0) throw std::runtime_error("negative SSB revenue");
            maxRevenue = std::max(maxRevenue, static_cast<unsigned long long>(revenue));
        }
        if (maxRevenue && counts[9] > std::numeric_limits<unsigned long long>::max() / maxRevenue)
            throw std::overflow_error("aggregate bound overflow");
        Buffer<int> dTable(stream.value, dd.slots), cTable(stream.value, cd.slots), sTable(stream.value, sd.slots);
        Buffer<unsigned long long> result(stream.value, groups), hostResult(stream.value, groups, true);
        check(pcudaStreamSynchronize(stream.value), "finish data upload");
        const auto loadStart = Clock::now();
        Module<q33::BuildLaunch> date(argv[3], stream.value, "q33_date");
        Module<q33::BuildLaunch> customer(argv[4], stream.value, "q33_customer");
        Module<q33::BuildLaunch> supplier(argv[5], stream.value, "q33_supplier");
        Module<q33::ProbeLaunch> probe(argv[6], stream.value, "q33_probe");
        const auto loadMs = elapsed(loadStart);
        std::cout << std::setprecision(9) << "{\"load_ms\":" << loadMs << ",\"rows\":" << counts[6]
                  << ",\"resident_bytes\":" << resident + groups * sizeof(unsigned long long) << ",\"runs\":[";
        for (int run = 0; run < runs; ++run) {
            const auto start = Clock::now();
            check(pcudaMemsetAsync(dTable.data, 0, checkedBytes(dd.slots, sizeof(int)), stream.value), "clear date PHT");
            check(pcudaMemsetAsync(cTable.data, 0, checkedBytes(cd.slots, sizeof(int)), stream.value), "clear customer PHT");
            check(pcudaMemsetAsync(sTable.data, 0, checkedBytes(sd.slots, sizeof(int)), stream.value), "clear supplier PHT");
            check(pcudaMemsetAsync(result.data, 0, checkedBytes(groups, sizeof(unsigned long long)), stream.value), "clear result");
            check(date.launch(stream.value, {dk.device.data, dy.device.data, dTable.data, dk.count, dd.slots, dd.min, 1992, 1997}), "date build");
            check(customer.launch(stream.value, {ck.device.data, cc.device.data, cTable.data, ck.count, cd.slots, cd.min, 231, 235}), "customer build");
            check(supplier.launch(stream.value, {sk.device.data, sc.device.data, sTable.data, sk.count, sd.slots, sd.min, 231, 235}), "supplier build");
            check(probe.launch(stream.value, {columns[6]->device.data, columns[7]->device.data, columns[8]->device.data,
                  columns[9]->device.data, dTable.data, cTable.data, sTable.data, result.data, counts[6],
                  dd.slots, cd.slots, sd.slots, dd.min, cd.min, sd.min, 1992, 1997, 231, 235}), "probe aggregate");
            check(pcudaStreamSynchronize(stream.value), "finish Q3.3");
            const auto executionMs = elapsed(start);
            const auto fetchStart = Clock::now();
            check(pcudaMemcpyAsync(hostResult.data, result.data, checkedBytes(groups, sizeof(unsigned long long)),
                                   pcudaMemcpyDeviceToHost, stream.value), "fetch Q3.3");
            check(pcudaStreamSynchronize(stream.value), "await Q3.3 result");
            const auto fetchMs = elapsed(fetchStart);
            if (run) std::cout << ',';
            std::cout << "{\"execute_ms\":" << executionMs << ",\"fetch_ms\":" << fetchMs << ",\"result\":[";
            bool comma = false;
            for (std::size_t group = 0; group < groups; ++group) {
                if (!hostResult.data[group]) continue;
                if (comma) std::cout << ',';
                comma = true;
                std::cout << '[' << 1992 + group / 4 << ',' << (group % 4 / 2 ? 235 : 231)
                          << ',' << (group % 2 ? 235 : 231) << ',' << hostResult.data[group] << ']';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
