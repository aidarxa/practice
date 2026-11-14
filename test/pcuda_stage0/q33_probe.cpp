#include "q33.hpp"

extern "C" pcudaError_t q33_probe(pcudaStream_t stream, q33::ProbeArgs args) {
    if (!stream || !args.dates || !args.customers || !args.suppliers || !args.revenue ||
        !args.date_table || !args.customer_table || !args.supplier_table || !args.result ||
        args.year_last < args.year_first || args.city_first == args.city_last || !q33::validCount(args.count))
        return pcudaErrorInvalidValue;
    if (!args.count) return pcudaSuccess;
    constexpr unsigned tile = q33::block * q33::items;
    const auto grid = static_cast<unsigned>(args.count / tile + (args.count % tile != 0));
    return pcudaParallelFor(grid, q33::block, 0, stream, [=]() {
        const std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.x;
        for (unsigned item = 0; item < q33::items; ++item) {
            const auto i = base + static_cast<std::size_t>(item) * q33::block;
            if (i >= args.count) continue;
            const auto s = static_cast<long long>(args.suppliers[i]) - args.supplier_min;
            if (s < 0 || static_cast<unsigned long long>(s) >= args.supplier_slots) continue;
            const int supplier = args.supplier_table[s];
            if (supplier != args.city_first && supplier != args.city_last) continue;
            const auto c = static_cast<long long>(args.customers[i]) - args.customer_min;
            if (c < 0 || static_cast<unsigned long long>(c) >= args.customer_slots) continue;
            const int customer = args.customer_table[c];
            if (customer != args.city_first && customer != args.city_last) continue;
            const auto d = static_cast<long long>(args.dates[i]) - args.date_min;
            if (d < 0 || static_cast<unsigned long long>(d) >= args.date_slots) continue;
            const int year = args.date_table[d];
            if (year < args.year_first || year > args.year_last) continue;
            const std::size_t group = static_cast<std::size_t>(static_cast<long long>(year) - args.year_first) * 4
                                   + (customer == args.city_last ? 2 : 0) + (supplier == args.city_last ? 1 : 0);
            atomicAdd(args.result + group, static_cast<unsigned long long>(args.revenue[i]));
        }
    });
}
