#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include "crystal/utils.h"
#include "crystal/load.h"
#include "crystal/pred.h"

using namespace sycl;

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

class select_kernel;

int main() {
    int num_trials = 4;
    std::chrono::duration<double> mean{};

    queue q(gpu_selector_v);

    // Host memory allocations
    int* h_lo_orderdate = malloc_host<int>(LO_LEN, q);
    int* h_lo_discount = malloc_host<int>(LO_LEN, q);
    int* h_lo_quantity = malloc_host<int>(LO_LEN, q);
    int* h_lo_extendedprice = malloc_host<int>(LO_LEN, q);
    unsigned long long* h_result = malloc_host<unsigned long long>(1, q);

    // Data loading
    std::cout << "** LOADING DATA **" << "\n";
    auto v_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    auto v_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
    auto v_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
    auto v_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);

    std::copy(v_lo_orderdate.begin(), v_lo_orderdate.end(), h_lo_orderdate);
    std::copy(v_lo_discount.begin(), v_lo_discount.end(), h_lo_discount);
    std::copy(v_lo_quantity.begin(), v_lo_quantity.end(), h_lo_quantity);
    std::copy(v_lo_extendedprice.begin(), v_lo_extendedprice.end(), h_lo_extendedprice);

    // Device memory allocations
    int* d_lo_orderdate = malloc_device<int>(LO_LEN, q);
    int* d_lo_discount = malloc_device<int>(LO_LEN, q);
    int* d_lo_quantity = malloc_device<int>(LO_LEN, q);
    int* d_lo_extendedprice = malloc_device<int>(LO_LEN, q);
    unsigned long long* d_result = malloc_device<unsigned long long>(1, q);

    // Data transfer to device
    q.memcpy(d_lo_orderdate, h_lo_orderdate, LO_LEN * sizeof(int));
    q.memcpy(d_lo_discount, h_lo_discount, LO_LEN * sizeof(int));
    q.memcpy(d_lo_quantity, h_lo_quantity, LO_LEN * sizeof(int));
    q.memcpy(d_lo_extendedprice, h_lo_extendedprice, LO_LEN * sizeof(int));
    q.wait();

    std::cout << "** LOADED DATA **" << "\n";

    for (int t = 0; t < num_trials; t++) {
        h_result[0] = 0;
        q.memcpy(d_result, h_result, sizeof(unsigned long long)).wait();

        auto start = std::chrono::high_resolution_clock::now();

        q.submit([&](handler& h) {
            int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
            
            h.parallel_for<select_kernel>(nd_range<1>(num_tiles * BLOCK_THREADS, BLOCK_THREADS), [=](nd_item<1> it) {
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];

                size_t tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id() * TILE_SIZE;
                int num_tile_items = TILE_SIZE;

                if (it.get_group_linear_id() == it.get_group_range(0) - 1) {
                    num_tile_items = LO_LEN - tile_offset;
                }

                InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

                BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset, tid, tile_offset, items, num_tile_items);
                BlockPredGT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 19930000, num_tile_items);
                BlockPredALT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 19940000, num_tile_items);

                BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_quantity + tile_offset, tid, tile_offset, items, num_tile_items);
                BlockPredALT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 25, num_tile_items);

                BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_discount + tile_offset, tid, tile_offset, items, num_tile_items);
                BlockPredAGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 1, num_tile_items);
                BlockPredALTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 3, num_tile_items);

                BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_extendedprice + tile_offset, tid, tile_offset, items2, num_tile_items);

                unsigned long long sum = 0;
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    if (tid + BLOCK_THREADS * i < num_tile_items) {
                        if (flags[i]) {
                            sum += (unsigned long long)items2[i] * items[i];
                        }
                    }
                }

                unsigned long long aggregate = reduce_over_group(it.get_group(), sum, plus<unsigned long long>());

                if (tid == 0) {
                    atomic_ref<unsigned long long, memory_order::relaxed, memory_scope::device, access::address_space::global_space>
                        atomic_revenue(d_result[0]);
                    atomic_revenue.fetch_add(aggregate);
                }
            });
        }).wait();

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = finish - start;
        if (t > 0) mean += diff;

        q.memcpy(h_result, d_result, sizeof(unsigned long long)).wait();
        std::cout << "Result: " << h_result[0] << "\n";
        std::cout << "Total time: " << diff.count() * 1000 << " ms\n";
    }

    std::cout << "Mean time: " << (mean.count() / (num_trials - 1)) * 1000 << " ms\n";

    // Free memory
    free(h_lo_orderdate, q);
    free(h_lo_discount, q);
    free(h_lo_quantity, q);
    free(h_lo_extendedprice, q);
    free(h_result, q);
    free(d_lo_orderdate, q);
    free(d_lo_discount, q);
    free(d_lo_quantity, q);
    free(d_lo_extendedprice, q);
    free(d_result, q);

    return 0;
}

