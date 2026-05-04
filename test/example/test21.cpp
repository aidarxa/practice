#include <sycl/sycl.hpp>
#include "crystal/load.h"
#include "crystal/pred.h"
#include "crystal/join.h"
#include "crystal/utils.h"

using namespace sycl;

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

class build_hashtable_d;
class build_hashtable_p;
class build_hashtable_s;
class select_kernel;

extern "C" void execute_query(
    sycl::queue& q,
    int* d_d_datekey,
    int* d_d_year,
    int* d_lo_orderdate,
    int* d_lo_partkey,
    int* d_lo_revenue,
    int* d_lo_suppkey,
    int* d_p_brand1,
    int* d_p_category,
    int* d_p_partkey,
    int* d_s_region,
    int* d_s_suppkey,
    unsigned long long* d_result
) {
    int* d_d_hash_table = sycl::malloc_device<int>(2*61130, q);
    q.memset(d_d_hash_table, 0, 2*61130 * sizeof(int));
    int* d_p_hash_table = sycl::malloc_device<int>(2*800000, q);
    q.memset(d_p_hash_table, 0, 2*800000 * sizeof(int));
    int* d_s_hash_table = sycl::malloc_device<int>(20000, q);
    q.memset(d_s_hash_table, 0, 20000 * sizeof(int));
    q.memset(d_result, 0, 21000 * sizeof(unsigned long long));

    q.submit([&](sycl::handler& h) {
        int num_tiles = (D_LEN + TILE_SIZE - 1) / TILE_SIZE;
        size_t local = BLOCK_THREADS;
        size_t global = num_tiles * BLOCK_THREADS;
        h.parallel_for<class build_hashtable_d>(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int items2[ITEMS_PER_THREAD];

            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tiles_local = (D_LEN + TILE_SIZE - 1) / TILE_SIZE;
            int num_tile_items = TILE_SIZE;
            if (it.get_group_linear_id() == num_tiles_local - 1) {
                num_tile_items = D_LEN - tile_offset;
            }

            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_d_datekey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_d_year + tile_offset, tid, tile_offset, items2, num_tile_items);
            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, items2, flags, d_d_hash_table, 2*61130, 19920101, num_tile_items);
        });
    });

    q.submit([&](sycl::handler& h) {
        int num_tiles = (P_LEN + TILE_SIZE - 1) / TILE_SIZE;
        size_t local = BLOCK_THREADS;
        size_t global = num_tiles * BLOCK_THREADS;
        h.parallel_for<class build_hashtable_p>(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int items2[ITEMS_PER_THREAD];

            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tiles_local = (P_LEN + TILE_SIZE - 1) / TILE_SIZE;
            int num_tile_items = TILE_SIZE;
            if (it.get_group_linear_id() == num_tiles_local - 1) {
                num_tile_items = P_LEN - tile_offset;
            }

            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_p_category + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockPredEq<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 1, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_p_partkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_p_brand1 + tile_offset, tid, tile_offset, items2, num_tile_items);
            BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, items2, flags, d_p_hash_table, 2*800000, 1, num_tile_items);
        });
    });

    q.submit([&](sycl::handler& h) {
        int num_tiles = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
        size_t local = BLOCK_THREADS;
        size_t global = num_tiles * BLOCK_THREADS;
        h.parallel_for<class build_hashtable_s>(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];

            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tiles_local = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
            int num_tile_items = TILE_SIZE;
            if (it.get_group_linear_id() == num_tiles_local - 1) {
                num_tile_items = S_LEN - tile_offset;
            }

            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_s_region + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockPredEq<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, 1, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_s_suppkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, d_s_hash_table, 20000, 1, num_tile_items);
        });
    });

    q.submit([&](sycl::handler& h) {
        int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
        size_t local = BLOCK_THREADS;
        size_t global = num_tiles * BLOCK_THREADS;
        h.parallel_for<class select_kernel>(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
            int items[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];
            int d_year[ITEMS_PER_THREAD];
            int p_brand1[ITEMS_PER_THREAD];
            int lo_revenue[ITEMS_PER_THREAD];

            int tid = it.get_local_linear_id();
            int tile_offset = it.get_group_linear_id() * TILE_SIZE;
            int num_tiles_local = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
            int num_tile_items = TILE_SIZE;
            if (it.get_group_linear_id() == num_tiles_local - 1) {
                num_tile_items = LO_LEN - tile_offset;
            }

            InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(flags);

            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, d_year, flags, d_d_hash_table, 2*61130, 19920101, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_partkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, p_brand1, flags, d_p_hash_table, 2*800000, 0, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_suppkey + tile_offset, tid, tile_offset, items, num_tile_items);
            BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(tid, items, flags, d_s_hash_table, 20000, 0, num_tile_items);
            BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_lo_revenue + tile_offset, tid, tile_offset, lo_revenue, num_tile_items);

            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                if (flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)) {
                    int hash = (d_year[i] - 1992) * 7 + p_brand1[i]) % 7000;
                    d_result[hash*3+0] = d_year[i];
                    d_result[hash*3+1] = p_brand1[i];
                    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_agg_0(d_result[hash*3+2]);
                    atomic_agg_0.fetch_add(lo_revenue[i]);
                }
            }
        });
    });

    q.wait();

    sycl::free(d_d_hash_table, q);
    sycl::free(d_p_hash_table, q);
    sycl::free(d_s_hash_table, q);
}