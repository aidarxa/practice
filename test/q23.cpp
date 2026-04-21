#include <sycl/sycl.hpp>
#include "../include/load.h"
#include "../include/pred.h"
#include "../include/join.h"
#include "../include/utils.h"

#include <chrono>
using namespace sycl;

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

class select_kernel;
class build_hashtable_s;
class build_hashtable_p;
class build_hashtable_d;

int main() {
    int num_trials = 4;
    std::chrono::duration<double> mean;

    std::vector<int> h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    std::vector<int> h_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
    std::vector<int> h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
    std::vector<int> h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

    std::vector<int> h_p_partkey = loadColumn<int>("p_partkey", P_LEN);
    std::vector<int> h_p_brand1 = loadColumn<int>("p_brand1", P_LEN);

    std::vector<int> h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    std::vector<int> h_d_year = loadColumn<int>("d_year", D_LEN);

    std::vector<int> h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
    std::vector<int> h_s_region = loadColumn<int>("s_region", S_LEN);

    std::cout << "** LOADED DATA **" << "\n";

    buffer<int> b_lo_orderdate (h_lo_orderdate.data(),range<1>(LO_LEN));
    buffer<int> b_lo_partkey (h_lo_partkey.data(),range<1>(LO_LEN));
    buffer<int> b_lo_suppkey (h_lo_suppkey.data(),range<1>(LO_LEN));
    buffer<int> b_lo_revenue (h_lo_revenue.data(),range<1>(LO_LEN));

    buffer<int> b_p_partkey (h_p_partkey.data(),range<1>(P_LEN));
    buffer<int> b_p_brand1 (h_p_brand1.data(),range<1>(P_LEN));

    buffer<int> b_d_datekey (h_d_datekey.data(),range<1>(D_LEN));
    buffer<int> b_d_year (h_d_year.data(),range<1>(D_LEN));

    buffer<int> b_s_suppkey (h_s_suppkey.data(),range<1>(S_LEN));
    buffer<int> b_s_region (h_s_region.data(),range<1>(S_LEN));

    int res_size = ((1998-1992+1) * 1000);
    int res_array_size = res_size * 3;
    buffer<unsigned long long, 1> b_result{range<1>(res_array_size)};

    int d_val_len = 19981230 - 19920101 + 1;
    buffer<int> b_s_hash_table(range<1>(2*S_LEN));
    buffer<int> b_p_hash_table(range<1>(2*P_LEN));
    buffer<int> b_d_hash_table(range<1>(2*d_val_len));
    
    auto q = queue(gpu_selector_v);

    for (int t = 0; t < num_trials; t++) {
        {
            auto host_res = b_result.get_host_access(write_only);
            std::fill(host_res.begin(), host_res.end(), 0);

            auto s_ht = b_s_hash_table.get_host_access(write_only);
            std::fill(s_ht.begin(), s_ht.end(), 0);

            auto p_ht = b_p_hash_table.get_host_access(write_only);
            std::fill(p_ht.begin(), p_ht.end(), 0);

            auto d_ht = b_d_hash_table.get_host_access(write_only);
            std::fill(d_ht.begin(), d_ht.end(), 0);
        }
        using namespace std::chrono;
        high_resolution_clock::time_point start, finish;
        start = high_resolution_clock::now();

        // 1 build_hashtable_s
        q.submit([&](sycl::handler& h) {
            auto a_s_region = b_s_region.get_access<access_mode::read>(h);
            auto a_s_suppkey = b_s_suppkey.get_access<access_mode::read>(h);
            auto a_s_hash_table= b_s_hash_table.get_access<access_mode::read_write>(h);

            int num_tiles = (S_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_s>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;
                
                int num_tiles = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = S_LEN - tile_offset;
                }

                BlockLoad<decltype(a_s_region),BLOCK_THREADS,ITEMS_PER_THREAD>(a_s_region,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,3,num_tile_items);

                BlockLoad<decltype(a_s_suppkey),BLOCK_THREADS,ITEMS_PER_THREAD>(a_s_suppkey,tid,tile_offset,items,num_tile_items);
                BlockBuildSelectivePHT_1<decltype(a_s_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,a_s_hash_table,S_LEN,num_tile_items);
            });
        });
        // 2 build_hashtable_p
        q.submit([&](sycl::handler& h){
            auto a_p_partkey = b_p_partkey.get_access<access_mode::read>(h);
            auto a_p_brand1 = b_p_brand1.get_access<access_mode::read>(h);
            auto a_p_hash_table = b_p_hash_table.get_access<access_mode::read_write>(h);

            int num_tiles = (P_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_p>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;
                
                int num_tiles = (P_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = P_LEN - tile_offset;
                }

                BlockLoad<decltype(a_p_brand1),BLOCK_THREADS,ITEMS_PER_THREAD>(a_p_brand1,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,260,num_tile_items);

                BlockLoad<decltype(a_p_partkey),BLOCK_THREADS,ITEMS_PER_THREAD>(a_p_partkey,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<decltype(a_p_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items2,items,flags,a_p_hash_table,P_LEN,num_tile_items);
            });
        });
        // 3 build_hashtable_d
        q.submit([&](sycl::handler& h){
            auto a_d_datekey = b_d_datekey.get_access<access_mode::read>(h);
            auto a_d_year = b_d_year.get_access<access_mode::read>(h);
            auto a_d_hash_table = b_d_hash_table.get_access<access_mode::read_write>(h);

            int num_tiles = (D_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_d>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                
                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;

                int num_tiles = (D_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = D_LEN - tile_offset;
                }

                InitFlags<BLOCK_THREADS,ITEMS_PER_THREAD>(flags);

                BlockLoad<decltype(a_d_datekey),BLOCK_THREADS,ITEMS_PER_THREAD>(a_d_datekey,tid,tile_offset,items,num_tile_items);
                BlockLoad<decltype(a_d_year),BLOCK_THREADS,ITEMS_PER_THREAD>(a_d_year,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<decltype(a_d_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,items2,flags,a_d_hash_table,d_val_len,19920101,num_tile_items);
            });
        });

        //4 select
        q.submit([&](sycl::handler& h){
            auto a_lo_orderdate = b_lo_orderdate.get_access<access_mode::read>(h);
            auto a_lo_partkey = b_lo_partkey.get_access<access_mode::read>(h);
            auto a_lo_suppkey = b_lo_suppkey.get_access<access_mode::read>(h);
            auto a_lo_revenue = b_lo_revenue.get_access<access_mode::read>(h);

            auto a_s_hash_table = b_s_hash_table.get_access<access_mode::read>(h);
            auto a_p_hash_table = b_p_hash_table.get_access<access_mode::read>(h);
            auto a_d_hash_table = b_d_hash_table.get_access<access_mode::read>(h);

            auto a_result = b_result.get_access<access_mode::write>(h);

            int num_tiles = (LO_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<select_kernel>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                int brand[ITEMS_PER_THREAD];
                int year[ITEMS_PER_THREAD];
                int revenue[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;

                int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = LO_LEN - tile_offset;
                }

                InitFlags<BLOCK_THREADS,ITEMS_PER_THREAD>(flags);

                BlockLoad<decltype(a_lo_suppkey),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_suppkey,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_1<decltype(a_s_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,a_s_hash_table,S_LEN,num_tile_items);

                BlockLoad<decltype(a_lo_partkey),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_partkey,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<decltype(a_p_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,brand,flags,a_p_hash_table,P_LEN,num_tile_items);

                BlockLoad<decltype(a_lo_orderdate),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_orderdate,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<decltype(a_d_hash_table),BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,year,flags,a_d_hash_table,d_val_len,19920101,num_tile_items);

                BlockLoad<decltype(a_lo_revenue),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_revenue,tid,tile_offset,revenue,num_tile_items);

                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    if(flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)){
                        int hash = (brand[i] * 7 + (year[i] - 1992)) % ((1998-1992+1) * (5*5*40));
                        a_result[hash*3]= year[i];
                        a_result[hash*3+1]= brand[i];
                        sycl::atomic_ref<
                            unsigned long long,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_revenue(a_result[hash*3+2]);
                        atomic_revenue.fetch_add(revenue[i]);
                    }
                }
            });
        });
        q.wait();
        finish = high_resolution_clock::now();
        std::chrono::duration<double> diff = finish - start;
        if(t>0) mean+=diff;
        {
            auto host_res = b_result.get_host_access();
            int res_count = 0;
            for (int i=0; i<res_size; i++) {
                if (host_res[3*i] != 0) {
                std::cout << host_res[3*i] << " " << host_res[3*i + 1] << " " << reinterpret_cast<unsigned long long*>(&host_res[3*i + 2])[0]  << "\n";
                res_count += 1;
                }
            }
            std::cout << "Result count: " << res_count << "\n";
        }
        std::cout << "Total time: " << diff.count() * 1000 << " ms\n";
        
    }
    std::cout << "Mean time: " << mean.count()/3 * 1000 << " ms\n";
    return 0;
}
