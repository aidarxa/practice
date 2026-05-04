#include <sycl/sycl.hpp>
#include "crystal/load.h"
#include "crystal/pred.h"
#include "crystal/join.h"
#include "crystal/utils.h"

#include <iostream>
#include <chrono>
#include <vector>
using namespace sycl;

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

class select_kernel;
class build_hashtable_s;
class build_hashtable_c;
class build_hashtable_p;
class build_hashtable_d;

int main() {
    queue q(gpu_selector_v, property::queue::in_order());
    int num_trials = 4;
    std::chrono::duration<double> mean;

    auto v_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    int* h_lo_orderdate = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_orderdate.begin(), v_lo_orderdate.end(), h_lo_orderdate);
    int* d_lo_orderdate = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_orderdate, h_lo_orderdate, LO_LEN * sizeof(int));
    auto v_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
    int* h_lo_suppkey = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_suppkey.begin(), v_lo_suppkey.end(), h_lo_suppkey);
    int* d_lo_suppkey = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_suppkey, h_lo_suppkey, LO_LEN * sizeof(int));
    auto v_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
    int* h_lo_custkey = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_custkey.begin(), v_lo_custkey.end(), h_lo_custkey);
    int* d_lo_custkey = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_custkey, h_lo_custkey, LO_LEN * sizeof(int));
    auto v_lo_partkey = loadColumn<int>("lo_partkey", LO_LEN);
    int* h_lo_partkey = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_partkey.begin(), v_lo_partkey.end(), h_lo_partkey);
    int* d_lo_partkey = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_partkey, h_lo_partkey, LO_LEN * sizeof(int));
    auto v_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
    int* h_lo_revenue = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_revenue.begin(), v_lo_revenue.end(), h_lo_revenue);
    int* d_lo_revenue = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_revenue, h_lo_revenue, LO_LEN * sizeof(int));
    auto v_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);
    int* h_lo_supplycost = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_supplycost.begin(), v_lo_supplycost.end(), h_lo_supplycost);
    int* d_lo_supplycost = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_supplycost, h_lo_supplycost, LO_LEN * sizeof(int));

    auto v_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    int* h_d_datekey = malloc_host<int>(D_LEN, q);
    std::copy(v_d_datekey.begin(), v_d_datekey.end(), h_d_datekey);
    int* d_d_datekey = malloc_device<int>(D_LEN, q);
    q.memcpy(d_d_datekey, h_d_datekey, D_LEN * sizeof(int));
    auto v_d_year = loadColumn<int>("d_year", D_LEN);
    int* h_d_year = malloc_host<int>(D_LEN, q);
    std::copy(v_d_year.begin(), v_d_year.end(), h_d_year);
    int* d_d_year = malloc_device<int>(D_LEN, q);
    q.memcpy(d_d_year, h_d_year, D_LEN * sizeof(int));
    auto v_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);
    int* h_d_yearmonthnum = malloc_host<int>(D_LEN, q);
    std::copy(v_d_yearmonthnum.begin(), v_d_yearmonthnum.end(), h_d_yearmonthnum);
    int* d_d_yearmonthnum = malloc_device<int>(D_LEN, q);
    q.memcpy(d_d_yearmonthnum, h_d_yearmonthnum, D_LEN * sizeof(int));

    auto v_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
    int* h_s_suppkey = malloc_host<int>(S_LEN, q);
    std::copy(v_s_suppkey.begin(), v_s_suppkey.end(), h_s_suppkey);
    int* d_s_suppkey = malloc_device<int>(S_LEN, q);
    q.memcpy(d_s_suppkey, h_s_suppkey, S_LEN * sizeof(int));
    auto v_s_region = loadColumn<int>("s_region", S_LEN);
    int* h_s_region = malloc_host<int>(S_LEN, q);
    std::copy(v_s_region.begin(), v_s_region.end(), h_s_region);
    int* d_s_region = malloc_device<int>(S_LEN, q);
    q.memcpy(d_s_region, h_s_region, S_LEN * sizeof(int));
    auto v_s_nation = loadColumn<int>("s_nation", S_LEN);
    int* h_s_nation = malloc_host<int>(S_LEN, q);
    std::copy(v_s_nation.begin(), v_s_nation.end(), h_s_nation);
    int* d_s_nation = malloc_device<int>(S_LEN, q);
    q.memcpy(d_s_nation, h_s_nation, S_LEN * sizeof(int));

    auto v_p_partkey = loadColumn<int>("p_partkey", P_LEN);
    int* h_p_partkey = malloc_host<int>(P_LEN, q);
    std::copy(v_p_partkey.begin(), v_p_partkey.end(), h_p_partkey);
    int* d_p_partkey = malloc_device<int>(P_LEN, q);
    q.memcpy(d_p_partkey, h_p_partkey, P_LEN * sizeof(int));
    auto v_p_mfgr = loadColumn<int>("p_mfgr", P_LEN);
    int* h_p_mfgr = malloc_host<int>(P_LEN, q);
    std::copy(v_p_mfgr.begin(), v_p_mfgr.end(), h_p_mfgr);
    int* d_p_mfgr = malloc_device<int>(P_LEN, q);
    q.memcpy(d_p_mfgr, h_p_mfgr, P_LEN * sizeof(int));
    auto v_p_category = loadColumn<int>("p_category", P_LEN);
    int* h_p_category = malloc_host<int>(P_LEN, q);
    std::copy(v_p_category.begin(), v_p_category.end(), h_p_category);
    int* d_p_category = malloc_device<int>(P_LEN, q);
    q.memcpy(d_p_category, h_p_category, P_LEN * sizeof(int));

    auto v_c_custkey = loadColumn<int>("c_custkey", C_LEN);
    int* h_c_custkey = malloc_host<int>(C_LEN, q);
    std::copy(v_c_custkey.begin(), v_c_custkey.end(), h_c_custkey);
    int* d_c_custkey = malloc_device<int>(C_LEN, q);
    q.memcpy(d_c_custkey, h_c_custkey, C_LEN * sizeof(int));
    auto v_c_region = loadColumn<int>("c_region", C_LEN);
    int* h_c_region = malloc_host<int>(C_LEN, q);
    std::copy(v_c_region.begin(), v_c_region.end(), h_c_region);
    int* d_c_region = malloc_device<int>(C_LEN, q);
    q.memcpy(d_c_region, h_c_region, C_LEN * sizeof(int));

    std::cout << "** LOADING DATA **" << "\n";

                        
            
            
            
        
    int res_size = ((1998-1992+1) * 25 * 25);
    int res_array_size = res_size * 6;
    unsigned long long* h_result = malloc_host<unsigned long long>(res_array_size, q);
    unsigned long long* d_result = malloc_device<unsigned long long>(res_array_size, q);

    int d_val_len = 19981230 - 19920101 + 1;
    int* h_s_hash_table = malloc_host<int>(2*S_LEN, q);
    int* d_s_hash_table = malloc_device<int>(2*S_LEN, q);
    int s_hash_table_sz = 2*S_LEN;
    int* h_c_hash_table = malloc_host<int>(2*C_LEN, q);
    int* d_c_hash_table = malloc_device<int>(2*C_LEN, q);
    int c_hash_table_sz = 2*C_LEN;
    int* h_p_hash_table = malloc_host<int>(2*P_LEN, q);
    int* d_p_hash_table = malloc_device<int>(2*P_LEN, q);
    int p_hash_table_sz = 2*P_LEN;
    int* h_d_hash_table = malloc_host<int>(2*d_val_len, q);
    int* d_d_hash_table = malloc_device<int>(2*d_val_len, q);
    int d_hash_table_sz = 2*d_val_len;

    q.wait();
    std::cout << "** LOADING DATA **" << "\n";

    for (int t = 0; t < num_trials; t++) {
        {
            std::fill_n(h_result, res_array_size, 0);
            q.memcpy(d_result, h_result, res_array_size * sizeof(unsigned long long)).wait();

            q.memset(d_s_hash_table, 0, s_hash_table_sz * sizeof(int));

            q.memset(d_c_hash_table, 0, c_hash_table_sz * sizeof(int));

            q.memset(d_p_hash_table, 0, p_hash_table_sz * sizeof(int));

            q.memset(d_d_hash_table, 0, d_hash_table_sz * sizeof(int));
        }
        using namespace std::chrono;
        high_resolution_clock::time_point start, finish;
        start = high_resolution_clock::now();

        // 1 build_hashtable_s
        q.submit([&](sycl::handler& h) {
                                                
            int num_tiles = (S_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_s>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;
                
                int num_tiles = (S_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = S_LEN - tile_offset;
                }
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_s_region + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,1,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_s_suppkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_s_nation + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,items2,flags,d_s_hash_table,S_LEN,num_tile_items);
            });
        });

        // 2 build_hashtable_c
        q.submit([&](sycl::handler& h){
                                    
            int num_tiles = (C_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_c>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;
                
                int num_tiles = (C_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = C_LEN - tile_offset;
                }
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_c_region + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,1,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_c_custkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockBuildSelectivePHT_1<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,d_c_hash_table,C_LEN,num_tile_items);
            });
        });
        
        // 3 build_hashtable_p
        q.submit([&](sycl::handler& h){
                                                
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
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_p_mfgr + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,0,num_tile_items);
                BlockPredOrEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,1,num_tile_items);
                
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_p_partkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_p_category + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,items2,flags,d_p_hash_table,P_LEN,num_tile_items);
            });
        });

        // 4 build_hashtable_d
        q.submit([&](sycl::handler& h){
                                    
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
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_d_year + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,1997,num_tile_items);
                BlockPredOrEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,1998,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_d_datekey + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items2,items,flags,d_d_hash_table,d_val_len,19920101,num_tile_items);
            });
        });
        
        //5 select
        q.submit([&](sycl::handler& h){
                                                                        
                                                
            
            int num_tiles = (LO_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<select_kernel>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                int category[ITEMS_PER_THREAD];
                int s_nation[ITEMS_PER_THREAD];
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

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_suppkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,s_nation,flags,d_s_hash_table,S_LEN,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_custkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_1<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,d_c_hash_table,C_LEN,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_partkey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,category,flags,d_p_hash_table,P_LEN,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,year,flags,d_d_hash_table,d_val_len,19920101,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_revenue + tile_offset,tid,tile_offset,revenue,num_tile_items);
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_supplycost + tile_offset,tid,tile_offset,items,num_tile_items);

                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    if(flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)){
                        int hash = ((year[i] - 1992) * 25 * 25 + s_nation[i] * 25 + category[i]) % ((1998-1992+1) * 25 * 25);
                        d_result[hash*6]= year[i];
                        d_result[hash*6+1]= s_nation[i];
                        d_result[hash*6+2]= category[i];
                        sycl::atomic_ref<
                            unsigned long long,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_revenue(d_result[hash*6+4]);
                        atomic_revenue.fetch_add(revenue[i]-items[i]);
                    }
                }
            });
        });
        q.wait();
        finish = high_resolution_clock::now();
        std::chrono::duration<double> diff = finish - start;
        if(t>0) mean+=diff;
        {
            q.memcpy(h_result, d_result, res_array_size * sizeof(unsigned long long)).wait();
            unsigned long long* host_res = h_result;
            int res_count = 0;
            for (int i=0; i<res_size; i++) {
                if (host_res[6*i] != 0) {
                std::cout << host_res[6*i] << " " << host_res[6*i + 1] << " " << host_res[6*i + 2] << " " << reinterpret_cast<unsigned long long*>(&host_res[6*i + 4])[0]  << "\n";
                res_count += 1;
                }
            }
            std::cout << "Result count: " << res_count << "\n";
        }
        std::cout << "Total time: " << diff.count() * 1000 << " ms\n";
        
    }
    std::cout << "Mean time: " << mean.count()/3 * 1000 << " ms\n";
    // Free memory
    free(h_lo_orderdate, q);
    free(d_lo_orderdate, q);
    free(h_lo_suppkey, q);
    free(d_lo_suppkey, q);
    free(h_lo_custkey, q);
    free(d_lo_custkey, q);
    free(h_lo_partkey, q);
    free(d_lo_partkey, q);
    free(h_lo_revenue, q);
    free(d_lo_revenue, q);
    free(h_lo_supplycost, q);
    free(d_lo_supplycost, q);
    free(h_d_datekey, q);
    free(d_d_datekey, q);
    free(h_d_year, q);
    free(d_d_year, q);
    free(h_d_yearmonthnum, q);
    free(d_d_yearmonthnum, q);
    free(h_s_suppkey, q);
    free(d_s_suppkey, q);
    free(h_s_region, q);
    free(d_s_region, q);
    free(h_s_nation, q);
    free(d_s_nation, q);
    free(h_p_partkey, q);
    free(d_p_partkey, q);
    free(h_p_mfgr, q);
    free(d_p_mfgr, q);
    free(h_p_category, q);
    free(d_p_category, q);
    free(h_c_custkey, q);
    free(d_c_custkey, q);
    free(h_c_region, q);
    free(d_c_region, q);
    free(h_s_hash_table, q);
    free(d_s_hash_table, q);
    free(h_c_hash_table, q);
    free(d_c_hash_table, q);
    free(h_p_hash_table, q);
    free(d_p_hash_table, q);
    free(h_d_hash_table, q);
    free(d_d_hash_table, q);
    free(h_result, q);
    free(d_result, q);
    return 0;
}
