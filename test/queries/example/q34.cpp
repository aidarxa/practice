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
    auto v_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
    int* h_lo_custkey = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_custkey.begin(), v_lo_custkey.end(), h_lo_custkey);
    int* d_lo_custkey = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_custkey, h_lo_custkey, LO_LEN * sizeof(int));
    auto v_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
    int* h_lo_suppkey = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_suppkey.begin(), v_lo_suppkey.end(), h_lo_suppkey);
    int* d_lo_suppkey = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_suppkey, h_lo_suppkey, LO_LEN * sizeof(int));
    auto v_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);
    int* h_lo_revenue = malloc_host<int>(LO_LEN, q);
    std::copy(v_lo_revenue.begin(), v_lo_revenue.end(), h_lo_revenue);
    int* d_lo_revenue = malloc_device<int>(LO_LEN, q);
    q.memcpy(d_lo_revenue, h_lo_revenue, LO_LEN * sizeof(int));

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
    auto v_s_city = loadColumn<int>("s_city", S_LEN);
    int* h_s_city = malloc_host<int>(S_LEN, q);
    std::copy(v_s_city.begin(), v_s_city.end(), h_s_city);
    int* d_s_city = malloc_device<int>(S_LEN, q);
    q.memcpy(d_s_city, h_s_city, S_LEN * sizeof(int));

    auto v_c_custkey = loadColumn<int>("c_custkey", C_LEN);
    int* h_c_custkey = malloc_host<int>(C_LEN, q);
    std::copy(v_c_custkey.begin(), v_c_custkey.end(), h_c_custkey);
    int* d_c_custkey = malloc_device<int>(C_LEN, q);
    q.memcpy(d_c_custkey, h_c_custkey, C_LEN * sizeof(int));
    auto v_c_city = loadColumn<int>("c_city", C_LEN);
    int* h_c_city = malloc_host<int>(C_LEN, q);
    std::copy(v_c_city.begin(), v_c_city.end(), h_c_city);
    int* d_c_city = malloc_device<int>(C_LEN, q);
    q.memcpy(d_c_city, h_c_city, C_LEN * sizeof(int));

    std::cout << "** LOADING DATA **" << "\n";

                
            
        
        
    int res_size = ((1998-1992+1) * 250 * 250);
    int res_array_size = res_size * 4;
    unsigned long long* h_result = malloc_host<unsigned long long>(res_array_size, q);
    unsigned long long* d_result = malloc_device<unsigned long long>(res_array_size, q);

    int d_val_len = 19981230 - 19920101 + 1;
    int* h_s_hash_table = malloc_host<int>(2*S_LEN, q);
    int* d_s_hash_table = malloc_device<int>(2*S_LEN, q);
    int s_hash_table_sz = 2*S_LEN;
    int* h_c_hash_table = malloc_host<int>(2*C_LEN, q);
    int* d_c_hash_table = malloc_device<int>(2*C_LEN, q);
    int c_hash_table_sz = 2*C_LEN;
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

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_s_city + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,231,num_tile_items);
                BlockPredOrEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,235,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_s_suppkey + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items2,items,flags,d_s_hash_table,S_LEN,num_tile_items);
            });
        });
        // 2 build_hashtable_c
        q.submit([&](sycl::handler& h){
                                    
            int num_tiles = (C_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<build_hashtable_c>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];

                int tid = it.get_local_linear_id();
                int tile_offset = it.get_group_linear_id()*TILE_SIZE;
                
                int num_tiles = (C_LEN + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;
                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = C_LEN - tile_offset;
                }

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_c_city + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,231,num_tile_items);
                BlockPredOrEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,235,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_c_custkey + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items2,items,flags,d_c_hash_table,C_LEN,num_tile_items);
            });
        });
        // 3 build_hashtable_d
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

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_d_yearmonthnum + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockPredEq<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,199712,num_tile_items);
                
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_d_datekey + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_d_year + tile_offset,tid,tile_offset,items2,num_tile_items);
                BlockBuildSelectivePHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,items2,flags,d_d_hash_table,d_val_len,19920101,num_tile_items);
            });
        });

        //4 select
        q.submit([&](sycl::handler& h){
                                                
                                    
            
            int num_tiles = (LO_LEN + TILE_SIZE - 1)/TILE_SIZE;
            size_t local = BLOCK_THREADS;
            size_t global = num_tiles*BLOCK_THREADS;
            h.parallel_for<select_kernel>(nd_range<1>(global,local),[=](nd_item<1> it){
                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                int c_nation[ITEMS_PER_THREAD];
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
                BlockProbeAndPHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,c_nation,flags,d_c_hash_table,C_LEN,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_orderdate + tile_offset,tid,tile_offset,items,num_tile_items);
                BlockProbeAndPHT_2<int,int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,year,flags,d_d_hash_table,d_val_len,19920101,num_tile_items);

                BlockLoad<int,BLOCK_THREADS,ITEMS_PER_THREAD>(d_lo_revenue + tile_offset,tid,tile_offset,revenue,num_tile_items);

                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    if(flags[i] && (tid + BLOCK_THREADS * i < num_tile_items)){
                        int hash = (s_nation[i] * 250 * 7  + c_nation[i] * 7 + (year[i] - 1992)) % ((1998-1992+1) * 250 * 250);
                        d_result[hash*4]= year[i];
                        d_result[hash*4+1]= c_nation[i];
                        d_result[hash*4+2]= s_nation[i];
                        sycl::atomic_ref<
                            unsigned long long,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_revenue(d_result[hash*4+3]);
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
            q.memcpy(h_result, d_result, res_array_size * sizeof(unsigned long long)).wait();
            unsigned long long* host_res = h_result;
            int res_count = 0;
            for (int i=0; i<res_size; i++) {
                if (host_res[4*i] != 0) {
                std::cout << host_res[4*i] << " " << host_res[4*i + 1] << " " << host_res[4*i + 2] << " " << host_res[4*i + 3] << "\n";
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
    free(h_lo_custkey, q);
    free(d_lo_custkey, q);
    free(h_lo_suppkey, q);
    free(d_lo_suppkey, q);
    free(h_lo_revenue, q);
    free(d_lo_revenue, q);
    free(h_d_datekey, q);
    free(d_d_datekey, q);
    free(h_d_year, q);
    free(d_d_year, q);
    free(h_d_yearmonthnum, q);
    free(d_d_yearmonthnum, q);
    free(h_s_suppkey, q);
    free(d_s_suppkey, q);
    free(h_s_city, q);
    free(d_s_city, q);
    free(h_c_custkey, q);
    free(d_c_custkey, q);
    free(h_c_city, q);
    free(d_c_city, q);
    free(h_s_hash_table, q);
    free(d_s_hash_table, q);
    free(h_c_hash_table, q);
    free(d_c_hash_table, q);
    free(h_d_hash_table, q);
    free(d_d_hash_table, q);
    free(h_result, q);
    free(d_result, q);
    return 0;
}
