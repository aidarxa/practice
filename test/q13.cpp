#include <sycl/sycl.hpp>
#include "../include/load.h"
#include "../include/pred.h"
#include "../include/utils.h"
#include <chrono>
using namespace sycl;

constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

class select_kernel;


int main() {
    int num_trials = 4;
    std::chrono::duration<double> mean;

    std::vector<int> h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    std::vector<int> h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
    std::vector<int> h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
    std::vector<int> h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
    std::vector<int> h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    std::vector<int> h_d_year = loadColumn<int>("d_year", D_LEN);

    std::cout << "** LOADED DATA **" << "\n";

    buffer<int> b_lo_orderdate (h_lo_orderdate.data(),range<1>(LO_LEN));
    buffer<int> b_lo_discount (h_lo_discount.data(),range<1>(LO_LEN));
    buffer<int> b_lo_quantity (h_lo_quantity.data(),range<1>(LO_LEN));
    buffer<int> b_lo_extendedprice (h_lo_extendedprice.data(),range<1>(LO_LEN));
    buffer<int> b_d_datekey (h_d_datekey.data(),range<1>(D_LEN));
    buffer<int> b_d_year (h_d_year.data(),range<1>(D_LEN));
    
    buffer<unsigned long long, 1> b_result{range<1>(1)};
    {
        auto host_res = b_result.get_host_access();
        host_res[0] = 0;
    }

    auto q = queue(gpu_selector_v);

    for (int t = 0; t < num_trials; t++) {
        using namespace std::chrono;
        high_resolution_clock::time_point start, finish;
        start = high_resolution_clock::now();
        q.submit([&](sycl::handler& h) {
            auto a_lo_orderdate = b_lo_orderdate.get_access<access_mode::read>(h);
            auto a_lo_discount = b_lo_discount.get_access<access_mode::read>(h);
            auto a_lo_quantity = b_lo_quantity.get_access<access_mode::read>(h);
            auto a_lo_extendedprice = b_lo_extendedprice.get_access<access_mode::read>(h);
            auto a_result = b_result.get_access<access_mode::write>(h);

            int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
            size_t local  = BLOCK_THREADS;
            size_t global = num_tiles * BLOCK_THREADS;
            h.parallel_for<select_kernel>(nd_range<1>(global,local),[=](nd_item<1> it){

                int items[ITEMS_PER_THREAD];
                int flags[ITEMS_PER_THREAD];
                int items2[ITEMS_PER_THREAD];
                
                int tid = it.get_local_linear_id();
                int num_entries = LO_LEN;
                unsigned long long sum = 0;
                
                int tile_offset =  it.get_group_linear_id()* TILE_SIZE;
                int num_tiles = (num_entries + TILE_SIZE - 1) / TILE_SIZE;
                int num_tile_items = TILE_SIZE;

                if (it.get_group_linear_id() == num_tiles - 1) {
                    num_tile_items = num_entries - tile_offset;
                }
                BlockLoad<decltype(a_lo_orderdate),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_orderdate,tid,tile_offset,items,num_tile_items);
                BlockPredGTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,19940204,num_tile_items);
                BlockPredALTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,19940210,num_tile_items);

                BlockLoad<decltype(a_lo_quantity),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_quantity,tid,tile_offset,items,num_tile_items);
                BlockPredAGTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,26,num_tile_items);
                BlockPredALTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,35,num_tile_items);

                BlockLoad<decltype(a_lo_discount),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_discount,tid,tile_offset,items,num_tile_items);
                BlockPredAGTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,5,num_tile_items);
                BlockPredALTE<int,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,7,num_tile_items);

                BlockLoad<decltype(a_lo_extendedprice),BLOCK_THREADS,ITEMS_PER_THREAD>(a_lo_extendedprice,tid,tile_offset,items2,num_tile_items);

                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    if(tid + BLOCK_THREADS * i < num_tile_items){
                        if(flags[i]){
                            sum += items2[i] * items[i] ;
                        }
                    }
                }

                unsigned long long aggregate = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<unsigned long long>());

                if(tid == 0){
                    sycl::atomic_ref<
                            unsigned long long,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                            atomic_revenue(a_result[0]);

                    atomic_revenue.fetch_add(static_cast<unsigned long long>(aggregate));
                }
            });
        });
        q.wait();
        finish = high_resolution_clock::now();
        std::chrono::duration<double> diff = finish - start;
        if(t>0) mean+=diff;
        {
            auto host_res = b_result.get_host_access();
            std::cout << "Result: " << host_res[0] << "\n";
            host_res[0] = 0;
        }
        std::cout << "Total time: " << diff.count() * 1000 << " ms\n";
    }
    std::cout << "Mean time: " << mean.count()/3 * 1000 << " ms\n";
    return 0;
}
