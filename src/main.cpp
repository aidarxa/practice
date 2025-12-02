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
template <typename Acc>
inline unsigned long long select(nd_item<1> it, Acc lo_orderdate, Acc lo_discount, Acc lo_quantity, Acc lo_extendedprice, int num_entries){
    int items[ITEMS_PER_THREAD];
    int flags[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];

    unsigned long long sum = 0;
    
    int tile_offset =  it.get_group_linear_id()* TILE_SIZE;
    int num_tiles = (num_entries + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (it.get_group_linear_id() == num_tiles - 1) {
        num_tile_items = lo_num_entries - tile_offset;
    }
}

int main() {
    int num_trials = 3;

    int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    int *h_lo_discount = loadColumn<int>("lo_discount", LO_LEN);
    int *h_lo_quantity = loadColumn<int>("lo_quantity", LO_LEN);
    int *h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);
    int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    int *h_d_year = loadColumn<int>("d_year", D_LEN);

    std::cout << "** LOADED DATA **" << "\n";

    buffer<int> b_lo_orderdate (h_lo_orderdate,range<1>(LO_LEN));
    buffer<int> b_lo_discount (h_lo_discount,range<1>(LO_LEN));
    buffer<int> b_lo_quantity (h_lo_quantity,range<1>(LO_LEN));
    buffer<int> b_lo_extendedprice (h_lo_extendedprice,range<1>(LO_LEN));
    buffer<int> b_d_datekey (h_d_datekey,range<1>(D_LEN));
    buffer<int> b_d_year (h_d_year,range<1>(D_LEN));
    
    buffer<unsigned long long, 1> b_result{range<1>(1)};

    auto q = queue(default_selector_v);

    for (int t = 0; t < num_trials; t++) {
        using namespace std::chrono;
        high_resolution_clock::time_point start, finish;
        start = high_resolution_clock::now();
        q.submit([&](sycl::handler& h) {
            auto a_lo_orderdate = b_lo_orderdate.get_access<access_mode::read>(h);
            auto a_lo_discount = b_lo_discount.get_access<access_mode::read>(h);
            auto a_lo_quantity = b_lo_quantity.get_access<access_mode::read>(h);
            auto a_lo_extendedprice = b_lo_extendedprice.get_access<access_mode::read>(h);
            auto a_result = b_result.get_access<access_mode::discard_write>(h);

            int num_tiles = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
            size_t local  = BLOCK_THREADS;
            size_t global = num_tiles * BLOCK_THREADS;
            h.parallel_for<select_kernel>(nd_range<1>(global,local),[=](nd_item<1> it){
                unsigned long long local_sum = select(it, a_lo_orderdate, a_lo_discount, a_lo_quantity, a_lo_extendedprice, LO_LEN);
            });
        });
        q.wait();
        finish = high_resolution_clock::now();
        std::chrono::duration<double> diff = finish - start;
        {
            auto host_res = b_result.get_host_access();
            std::cout << "Result: " << host_res[0] << "\n";
        }
        std::cout << "Total time: " << diff.count() * 1000 << " ms\n";
    }
    return 0;
}
