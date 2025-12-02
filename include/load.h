#pragma once
#include <cstddef>
//ACPP_KERNEL_TARGET

// -------------------------------------------BLOCK LOAD---------------------------------------------
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T = typename Acc::value_type>
inline void BlockLoadDirect(const Acc& acc,size_t tid, int tile_offset, T (&items)[ITEMS_PER_THREAD]) {
    size_t tid_offset = tid + tile_offset;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] = acc[tid_offset + i * BLOCK_THREADS;];
    }
}

template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T = typename Acc::value_type>
inline void BlockLoadDirect(const Acc& acc,size_t tid, int tile_offset, T (&items)[ITEMS_PER_THREAD],int num_items) {
    size_t tid_offset = tid + tile_offset;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(i*BLOCK_THREADS + tid < num_items){
            items[i] = acc[tid_offset + i * BLOCK_THREADS;];
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoad(Acc &acc, size_t tid, int tile_offset, int (&items)[ITEMS_PER_THREAD],int num_items) {
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockLoadDirect<typename Acc::value_type,BLOCK_THREADS,ITEMS_PER_THREAD>(
            acc,tid,tile_offset,items);
    } else {
        BlockLoadDirect<typename Acc::value_type,BLOCK_THREADS,ITEMS_PER_THREAD>(
            acc,tid,tile_offset,items,num_items);
    }
}
// -------------------------------------------BLOCK LOAD---------------------------------------------
// -----------------------------------------------END------------------------------------------------

