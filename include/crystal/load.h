#pragma once
#include <cstddef>
//ACPP_KERNEL_TARGET

// -------------------------------------------BLOCK LOAD---------------------------------------------
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoadDirect(T* block_iter,size_t tid, T (&items)[ITEMS_PER_THREAD]) {
    T* thread_iter = block_iter + tid;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] = thread_iter[i * BLOCK_THREADS];
    }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoadDirect(T* block_iter,size_t tid, T (&items)[ITEMS_PER_THREAD],int num_items) {
    T* thread_iter = block_iter + tid;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(i*BLOCK_THREADS + tid < num_items){
            items[i] = thread_iter[i * BLOCK_THREADS];
        }
    }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoad(T* input, size_t tid, int tile_offset, int (&items)[ITEMS_PER_THREAD],int num_items) 
{
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockLoadDirect<T,BLOCK_THREADS,ITEMS_PER_THREAD>(
            input,tid,items);
    } else {
        BlockLoadDirect<T,BLOCK_THREADS,ITEMS_PER_THREAD>(
            input,tid,items,num_items);
    }
}
// -------------------------------------------BLOCK LOAD---------------------------------------------
// -----------------------------------------------END------------------------------------------------


// ----------------------------------------BLOCK GATHER---------------------------------------------
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockGather(T* input,
                        int tid,
                        int (&row_ids)[ITEMS_PER_THREAD],
                        int (&flags)[ITEMS_PER_THREAD],
                        T (&items)[ITEMS_PER_THREAD],
                        int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
            items[i] = input[row_ids[i]];
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockMakeRowIds(int tid,
                            int tile_offset,
                            int (&row_ids)[ITEMS_PER_THREAD],
                            int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (i * BLOCK_THREADS + tid < num_items) {
            row_ids[i] = tile_offset + tid + i * BLOCK_THREADS;
        }
    }
}
// ----------------------------------------BLOCK GATHER---------------------------------------------
