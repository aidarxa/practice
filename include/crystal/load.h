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

// ----------------------------------------NULL VALIDITY---------------------------------------------
inline int ColumnValidAt(const uint64_t* null_bitmap, int row_id) {
    if (null_bitmap == nullptr) return 1;
    const uint64_t word = null_bitmap[row_id >> 6];
    return static_cast<int>((word >> (row_id & 63)) & 1ULL);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockLoadValidity(uint64_t* null_bitmap,
                              int tid,
                              int tile_offset,
                              int (&valid)[ITEMS_PER_THREAD],
                              int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int local_idx = i * BLOCK_THREADS + tid;
        if (local_idx < num_items) {
            valid[i] = ColumnValidAt(null_bitmap, tile_offset + local_idx);
        } else {
            valid[i] = 0;
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockGatherValidity(uint64_t* null_bitmap,
                                int tid,
                                int (&row_ids)[ITEMS_PER_THREAD],
                                int (&flags)[ITEMS_PER_THREAD],
                                int (&valid)[ITEMS_PER_THREAD],
                                int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
            valid[i] = ColumnValidAt(null_bitmap, row_ids[i]);
        } else {
            valid[i] = 0;
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockApplyValidityAnd(int tid,
                                  int (&flags)[ITEMS_PER_THREAD],
                                  int (&valid)[ITEMS_PER_THREAD],
                                  int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (i * BLOCK_THREADS + tid < num_items) {
            flags[i] = flags[i] && valid[i];
        }
    }
}
// ----------------------------------------NULL VALIDITY---------------------------------------------
