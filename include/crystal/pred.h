#pragma once
#include <cstddef>
//ACPP_KERNEL_TARGET
// -------------------------------------------PREDICATE----------------------------------------------
template <typename T>
struct GreaterThan{
    T _compare;
    GreaterThan(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value >_compare;
    };
};
template <typename T>
struct LessThan{
    T _compare;
    LessThan(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value <_compare;
    };
};
template <typename T>
struct GreaterThanEq{
    T _compare;
    GreaterThanEq(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value >=_compare;
    };
};
template <typename T>
struct LessThanEq{
    T _compare;
    LessThanEq(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value <= _compare;
    };
};
template <typename T>
struct Equal{
    T _compare;
    Equal(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value == _compare;
    };
};
template <typename T>
struct NotEqual{
    T _compare;
    NotEqual(T compare):_compare(compare){};
    bool operator()(const T& value) const noexcept {
        return value == _compare;
    };
};


template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void InitFlags(int (&flags)[ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        flags[i] = 1;
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void InitFlagsZero(int (&flags)[ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        flags[i] = 0;
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        flags[i] = pred(items[i]);
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(i*BLOCK_THREADS + tid < num_items){
            flags[i] = pred(items[i]);
        }
    }
}
template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPred(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockPredDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred);
    } else {
        BlockPredDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred,num_items);
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        flags[i] = flags[i] && pred(items[i]);
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAndDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(i*BLOCK_THREADS + tid < num_items){
            flags[i] = flags[i] && pred(items[i]);
        }
    }
}
template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAnd(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockPredAndDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred);
    } else {
        BlockPredAndDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred,num_items);
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOrDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        flags[i] = flags[i] || pred(items[i]);
    }
}

template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOrDirect(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(i*BLOCK_THREADS + tid < num_items){
            flags[i] = flags[i] || pred(items[i]);
        }
    }
}
template <typename T,typename Predicate, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOr(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], Predicate pred, int num_items) {
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockPredOrDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred);
    } else {
        BlockPredOrDirect<T,Predicate,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,pred,num_items);
    }
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredLT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThan<T> pred(compare);
    BlockPred<T,LessThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredALT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThan<T> pred(compare);
    BlockPredAnd<T,LessThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOLT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThan<T> pred(compare);
    BlockPredOr<T,LessThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredGT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThan<T> pred(compare);
    BlockPred<T,GreaterThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAGT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThan<T> pred(compare);
    BlockPredAnd<T,GreaterThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOGT(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThan<T> pred(compare);
    BlockPredOr<T,GreaterThan<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredLTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThanEq<T> pred(compare);
    BlockPred<T,LessThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredALTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThanEq<T> pred(compare);
    BlockPredAnd<T,LessThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOLTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    LessThanEq<T> pred(compare);
    BlockPredOr<T,LessThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredGTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThanEq<T> pred(compare);
    BlockPred<T,GreaterThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAGTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThanEq<T> pred(compare);
    BlockPredAnd<T,GreaterThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOGTE(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    GreaterThanEq<T> pred(compare);
    BlockPredOr<T,GreaterThanEq<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    Equal<T> pred(compare);
    BlockPred<T,Equal<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredAEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    Equal<T> pred(compare);
    BlockPredAnd<T,Equal<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOrEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    Equal<T> pred(compare);
    BlockPredOr<T,Equal<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredNEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    NotEqual<T> pred(compare);
    BlockPred<T,NotEqual<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredANEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    NotEqual<T> pred(compare);
    BlockPredAnd<T,NotEqual<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockPredOrNEq(size_t tid, T (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], T compare, int num_items) {
    NotEqual<T> pred(compare);
    BlockPredOr<T,NotEqual<T>,BLOCK_THREADS,ITEMS_PER_THREAD>(
        tid,items,flags,pred,num_items);
}
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockApplyMaskOr(size_t tid, int (&target_mask)[ITEMS_PER_THREAD], int (&source_mask)[ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        target_mask[i] = target_mask[i] || source_mask[i];
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockApplyMaskAnd(size_t tid, int (&target_mask)[ITEMS_PER_THREAD], int (&source_mask)[ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        target_mask[i] = target_mask[i] && source_mask[i];
    }
}
// -------------------------------------------PREDICATE----------------------------------------------