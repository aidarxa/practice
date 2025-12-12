#pragma once
#define HASH(X,Y,Z) ((X-Z)%Y)

// Check presence in perfect hash table (1 - in hash table save only keys, 2 - in hash table save key-value pairs)
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type>
inline void BlockProbeDirectAndPHT_1(int tid,K (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i]){
            int hash = HASH(items[i], ht_len, key_mins);
            K slot = acc_ht[hash];
            if(slot != 0){
                flags[i] = 1;
            } else {
                flags[i] = 0;
            }
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type>
inline void BlockProbeDirectAndPHT_1(int tid,K (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i] && (i*BLOCK_THREADS + tid < num_items)){
            int hash = HASH(items[i], ht_len, key_mins);
            K slot = acc_ht[hash];
            if(slot != 0){
                flags[i] = 1;
            } else {
                flags[i] = 0;
            }
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type>
inline void BlockProbeAndPHT_1(int tid,K (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items) 
{
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockProbeDirectAndPHT_1<decltype(acc_ht),BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,acc_ht,ht_len,key_mins);
    } else {
        BlockProbeDirectAndPHT_1<decltype(acc_ht),BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,flags,acc_ht,ht_len,key_mins,num_items);
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type>
inline void BlockProbeAndPHT_1(int tid,K (&items)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len,int num_items) 
{
    BlockProbeAndPHT_1<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,flags,acc_ht,ht_len,0,num_items);
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockProbeDirectAndPHT_2(int tid,K (&items)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i]){
            int hash = HASH(items[i], ht_len, key_mins);
            int key  = acc_ht[hash << 1];
            if(key != 0){
                res[i] = static_cast<V>(acc_ht[(hash << 1) + 1]);
            } else {
                flags[i] = 0;
            }
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockProbeDirectAndPHT_2(int tid,K (&items)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i] && (i*BLOCK_THREADS + tid < num_items)){
            int hash = HASH(items[i], ht_len, key_mins);
            int key  = acc_ht[hash << 1];
            if(key != 0){
                res[i] = static_cast<V>(acc_ht[(hash << 1) + 1]);
            } else {
                flags[i] = 0;
            }
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockProbeAndPHT_2(int tid,K (&items)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items) 
{
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockProbeDirectAndPHT_2<decltype(acc_ht),BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,res,flags,acc_ht,ht_len,key_mins);
    } else {
        BlockProbeDirectAndPHT_2<decltype(acc_ht),BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,items,res,flags,acc_ht,ht_len,key_mins,num_items);
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockProbeAndPHT_2(int tid,K (&items)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len,int num_items) 
{
    BlockProbeAndPHT_2<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,items,res,flags,acc_ht,ht_len,0,num_items);
}

// Construct hash table on device

template <typename Acc, int BlOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type>
inline void BlockBuildDirectSelectivePHT_1(int tid,K (&keys)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i]){
            int hash = HASH(keys[i], ht_len, key_mins);

            //atomicCAS(&acc_ht[hash], 0, keys[i]);

            sycl::atomic_ref<
                K,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> 
                at(acc_ht[hash]);
            K expected = 0;
            at.compare_exchange_strong(expected, keys[i]);
        }
    }
}
template <typename Acc, int BlOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type>
inline void BlockBuildDirectSelectivePHT_1(int tid,K (&keys)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins, int num_items){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i] && (i*BlOCK_THREADS + tid < num_items)){
            int hash = HASH(keys[i], ht_len, key_mins);

            //atomicCAS(&acc_ht[hash], 0, keys[i]);

            sycl::atomic_ref<
                K,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> 
                at(acc_ht[hash]);
            K expected = 0;
            at.compare_exchange_strong(expected, keys[i]);
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type>
inline void BlockBuildSelectivePHT_1(int tid,K (&keys)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD],const Acc& acc_ht,int ht_len, K key_mins,int num_items) 
{
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockBuildDirectSelectivePHT_1<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,keys,flags,acc_ht,ht_len,key_mins);
    } else {
        BlockBuildDirectSelectivePHT_1<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,keys,flags,acc_ht,ht_len,key_mins,num_items);
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type>
inline void BlockBuildSelectivePHT_1(int tid,K (&keys)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD],const Acc& acc_ht,int ht_len,int num_items) 
{
    BlockBuildSelectivePHT_1<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,keys,flags,acc_ht,ht_len,0,num_items);
}
template <typename Acc, int BlOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockBuildDirectSelectivePHT_2(int tid,K (&keys)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i]){
            int hash = HASH(keys[i], ht_len, key_mins);

            //atomicCAS(&acc_ht[hash << 1], 0, keys[i]);
            sycl::atomic_ref<
                K,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> 
                at(acc_ht[hash << 1]);
            K expected = 0;
            at.compare_exchange_strong(expected, keys[i]);
            acc_ht[(hash << 1) + 1] = res[i];
        }
    }
}
template <typename Acc, int BlOCK_THREADS, int ITEMS_PER_THREAD, typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockBuildDirectSelectivePHT_2(int tid,K (&keys)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items){
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if(flags[i] && (i*BlOCK_THREADS + tid < num_items)){
            int hash = HASH(keys[i], ht_len, key_mins);
            
            //atomicCAS(&acc_ht[hash << 1], 0, keys[i]);
            sycl::atomic_ref<
                K,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> 
                at(acc_ht[hash << 1]);
            K expected = 0;
            at.compare_exchange_strong(expected, keys[i]);
            acc_ht[(hash << 1) + 1] = res[i];
        }
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockBuildSelectivePHT_2(int tid,K (&keys)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD], const Acc& acc_ht,int ht_len, K key_mins,int num_items) 
{
    if(BLOCK_THREADS * ITEMS_PER_THREAD == num_items){
        BlockBuildDirectSelectivePHT_2<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,keys,res,flags,acc_ht,ht_len,key_mins);
    } else {
        BlockBuildDirectSelectivePHT_2<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(
            tid,keys,res,flags,acc_ht,ht_len,key_mins,num_items);
    }
}
template <typename Acc, int BLOCK_THREADS, int ITEMS_PER_THREAD,typename K = typename Acc::value_type, typename V = typename Acc::value_type>
inline void BlockBuildSelectivePHT_2(int tid,K (&keys)[ITEMS_PER_THREAD],V (&res)[ITEMS_PER_THREAD],int (&flags)[ITEMS_PER_THREAD],const Acc& acc_ht,int ht_len,int num_items) 
{
    BlockBuildSelectivePHT_2<Acc,BLOCK_THREADS,ITEMS_PER_THREAD>(tid,keys,res,flags,acc_ht,ht_len,0,num_items);
}