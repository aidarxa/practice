#pragma once
#include <sycl/sycl.hpp>
#define HASH(X, Y, Z) ((X - Z) % Y)

// Check presence in perfect hash table (1 - in hash table save only keys, 2 -
// in hash table save key-value pairs)
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len,
                                     K key_mins) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i]) {
      int hash = HASH(items[i], ht_len, key_mins);
      K slot = ht[hash];
      if (slot == items[i]) {
        flags[i] = 1;
      } else {
        flags[i] = 0;
      }
    }
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len, K key_mins,
                                     int num_items) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
      int hash = HASH(items[i], ht_len, key_mins);
      K slot = ht[hash];
      if (slot == items[i]) {
        flags[i] = 1;
      } else {
        flags[i] = 0;
      }
    }
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                               int (&flags)[ITEMS_PER_THREAD],
                               K* ht, int ht_len, K key_mins,
                               int num_items) {
  if (BLOCK_THREADS * ITEMS_PER_THREAD == num_items) {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, items, flags, ht, ht_len, key_mins);
  } else {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, items, flags, ht, ht_len, key_mins, num_items);
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_1(int tid, K (&items)[ITEMS_PER_THREAD],
                               int (&flags)[ITEMS_PER_THREAD],
                               K* ht, int ht_len, int num_items) {
  BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      tid, items, flags, ht, ht_len, 0, num_items);
}
template <typename K,typename V,int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_2(int tid, K (&items)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len,
                                     K key_mins) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i]) {
      int hash = HASH(items[i], ht_len, key_mins);
      K slot_key = ht[hash << 1];
      if (slot_key == items[i]) {
        res[i] = static_cast<V>(ht[(hash << 1) + 1]);
      } else {
        flags[i] = 0;
      }
    }
  }
}
template <typename K,typename V,int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeDirectAndPHT_2(int tid, K (&items)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len, K key_mins,
                                     int num_items) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
      int hash = HASH(items[i], ht_len, key_mins);
      K slot_key = ht[hash << 1];
      if (slot_key == items[i]) {
        res[i] = static_cast<V>(ht[(hash << 1) + 1]);
      } else {
        flags[i] = 0;
      }
    }
  }
}
template <typename K,typename V,int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_2(int tid, K (&items)[ITEMS_PER_THREAD],
                   V (&res)[ITEMS_PER_THREAD], int (&flags)[ITEMS_PER_THREAD],
                   K* ht, int ht_len, K key_mins, int num_items) {
  if (BLOCK_THREADS * ITEMS_PER_THREAD == num_items) {
    BlockProbeDirectAndPHT_2<K,V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, items, res, flags, ht, ht_len, key_mins);
  } else {
    BlockProbeDirectAndPHT_2<K,V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, items, res, flags, ht, ht_len, key_mins, num_items);
  }
}
template <typename K,typename V,int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockProbeAndPHT_2(int tid, K (&items)[ITEMS_PER_THREAD],
                               V (&res)[ITEMS_PER_THREAD],
                               int (&flags)[ITEMS_PER_THREAD],
                               K* ht, int ht_len, int num_items) {
  BlockProbeAndPHT_2<K,V, BLOCK_THREADS, ITEMS_PER_THREAD>(
      tid, items, res, flags, ht, ht_len, 0, num_items);
}

// Construct hash table on device

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildDirectSelectivePHT_1(int tid, K (&keys)[ITEMS_PER_THREAD],
                                           int (&flags)[ITEMS_PER_THREAD],
                                           K* ht, int ht_len,
                                           K key_mins) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i]) {
      int hash = HASH(keys[i], ht_len, key_mins);

      // K old = atomicCAS(&ht[hash], 0, keys[i]);

      sycl::atomic_ref<K, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          at(ht[hash]);
      K expected = 0;
      at.compare_exchange_strong(expected, keys[i]);
    }
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildDirectSelectivePHT_1(int tid, K (&keys)[ITEMS_PER_THREAD],
                                           int (&flags)[ITEMS_PER_THREAD],
                                           K* ht, int ht_len,
                                           K key_mins, int num_items) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
      int hash = HASH(keys[i], ht_len, key_mins);

      // K old = atomicCAS(&ht[hash], 0, keys[i]);

      sycl::atomic_ref<K, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          at(ht[hash]);
      K expected = 0;
      at.compare_exchange_strong(expected, keys[i]);
    }
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_1(int tid, K (&keys)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len, K key_mins,
                                     int num_items) {
  if (BLOCK_THREADS * ITEMS_PER_THREAD == num_items) {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, keys, flags, ht, ht_len, key_mins);
  } else {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, keys, flags, ht, ht_len, key_mins, num_items);
  }
}
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_1(int tid, K (&keys)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len,
                                     int num_items) {
  BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      tid, keys, flags, ht, ht_len, 0, num_items);
}
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildDirectSelectivePHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                                           V (&res)[ITEMS_PER_THREAD],
                                           int (&flags)[ITEMS_PER_THREAD],
                                           K* ht, int ht_len,
                                           K key_mins) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i]) {
      int hash = HASH(keys[i], ht_len, key_mins);

      // K old = atomicCAS(&ht[hash << 1], 0, keys[i]);
      sycl::atomic_ref<K, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          at(ht[hash << 1]);
      K expected = 0;
      at.compare_exchange_strong(expected, keys[i]);
      ht[(hash << 1) + 1] = res[i];
    }
  }
}
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildDirectSelectivePHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                                           V (&res)[ITEMS_PER_THREAD],
                                           int (&flags)[ITEMS_PER_THREAD],
                                           K* ht, int ht_len,
                                           K key_mins, int num_items) {
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
      int hash = HASH(keys[i], ht_len, key_mins);

      // K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
      sycl::atomic_ref<K, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          at(ht[hash << 1]);
      K expected = 0;
      at.compare_exchange_strong(expected, keys[i]);
      ht[(hash << 1) + 1] = res[i];
    }
  }
}
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len, K key_mins,
                                     int num_items) {
  if (BLOCK_THREADS * ITEMS_PER_THREAD == num_items) {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, keys, res, flags, ht, ht_len, key_mins);
  } else {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        tid, keys, res, flags, ht, ht_len, key_mins, num_items);
  }
}
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildSelectivePHT_2(int tid, K (&keys)[ITEMS_PER_THREAD],
                                     V (&res)[ITEMS_PER_THREAD],
                                     int (&flags)[ITEMS_PER_THREAD],
                                     K* ht, int ht_len,
                                     int num_items) {
  BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
      tid, keys, res, flags, ht, ht_len, 0, num_items);
}
// ============================================================================
// Push-Model Thread-Local Probes (Inside the thread loop)
// ============================================================================

template <typename K>
inline bool ProbePHT_1(K key, K* ht, int ht_len, K key_mins) {
    int hash = HASH(key, ht_len, key_mins);
    K slot = ht[hash];
    return slot == key;
}

template <typename K, typename V>
inline bool ProbePHT_2(K key, V& res, K* ht, int ht_len, K key_mins) {
    int hash = HASH(key, ht_len, key_mins);
    K slot_key = ht[hash << 1];
    if (slot_key == key) {
        res = static_cast<V>(ht[(hash << 1) + 1]);
        return true;
    }
    return false;
}

// Multi-value Hash Table (MHT) Probe
// Directory format: [Key, Offset, Count] (3 elements per entry)
template <typename K>
inline void ProbeMultiHT(K key, int& offset, int& count, int& flag, K* ht_dir, int ht_len, K key_mins) {
    if (flag) {
        int hash = HASH(key, ht_len, key_mins);
        // We assume Directory is an array of K where each entry is 3 elements:
        // ht_dir[hash * 3 + 0] = Key
        // ht_dir[hash * 3 + 1] = Offset
        // ht_dir[hash * 3 + 2] = Count
        K slot_key = ht_dir[hash * 3];
        if (slot_key == key && ht_dir[hash * 3 + 2] > 0) {
            offset = ht_dir[hash * 3 + 1];
            count = ht_dir[hash * 3 + 2];
        } else {
            flag = 0;
            count = 0;
        }
    } else {
        count = 0;
    }
}

// ============================================================================
// Multi-value Hash Table Build Primitives (Block-level)
// ============================================================================

// Pass 1: Count
// For each active item, atomically increment the count at its hash slot.
// d_counts must be zero-initialised before this kernel runs.
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildMHT_Count(int tid,
                                K (&keys)[ITEMS_PER_THREAD],
                                int (&flags)[ITEMS_PER_THREAD],
                                int* d_counts, int ht_len, K key_mins,
                                int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
            int hash = HASH(keys[i], ht_len, key_mins);
            sycl::atomic_ref<int,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                at(d_counts[hash]);
            at.fetch_add(1);
        }
    }
}

// Pass 2: Write
// d_offsets   – prefix-sum result (base write position per hash slot).
// d_write_pos – per-slot atomic running counter; must be zero-initialised
//               before this kernel so each thread can grab a unique slot.
// payload     – contiguous array of size = total payload count.
// ht_dir      – directory: [Key, Offset, Count] per slot (3 ints each).
template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
inline void BlockBuildMHT_Write(int tid,
                                K (&keys)[ITEMS_PER_THREAD],
                                V (&vals)[ITEMS_PER_THREAD],
                                int (&flags)[ITEMS_PER_THREAD],
                                int* ht_dir,
                                int* d_offsets,
                                int* d_write_pos,
                                int* d_counts,
                                int* payload,
                                int ht_len, K key_mins,
                                int num_items) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (flags[i] && (i * BLOCK_THREADS + tid < num_items)) {
            int hash = HASH(keys[i], ht_len, key_mins);
            // Atomically grab the next write slot within this hash bucket.
            sycl::atomic_ref<int,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                at(d_write_pos[hash]);
            int local_pos = at.fetch_add(1);
            int global_pos = d_offsets[hash] + local_pos;
            // Write payload
            payload[global_pos] = (int)vals[i];
            // Populate directory entry for this slot
            ht_dir[hash * 3 + 0] = (int)keys[i];
            ht_dir[hash * 3 + 1] = d_offsets[hash];
            ht_dir[hash * 3 + 2] = d_counts[hash];
        }
    }
}
