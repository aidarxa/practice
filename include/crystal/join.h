#pragma once
/** @file
 * @brief Perfect-domain PHT/MHT primitives using public PCUDA integer atomics.
 * @details Host code validates positive lengths, allocation bytes, collision-free
 * build domains and Int32 MHT totals. Build/count/scan/write/probe are distinct
 * kernels in one ordered stream: no concurrent build/probe publication is offered.
 * PHT occupancy is a separate zero-initialised UInt32 array, one word per slot.
 * Key zero is ordinary data. Only the winning claimant writes a key/payload.
 * The optional null occupancy retains the nonzero-key contract of the independent,
 * not-yet-migrated manual SSB examples; production SQL always supplies occupancy.
 */
#include "crystal/load.h"
#include "crystal/hash.h"
#include <pcuda.hpp>

namespace crystal {
/** @brief Claim raw aligned integer device storage using PCUDA's supported CAS words. */
template <typename K> inline K compareExchangeKey(K* target,K expected,K desired) {
    static_assert(std::is_integral<K>::value && (sizeof(K)==4 || sizeof(K)==8), "32/64-bit integral key required");
    if constexpr (sizeof(K)==4)
        return static_cast<K>(atomicCAS(reinterpret_cast<unsigned*>(target),static_cast<unsigned>(expected),static_cast<unsigned>(desired)));
    else
        return static_cast<K>(atomicCAS(reinterpret_cast<unsigned long long*>(target),static_cast<unsigned long long>(expected),static_cast<unsigned long long>(desired)));
}
/** @brief Occupancy, not a key sentinel, determines whether production PHT slots exist. */
template <typename K> inline bool phtPresent(K key,const K* table,std::size_t slot,const unsigned* occupied) {
    const bool present = occupied ? occupied[slot]!=0 : key!=0;
    return present && *table==key;
}
template <typename K> inline bool claimPht(K key,K* target,unsigned* occupied,std::size_t slot) {
    if (occupied) return atomicCAS(occupied+slot,0U,1U)==0U;
    return key!=0 && compareExchangeKey(target,K{},key)==K{};
}
/** @brief Probe either PHT layout without loading a payload during a count pass. */
template <unsigned Width,typename K>
inline bool phtContains(K key,const K* table,std::size_t length,K minimum,const unsigned* occupied) {
    if (!length) return false;
    const auto slot=hashSlot(key,length,minimum);
    return phtPresent(key,table+hashElement(slot,Width),slot,occupied);
}
/** @brief One owner publishes a complete entry before the next stream kernel. */
template <unsigned Width,typename K>
inline void buildPhtEntry(K key,K value,K* table,std::size_t length,K minimum,unsigned* occupied) {
    const auto slot=hashSlot(key,length,minimum);
    const auto base=hashElement(slot,Width);
    if (!claimPht(key,table+base,occupied,slot)) return;
    // In nonzero-key compatibility mode the CAS itself already stored the key.
    if (occupied) table[base]=key;
    if constexpr (Width==2) table[base+1]=value;
}
/** @brief Central active-lane traversal for join build and probe entry points. */
template <int Threads,int Items,typename F>
inline void selectedLanes(std::size_t tid,int (&flags)[Items],std::size_t count,F operation) {
    for (int i=0;i<Items;++i) {
        if (tid+static_cast<std::size_t>(i)*Threads >= count) flags[i]=0;
        else if (flags[i]) operation(i);
    }
}
/** @brief Common selected build traversal; Width fixes whether values are required. */
template <unsigned Width,int B,int I,typename K,typename V>
inline void buildSelectedPht(std::size_t tid,K (&keys)[I],const V* values,int (&flags)[I],K* table,
                              std::size_t length,K minimum,std::size_t count,unsigned* occupied) {
    if (!length) return;
    selectedLanes<B,I>(tid,flags,count,[&](int i) {
        K value{};
        if constexpr (Width==2) value=static_cast<K>(values[i]);
        buildPhtEntry<Width>(keys[i],value,table,length,minimum,occupied);
    });
}
/** @brief Single MHT row writer; directory fields have exactly one writer per bucket. */
template <typename K,typename V>
inline void writeMhtEntry(K key,V value,int* directory,const int* offsets,int* write_pos,const int* counts,
                          int* payload,std::size_t length,K minimum) {
    if (!length) return;
    const auto slot=hashSlot(key,length,minimum);
    const int position=atomicAdd(write_pos+slot,1);
    payload[static_cast<std::size_t>(offsets[slot])+static_cast<unsigned>(position)]=static_cast<int>(value);
    if (position==0) {
        const auto base=hashElement(slot,3);
        directory[base]=static_cast<int>(key);
        directory[base+1]=offsets[slot];
        directory[base+2]=counts[slot];
    }
}
/** @brief Count one row in the same perfect-domain directory used by writeMhtEntry. */
template <typename K>
inline void countMhtEntry(K key,int* counts,std::size_t length,K minimum) {
    if (!length) return;
    atomicAdd(counts+hashSlot(key,length,minimum),1);
}
}

/** @brief Key-only PHT lookup; an unoccupied slot never matches zero. */
template <typename K> inline bool ProbePHT_1(K key,const K* table,std::size_t length,K minimum,const unsigned* occupied=nullptr) {
    return crystal::phtContains<1>(key,table,length,minimum,occupied);
}
/** @brief Interleaved key/payload PHT lookup with full-width element offsets. */
template <typename K,typename V> inline bool ProbePHT_2(K key,V& result,const K* table,std::size_t length,K minimum,const unsigned* occupied=nullptr) {
    if (!length) return false;
    const auto slot=crystal::hashSlot(key,length,minimum);
    const auto base=crystal::hashElement(slot,2);
    if (!crystal::phtPresent(key,table+base,slot,occupied)) return false;
    result=static_cast<V>(table[base+1]);
    return true;
}

namespace crystal {
/** @brief Compile-time payload policy retains the key-only projection count fast path. */
template <bool ReadValues,int B,int I,typename K,typename V>
inline void probeSelectedPht(std::size_t tid,K (&keys)[I],V* values,int (&flags)[I],const K* table,
                              std::size_t length,K minimum,std::size_t count,const unsigned* occupied) {
    selectedLanes<B,I>(tid,flags,count,[&](int i) {
        if constexpr (ReadValues) flags[i]=ProbePHT_2(keys[i],values[i],table,length,minimum,occupied);
        else flags[i]=phtContains<2>(keys[i],table,length,minimum,occupied);
    });
}
}

template <typename K,int B,int I>
inline void BlockProbeAndPHT_1(std::size_t tid,K (&keys)[I],int (&flags)[I],const K* table,std::size_t length,K minimum,std::size_t count,const unsigned* occupied=nullptr) {
    crystal::selectedLanes<B,I>(tid,flags,count,[&](int i) { flags[i]=ProbePHT_1(keys[i],table,length,minimum,occupied); });
}
template <typename K,typename V,int B,int I>
inline void BlockProbeAndPHT_2(std::size_t tid,K (&keys)[I],V (&values)[I],int (&flags)[I],const K* table,std::size_t length,K minimum,std::size_t count,const unsigned* occupied=nullptr) {
    crystal::probeSelectedPht<true,B,I>(tid,keys,values,flags,table,length,minimum,count,occupied);
}
template <typename K,int B,int I>
inline void BlockProbeAndPHT_2KeyOnly(std::size_t tid,K (&keys)[I],int (&flags)[I],const K* table,std::size_t length,K minimum,std::size_t count,const unsigned* occupied=nullptr) {
    crystal::probeSelectedPht<false,B,I>(tid,keys,static_cast<K*>(nullptr),flags,table,length,minimum,count,occupied);
}

/** @brief PHT build; duplicate claims cannot race on payload writes. */
template <typename K,int B,int I>
inline void BlockBuildSelectivePHT_1(std::size_t tid,K (&keys)[I],int (&flags)[I],K* table,std::size_t length,K minimum,std::size_t count,unsigned* occupied=nullptr) {
    crystal::buildSelectedPht<1,B,I>(tid,keys,static_cast<const K*>(nullptr),flags,table,length,minimum,count,occupied);
}
template <typename K,typename V,int B,int I>
inline void BlockBuildSelectivePHT_2(std::size_t tid,K (&keys)[I],V (&values)[I],int (&flags)[I],K* table,std::size_t length,K minimum,std::size_t count,unsigned* occupied=nullptr) {
    crystal::buildSelectedPht<2,B,I>(tid,keys,values,flags,table,length,minimum,count,occupied);
}

/** @name Nonzero-key manual-example compatibility overloads (minimum=0). */
/// @{
template <typename K,int B,int I> inline void BlockProbeAndPHT_1(std::size_t t,K (&k)[I],int (&f)[I],const K* h,std::size_t l,std::size_t n) { BlockProbeAndPHT_1<K,B,I>(t,k,f,h,l,K{},n); }
template <typename K,typename V,int B,int I> inline void BlockProbeAndPHT_2(std::size_t t,K (&k)[I],V (&v)[I],int (&f)[I],const K* h,std::size_t l,std::size_t n) { BlockProbeAndPHT_2<K,V,B,I>(t,k,v,f,h,l,K{},n); }
template <typename K,int B,int I> inline void BlockProbeAndPHT_2KeyOnly(std::size_t t,K (&k)[I],int (&f)[I],const K* h,std::size_t l,std::size_t n) { BlockProbeAndPHT_2KeyOnly<K,B,I>(t,k,f,h,l,K{},n); }
template <typename K,int B,int I> inline void BlockBuildSelectivePHT_1(std::size_t t,K (&k)[I],int (&f)[I],K* h,std::size_t l,std::size_t n) { BlockBuildSelectivePHT_1<K,B,I>(t,k,f,h,l,K{},n); }
template <typename K,typename V,int B,int I> inline void BlockBuildSelectivePHT_2(std::size_t t,K (&k)[I],V (&v)[I],int (&f)[I],K* h,std::size_t l,std::size_t n) { BlockBuildSelectivePHT_2<K,V,B,I>(t,k,v,f,h,l,K{},n); }
/// @}

/** @brief MHT directory lookup; counts establish presence, so zero keys need no sentinel. */
template <typename K>
inline void ProbeMultiHT(K key,int& offset,int& count,int& flag,const K* directory,std::size_t length,K minimum) {
    count=0;
    offset=0;
    if (!flag || !length) { flag=0; return; }
    const auto base=crystal::hashElement(crystal::hashSlot(key,length,minimum),3);
    if (directory[base]!=key || directory[base+2]<=0) { flag=0; return; }
    offset=static_cast<int>(directory[base+1]);
    count=static_cast<int>(directory[base+2]);
}

/** @brief Count selected rows. Host must prove total build rows <= INT32_MAX. */
template <typename K,int B,int I>
inline void BlockBuildMHT_Count(std::size_t tid,K (&keys)[I],int (&flags)[I],int* counts,std::size_t length,K minimum,std::size_t count) {
    crystal::selectedLanes<B,I>(tid,flags,count,[&](int i) { crystal::countMhtEntry(keys[i],counts,length,minimum); });
}

/** @brief Claim unique payload positions; exactly one writer populates each directory entry.
 * @pre Counts/offsets derive from the same selection, write_pos is zeroed, and
 * payload has sum(counts) entries. The host proves offsets/counts fit Int32.
 */
template <typename K,typename V,int B,int I>
inline void BlockBuildMHT_Write(std::size_t tid,K (&keys)[I],V (&values)[I],int (&flags)[I],int* directory,
                                const int* offsets,int* write_pos,const int* counts,int* payload,
                                std::size_t length,K minimum,std::size_t count) {
    crystal::selectedLanes<B,I>(tid,flags,count,[&](int i) {
        crystal::writeMhtEntry(keys[i],values[i],directory,offsets,write_pos,counts,payload,length,minimum);
    });
}
