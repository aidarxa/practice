#pragma once
/** @file
 * @brief Striped loads, gathers and validity masks shared by fused PCUDA kernels.
 * @details The input of BlockLoad is already tile-relative; tile_offset is kept
 * for source compatibility and is not added a second time. Inactive lanes are
 * always zero, so a later predicate cannot read uninitialised registers.
 * Host launch code validates extents and representability of row IDs beforehand.
 */
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace crystal {
/** @brief Widen before multiplying; template tile sizes must fit size_t. */
template <int Threads, int Items>
constexpr std::size_t tileItems() {
    static_assert(Threads > 0 && Items > 0, "positive tile shape required");
    static_assert(static_cast<std::size_t>(Threads) <= SIZE_MAX / Items, "tile size overflow");
    return static_cast<std::size_t>(Threads) * Items;
}
/** @brief One bounds policy for striped transforms; a reader is called only for valid lanes. */
template <int Threads,int Items,typename T,typename Reader>
inline void transformTile(std::size_t tid,T (&output)[Items],std::size_t count,Reader read) {
    for (int i=0;i<Items;++i) {
        const auto lane=tid+static_cast<std::size_t>(i)*Threads;
        output[i]=lane<count?read(lane,i):T{};
    }
}
/** @brief Selected gather policy shared by values and validity. */
template <int Threads,int Items,typename T,typename Row,typename Reader>
inline void gatherTile(std::size_t tid,T (&output)[Items],const Row (&rows)[Items],const int (&flags)[Items],std::size_t count,Reader read) {
    transformTile<Threads,Items>(tid,output,count,[&](std::size_t,int i) { return flags[i]?read(rows[i]):T{}; });
}
}

/** @brief Full tile load; every participating lane reads Items striped elements. */
template <typename T, int Threads, int Items>
inline void BlockLoadDirect(const T* input, std::size_t tid, T (&items)[Items]) {
    for (int i = 0; i < Items; ++i) items[i] = input[tid + static_cast<std::size_t>(i) * Threads];
}

/** @brief Partial tile load, including an empty tile without pointer arithmetic on null. */
template <typename T, int Threads, int Items>
inline void BlockLoadDirect(const T* input, std::size_t tid, T (&items)[Items], std::size_t count) {
    crystal::transformTile<Threads,Items>(tid,items,count,[&](std::size_t lane,int) { return input[lane]; });
}

/** @brief Select full/partial load without narrowing values to int. */
template <typename T, int Threads, int Items>
inline void BlockLoad(const T* input, std::size_t tid, std::size_t /*tile_offset*/,
                      T (&items)[Items], std::size_t count) {
    if (count == crystal::tileItems<Threads,Items>()) BlockLoadDirect<T,Threads,Items>(input,tid,items);
    else BlockLoadDirect<T,Threads,Items>(input,tid,items,count);
}

/** @brief Gather only active lanes; each active row ID must be in the input extent. */
template <typename T, int Threads, int Items, typename Row>
inline void BlockGather(const T* input, std::size_t tid, const Row (&rows)[Items],
                        const int (&flags)[Items], T (&items)[Items], std::size_t count) {
    crystal::gatherTile<Threads,Items>(tid,items,rows,flags,count,[&](Row row) { return input[row]; });
}

/** @brief Generate row IDs in their requested width after host-side extent validation. */
template <int Threads, int Items, typename Row>
inline void BlockMakeRowIds(std::size_t tid, std::size_t offset, Row (&rows)[Items], std::size_t count) {
    static_assert(std::is_integral<Row>::value, "integral row IDs required");
    crystal::transformTile<Threads,Items>(tid,rows,count,[&](std::size_t lane,int) { return static_cast<Row>(offset + lane); });
}

/** @brief Null bitmap means nonnullable; a set bit means SQL-valid. */
inline int ColumnValidAt(const std::uint64_t* bitmap, std::size_t row) {
    return bitmap ? static_cast<int>((bitmap[row >> 6] >> (row & 63)) & 1ULL) : 1;
}

/** @brief Load validity using full-width row offsets; padded lanes are invalid. */
template <int Threads, int Items>
inline void BlockLoadValidity(const std::uint64_t* bitmap, std::size_t tid, std::size_t offset,
                              int (&valid)[Items], std::size_t count) {
    crystal::transformTile<Threads,Items>(tid,valid,count,[&](std::size_t lane,int) { return ColumnValidAt(bitmap,offset+lane); });
}

/** @brief Gather validity only for selected rows. */
template <int Threads, int Items, typename Row>
inline void BlockGatherValidity(const std::uint64_t* bitmap, std::size_t tid, const Row (&rows)[Items],
                                const int (&flags)[Items], int (&valid)[Items], std::size_t count) {
    crystal::gatherTile<Threads,Items>(tid,valid,rows,flags,count,[&](Row row) { return ColumnValidAt(bitmap,row); });
}

/** @brief Intersect a selection with validity and clear padded lanes. */
template <int Threads, int Items>
inline void BlockApplyValidityAnd(std::size_t tid, int (&flags)[Items], const int (&valid)[Items], std::size_t count) {
    for (int i = 0; i < Items; ++i)
        flags[i] = tid + static_cast<std::size_t>(i) * Threads < count && flags[i] && valid[i];
}
