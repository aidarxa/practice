/** @file @brief CPU oracles for index arithmetic, lane masks and scan layout (not device coverage). */
#include "crystal/load.h"
#include "crystal/pred.h"
#include "crystal/hash.h"
#include "core/inline_math.h"
#include "core/scan_layout.h"
#include <cassert>
#include <limits>
#include <iostream>

static void loads() {
    const std::int64_t input[] = {9007199254740993LL, -7, 42};
    std::int64_t items[2] = {-1,-1};
    BlockLoad<std::int64_t,2,2>(input, 0, 0, items, 3);
    assert(items[0] == input[0] && items[1] == 42);
    BlockLoad<std::int64_t,2,2>(input, 1, 0, items, 3);
    assert(items[0] == -7 && items[1] == 0);
    BlockLoad<std::int64_t,2,2>(nullptr, 0, 0, items, 0);
    assert(items[0] == 0 && items[1] == 0);
    std::uint64_t rows[2] = {};
    BlockMakeRowIds<2,2>(1, 4294967296ULL, rows, 3);
    assert(rows[0] == 4294967297ULL && rows[1] == 0);
    int ids[2] = {2,0}, flags[2] = {1,1};
    BlockGather<std::int64_t,2,2>(input, 1, ids, flags, items, 3);
    assert(items[0] == 42 && items[1] == 0);
    const uint64_t bitmap[] = {1ULL << 63, 1};
    int valid[2] = {};
    BlockLoadValidity<1,2>(bitmap, 0, 63, valid, 2);
    assert(valid[0] == 1 && valid[1] == 1);
    assert(ColumnValidAt(nullptr, 4294967296ULL) == 1);
    BlockGatherValidity<2,2>(bitmap, 1, ids, flags, valid, 3);
    assert(valid[0] == 0 && valid[1] == 0);
    BlockApplyValidityAnd<2,2>(1, flags, valid, 3);
    assert(flags[0] == 0 && flags[1] == 0);
}

static void predicates() {
    int items[2] = {3,9}, flags[2] = {1,1}, other[2] = {3,1};
    BlockPredLT<int,2,2>(1, items, flags, 4, 3);
    assert(flags[0] == 1 && flags[1] == 0);
    BlockPredAGT<int,2,2>(1, items, flags, 4, 3);
    assert(flags[0] == 0 && flags[1] == 0);
    BlockPredOLT<int,2,2>(1, items, flags, 4, 3);
    assert(flags[0] == 1 && flags[1] == 0);
    BlockPredEq_Cols<int,2,2>(1, items, other, flags, 3);
    assert(flags[0] == 1 && flags[1] == 0);
    BlockPredOrGT_Cols<int,2,2>(1, items, other, flags, 0);
    assert(flags[0] == 0 && flags[1] == 0);
    InitFlagsZero<2,2>(flags);
    InitFlags<2,2>(other);
    BlockApplyMaskOr<2,2>(0,flags,other);
    assert(flags[0] == 1 && flags[1] == 1);
    BlockApplyMaskAnd<2,2>(0,flags,other);
    assert(flags[0] == 1 && flags[1] == 1);
}

/** @brief Exercise all public comparison adapters with independent literal truth tables. */
static void comparisonAdapters() {
    using Scalar = void(*)(size_t,int (&)[2],int (&)[2],int,size_t);
    const Scalar scalar[] = {
        BlockPredLT<int,2,2>,BlockPredALT<int,2,2>,BlockPredOLT<int,2,2>,
        BlockPredGT<int,2,2>,BlockPredAGT<int,2,2>,BlockPredOGT<int,2,2>,
        BlockPredLTE<int,2,2>,BlockPredALTE<int,2,2>,BlockPredOLTE<int,2,2>,
        BlockPredGTE<int,2,2>,BlockPredAGTE<int,2,2>,BlockPredOGTE<int,2,2>,
        BlockPredEq<int,2,2>,BlockPredAEq<int,2,2>,BlockPredOrEq<int,2,2>,
        BlockPredNEq<int,2,2>,BlockPredANEq<int,2,2>,BlockPredOrNEq<int,2,2>};
    using Columns = void(*)(size_t,int (&)[2],int (&)[2],int (&)[2],size_t);
    const Columns columns[] = {
        BlockPredLT_Cols<int,2,2>,BlockPredALT_Cols<int,2,2>,BlockPredOrLT_Cols<int,2,2>,
        BlockPredGT_Cols<int,2,2>,BlockPredAGT_Cols<int,2,2>,BlockPredOrGT_Cols<int,2,2>,
        BlockPredLTE_Cols<int,2,2>,BlockPredALTE_Cols<int,2,2>,BlockPredOrLTE_Cols<int,2,2>,
        BlockPredGTE_Cols<int,2,2>,BlockPredAGTE_Cols<int,2,2>,BlockPredOrGTE_Cols<int,2,2>,
        BlockPredEq_Cols<int,2,2>,BlockPredAEq_Cols<int,2,2>,BlockPredOrEq_Cols<int,2,2>,
        BlockPredNEq_Cols<int,2,2>,BlockPredANEq_Cols<int,2,2>,BlockPredOrNEq_Cols<int,2,2>};
    const bool truth[][2]={{false,false},{false,true},{true,false},{true,true},{true,false},{false,true}};
    for (size_t adapter=0;adapter<18;++adapter) {
        for (int seed=0;seed<=1;++seed) for (size_t count=0;count<=4;++count) {
            int values[2]={3,9},other[2]={3,1},flags[2]={seed,seed},paired[2]={seed,seed};
            scalar[adapter](1,values,flags,3,count);
            columns[adapter](1,values,other,paired,count);
            for (size_t i=0;i<2;++i) {
                bool expected=truth[adapter/3][i];
                if (adapter%3==1) expected=seed && expected;
                if (adapter%3==2) expected=seed || expected;
                if (1+i*2>=count) expected=false;
                assert(flags[i]==expected && paired[i]==expected);
            }
        }
    }
    int values[2]={3,9},flags[2]={};
    BlockPredDirect<int,Equal<int>,2,2>(0,values,flags,Equal<int>(3));
    assert(flags[0]==1 && flags[1]==0);
    BlockPredAndDirect<int,GreaterThanEq<int>,2,2>(0,values,flags,GreaterThanEq<int>(4));
    assert(flags[0]==0 && flags[1]==0);
    BlockPredOrDirect<int,NotEqual<int>,2,2>(0,values,flags,NotEqual<int>(3));
    assert(flags[0]==0 && flags[1]==1);
}

static void indices() {
    using crystal::hashSlot;
    assert(hashSlot(1, 8, 0) == 1 && hashSlot(8, 8, 0) == 0);
    assert(hashSlot(-1, 8, 0) == 7 && hashSlot(0,8,-3) == 3);
    assert(hashSlot(INT32_MAX, 7, INT32_MIN) == 3);
    assert(hashSlot(INT64_MAX, 7, INT64_MIN) == 1);
    assert(hashSlot(INT64_MIN, 7, INT64_MAX) == 6);
    assert(crystal::hashElement(2147483646ULL, 3, 2) == 6442450940ULL);
    assert(db::safe_mul(INT32_MAX, INT32_MAX) == 4611686014132420609LL);
    assert(db::safe_sub(INT32_MIN, INT32_MAX) == -4294967295LL);
    assert(db::safe_add(INT32_MAX,1)==2147483648LL && db::safe_div(7,2)==3);
    assert(db::safe_div(7,2.0)==3.5 && db::safe_mul(2.5f,4)==10.0);
    assert(db::safe_lt(1,2) && !db::safe_lt(2,1));
    assert(db::safe_gt(2,1) && !db::safe_gt(1,2));
    assert(db::safe_lte(1,1) && !db::safe_lte(2,1));
    assert(db::safe_gte(1,1) && !db::safe_gte(1,2));
    assert(db::safe_eq(1,1) && !db::safe_eq(1,2));
    assert(db::safe_neq(1,2) && !db::safe_neq(1,1));
    assert(db::bitmap_valid_at(nullptr,UINT64_MAX)==1);
    const uint64_t bitmap[]={0,1};
    assert(db::bitmap_valid_at(bitmap,64)==1 && db::bitmap_valid_at(bitmap,63)==0);
    const double minus_zero = -0.0;
    assert(db::bit_cast_double_to_ull(minus_zero) == (1ULL << 63));
    assert(db::bit_cast_ull_to_double(0x3ff8000000000000ULL) == 1.5);
}

static void layout() {
    assert(db::makeScanLayout(0,256).levels.empty());
    const auto small = db::makeScanLayout(257,256);
    assert(small.levels.size() == 2 && small.levels[0].blocks == 2);
    assert(small.levels[1].items == 2 && small.scratch_elements == 3);
    const auto deep = db::makeScanLayout(256ULL*256*256+1,256);
    assert(deep.levels.size() == 4 && deep.levels[0].blocks == 65537);
    const auto large = db::makeScanLayout(4294967297ULL,256);
    assert(large.levels[0].items == 4294967297ULL);
    for (auto block : {0U,1U,1025U}) {
        bool rejected = false;
        try { db::makeScanLayout(2,block); } catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    bool overflow = false;
    try { db::makeScanLayout(UINT64_MAX,256); } catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
}

int main() {
    loads(); predicates(); comparisonAdapters(); indices(); layout();
    std::cout << "host primitive contracts passed\n";
}
