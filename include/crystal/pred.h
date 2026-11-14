#pragma once
/** @file @brief Striped predicates: replace/intersect/union masks, with invalid tails cleared. */
#include "crystal/load.h"

template <typename T> struct GreaterThan { T _compare; explicit GreaterThan(T v):_compare(v){} bool operator()(T v) const { return v > _compare; } };
template <typename T> struct LessThan { T _compare; explicit LessThan(T v):_compare(v){} bool operator()(T v) const { return v < _compare; } };
template <typename T> struct GreaterThanEq { T _compare; explicit GreaterThanEq(T v):_compare(v){} bool operator()(T v) const { return v >= _compare; } };
template <typename T> struct LessThanEq { T _compare; explicit LessThanEq(T v):_compare(v){} bool operator()(T v) const { return v <= _compare; } };
template <typename T> struct Equal { T _compare; explicit Equal(T v):_compare(v){} bool operator()(T v) const { return v == _compare; } };
template <typename T> struct NotEqual { T _compare; explicit NotEqual(T v):_compare(v){} bool operator()(T v) const { return v != _compare; } };

namespace crystal {
/** @brief Compile-time mask policy, not a runtime SQL operator dispatch. */
enum class Mask { Replace, And, Or };
template <Mask Mode> inline int combineMask(int old, int value) {
    if constexpr (Mode == Mask::And) return old && value;
    if constexpr (Mode == Mask::Or) return old || value;
    return value;
}
/** @brief One bounds/mask policy for scalar and column comparisons. */
template <Mask Mode,int Threads,int Items,typename Predicate>
inline void applyMask(std::size_t tid,int (&flags)[Items],std::size_t count,Predicate predicate) {
    transformTile<Threads,Items>(tid,flags,count,[&](std::size_t,int i) {
        return combineMask<Mode>(Mode==Mask::Replace?0:flags[i],predicate(i));
    });
}
/** @brief Scalar predicate adapter; no predicate reads a padded lane. */
template <Mask Mode, int Threads, int Items, typename T, typename Predicate>
inline void applyPredicate(std::size_t tid, const T (&items)[Items], int (&flags)[Items], Predicate pred, std::size_t count) {
    applyMask<Mode,Threads,Items>(tid,flags,count,[&](int i) { return pred(items[i]); });
}
/** @brief Column-column form shares the same mask and tail semantics. */
template <Mask Mode, template<typename> class Predicate, int Threads, int Items, typename T>
inline void applyColumnPredicate(std::size_t tid, const T (&left)[Items], const T (&right)[Items], int (&flags)[Items], std::size_t count) {
    applyMask<Mode,Threads,Items>(tid,flags,count,[&](int i) { return Predicate<T>(right[i])(left[i]); });
}
}

template <int Threads,int Items> inline void InitFlags(int (&flags)[Items]) { for (auto& flag : flags) flag = 1; }
template <int Threads,int Items> inline void InitFlagsZero(int (&flags)[Items]) { for (auto& flag : flags) flag = 0; }
template <int Threads,int Items> inline void BlockApplyMaskOr(std::size_t, int (&target)[Items], const int (&source)[Items]) { for (int i=0;i<Items;++i) target[i] = target[i] || source[i]; }
template <int Threads,int Items> inline void BlockApplyMaskAnd(std::size_t, int (&target)[Items], const int (&source)[Items]) { for (int i=0;i<Items;++i) target[i] = target[i] && source[i]; }

template <typename T,typename P,int B,int I> inline void BlockPred(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n) { crystal::applyPredicate<crystal::Mask::Replace,B,I>(t,v,f,p,n); }
template <typename T,typename P,int B,int I> inline void BlockPredAnd(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n) { crystal::applyPredicate<crystal::Mask::And,B,I>(t,v,f,p,n); }
template <typename T,typename P,int B,int I> inline void BlockPredOr(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n) { crystal::applyPredicate<crystal::Mask::Or,B,I>(t,v,f,p,n); }
template <typename T,typename P,int B,int I> inline void BlockPredDirect(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n=crystal::tileItems<B,I>()) { BlockPred<T,P,B,I>(t,v,f,p,n); }
template <typename T,typename P,int B,int I> inline void BlockPredAndDirect(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n=crystal::tileItems<B,I>()) { BlockPredAnd<T,P,B,I>(t,v,f,p,n); }
template <typename T,typename P,int B,int I> inline void BlockPredOrDirect(std::size_t t,T (&v)[I],int (&f)[I],P p,std::size_t n=crystal::tileItems<B,I>()) { BlockPredOr<T,P,B,I>(t,v,f,p,n); }

/** @name Scalar comparison entry points retained for fused SQL codegen. */
/// @{
template <typename T,int B,int I> inline void BlockPredLT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,LessThan<T>,B,I>(t,v,f,LessThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredALT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,LessThan<T>,B,I>(t,v,f,LessThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOLT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,LessThan<T>,B,I>(t,v,f,LessThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredGT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,GreaterThan<T>,B,I>(t,v,f,GreaterThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredAGT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,GreaterThan<T>,B,I>(t,v,f,GreaterThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOGT(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,GreaterThan<T>,B,I>(t,v,f,GreaterThan<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredLTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,LessThanEq<T>,B,I>(t,v,f,LessThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredALTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,LessThanEq<T>,B,I>(t,v,f,LessThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOLTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,LessThanEq<T>,B,I>(t,v,f,LessThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredGTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,GreaterThanEq<T>,B,I>(t,v,f,GreaterThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredAGTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,GreaterThanEq<T>,B,I>(t,v,f,GreaterThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOGTE(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,GreaterThanEq<T>,B,I>(t,v,f,GreaterThanEq<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,Equal<T>,B,I>(t,v,f,Equal<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredAEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,Equal<T>,B,I>(t,v,f,Equal<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOrEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,Equal<T>,B,I>(t,v,f,Equal<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredNEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPred<T,NotEqual<T>,B,I>(t,v,f,NotEqual<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredANEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredAnd<T,NotEqual<T>,B,I>(t,v,f,NotEqual<T>(c),n); }
template <typename T,int B,int I> inline void BlockPredOrNEq(std::size_t t,T (&v)[I],int (&f)[I],T c,std::size_t n) { BlockPredOr<T,NotEqual<T>,B,I>(t,v,f,NotEqual<T>(c),n); }
/// @}

/** @name Column comparison entry points; A/Or combine with the incoming mask. */
/// @{
template <typename T,int B,int I> inline void BlockPredEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,Equal,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredAEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,Equal,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,Equal,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredNEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,NotEqual,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredANEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,NotEqual,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrNEq_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,NotEqual,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredLT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,LessThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredALT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,LessThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrLT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,LessThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredLTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,LessThanEq,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredALTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,LessThanEq,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrLTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,LessThanEq,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredGT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,GreaterThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredAGT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,GreaterThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrGT_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,GreaterThan,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredGTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Replace,GreaterThanEq,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredAGTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::And,GreaterThanEq,B,I>(t,a,b,f,n); }
template <typename T,int B,int I> inline void BlockPredOrGTE_Cols(std::size_t t,T (&a)[I],T (&b)[I],int (&f)[I],std::size_t n) { crystal::applyColumnPredicate<crystal::Mask::Or,GreaterThanEq,B,I>(t,a,b,f,n); }
/// @}
