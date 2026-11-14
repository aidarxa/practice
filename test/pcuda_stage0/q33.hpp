#pragma once

#include "pch.hpp"
#include <limits>

namespace q33 {
constexpr unsigned block = 128;
constexpr unsigned items = 4;

// Experimental benchmark arguments; not the future versioned SQL ABI.
struct BuildArgs {
    const int* keys;
    const int* values;
    int* table;
    std::size_t count;
    std::size_t slots;
    int key_min;
    int first;
    int last;
};
struct ProbeArgs {
    const int* dates;
    const int* customers;
    const int* suppliers;
    const int* revenue;
    const int* date_table;
    const int* customer_table;
    const int* supplier_table;
    unsigned long long* result;
    std::size_t count;
    std::size_t date_slots;
    std::size_t customer_slots;
    std::size_t supplier_slots;
    int date_min;
    int customer_min;
    int supplier_min;
    int year_first;
    int year_last;
    int city_first;
    int city_last;
};
using BuildLaunch = pcudaError_t (*)(pcudaStream_t, BuildArgs);
using ProbeLaunch = pcudaError_t (*)(pcudaStream_t, ProbeArgs);

inline bool validCount(std::size_t count) {
    constexpr std::size_t tile = block * items;
    return count / tile + (count % tile != 0) <= std::numeric_limits<unsigned>::max();
}
} // namespace q33
