#pragma once
/** @file @brief Test-only prebuilt kernel boundary, not the SQL module ABI. */
#include <cstddef>
struct BenchCall {
    void* stream;
    void* legacy_queue;
    int* input;
    int* keys;
    int* table;
    unsigned* occupied;
    unsigned long long* output;
    std::size_t count;
    std::size_t key_count;
};
void benchmarkLaunch(unsigned operation,const BenchCall& call);
