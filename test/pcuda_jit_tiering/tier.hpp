#pragma once
#include <pcuda.hpp>
#include <cstddef>

// A test-only POD contract, not the proposed SQL module ABI.
struct TierArgs {
    const long long* values;
    const unsigned char* valid;
    long long* output;
    std::size_t count;
    long long threshold;
    long long offset;
};
using TierLaunch = pcudaError_t (*)(pcudaStream_t, TierArgs);
