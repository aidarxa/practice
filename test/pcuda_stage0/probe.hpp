#pragma once

#include "pch.hpp"
#include <limits>

// Experimental test interface, NOT the proposed SQL module ABI.
using ProbeLaunch = pcudaError_t (*)(pcudaStream_t, std::uint32_t*, std::size_t,
                                    unsigned, std::uint32_t);

inline bool validProbeLaunch(pcudaStream_t stream, const void* data,
                             std::size_t count, unsigned block) {
    return stream && data && block &&
           count / block + (count % block != 0) <=
               std::numeric_limits<unsigned>::max();
}
