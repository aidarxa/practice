#pragma once
/** @file @brief Deterministic asynchronous PCUDA double, never used by the engine. */
#include <string>
#include <vector>

namespace fake_pcuda {
void reset();
void failAllocation();
void nullAllocation();
void failCopy();
void failWait(bool persistent = false);
void duplicateGpu(bool enabled);
void emptyFirstPlatform(bool enabled);
void capabilities(bool fp64, bool atomic64);
const std::vector<std::string>& events();
std::size_t allocations();
}
