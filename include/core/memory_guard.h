#pragma once
/** @file @brief One checked policy for resident device, pinned and transient bytes. */
#include "core/config.h"
#include <cstdint>

namespace db {
/** @brief Total memory minus the larger absolute/fractional reserve, clamped to zero. */
uint64_t deviceMemoryBudget(uint64_t total, const CrystalConfig& policy);
/** @brief Check peak resident bytes; disabling the budget never disables overflow checks. */
void checkDeviceMemoryRequest(uint64_t total, uint64_t resident, uint64_t additional,
                              const CrystalConfig& policy);
/** @brief Grow small reusable buffers by 1.5x, but allocate large analytical buffers exactly. */
size_t deviceBufferCapacity(size_t requested);
}
