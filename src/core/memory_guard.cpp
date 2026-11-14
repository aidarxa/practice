#include "core/memory_guard.h"
#include "core/size_expression.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace db {
uint64_t deviceMemoryBudget(uint64_t total, const CrystalConfig& policy) {
    const auto fraction = policy.memory_guard_reserve_fraction;
    if (!std::isfinite(fraction) || fraction < 0 || fraction > 0.9)
        throw std::runtime_error("invalid memory guard reserve fraction");
    const auto reserve = std::max<uint64_t>(policy.memory_guard_reserve_bytes,
        static_cast<uint64_t>(static_cast<long double>(total) * fraction));
    return total > reserve ? total - reserve : 0;
}
void checkDeviceMemoryRequest(uint64_t total, uint64_t resident, uint64_t additional,
                              const CrystalConfig& policy) {
    const auto required = checkedSizeAdd(resident, additional);
    const auto budget = deviceMemoryBudget(total, policy);
    if (policy.memory_guard_enabled && required > budget)
        throw std::runtime_error("Insufficient GPU memory: resident high-water requirement " +
            std::to_string(required) + " bytes exceeds safety budget " + std::to_string(budget) + " bytes");
}
size_t deviceBufferCapacity(size_t requested) {
    return requested <= (1ULL << 20) ? checkedSizeAdd(requested, requested / 2) : requested;
}
}
