#pragma once
/** @file @brief Checked host layout for an arbitrary-depth exclusive-sum hierarchy. */
#include "core/size_expression.h"
#include <vector>
#include <stdexcept>

namespace db {
/** @brief One scan launch and its block-total segment in runtime-owned scratch. */
struct ScanLevel { uint64_t items; uint64_t blocks; uint64_t offset; };
/** @brief Empty input has no launches and no scratch. */
struct ScanLayout { std::vector<ScanLevel> levels; uint64_t scratch_elements = 0; };
/** @brief Compute capacities without rounded-up additions or fixed level limits.
 * @throws std::overflow_error if any input/scratch byte extent is unrepresentable.
 * @throws std::invalid_argument for a nonshrinking/unsupported block size.
 */
inline ScanLayout makeScanLayout(uint64_t items, unsigned block) {
    if (block < 2 || block > 1024) throw std::invalid_argument("scan block must be in 2..1024");
    checkedSizeMultiply(items, sizeof(unsigned long long));
    ScanLayout layout;
    while (items) {
        const auto blocks = items / block + (items % block != 0);
        layout.levels.push_back({items, blocks, layout.scratch_elements});
        layout.scratch_elements = checkedSizeAdd(layout.scratch_elements, blocks);
        if (blocks == 1) break;
        items = blocks;
    }
    checkedSizeMultiply(layout.scratch_elements, sizeof(unsigned long long));
    return layout;
}
}
