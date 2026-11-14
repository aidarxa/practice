#pragma once
/** @file @brief Full-width, signed-key-safe indexing for perfect-domain hash tables. */
#include <cstddef>
#include <type_traits>

namespace crystal {
/** @brief Euclidean modulo of key-min without signed subtraction overflow.
 * @pre length > 0. Build keys must have distinct residues (MHT: distinct keys).
 * Probe keys may be outside the build domain; comparing stored keys rejects aliases.
 */
template <typename K>
inline std::size_t hashSlot(K key, std::size_t length, K minimum) {
    static_assert(std::is_integral<K>::value && (sizeof(K)==4 || sizeof(K)==8), "32/64-bit integral hash key required");
    using U = std::make_unsigned_t<K>;
    if (key >= minimum) return (static_cast<U>(key) - static_cast<U>(minimum)) % length;
    const auto remainder = (static_cast<U>(minimum) - static_cast<U>(key)) % length;
    return remainder == 0 ? 0 : length - remainder;
}
/** @brief Widen before interleaved table indexing; host validates allocation extent. */
inline std::size_t hashElement(std::size_t slot, std::size_t width, std::size_t field = 0) {
    return slot * width + field;
}
}
