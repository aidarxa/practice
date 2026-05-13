#include "crystal/utils.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static bool bit_at(const std::vector<uint64_t>& words, std::size_t row) {
    return ((words[row >> 6U] >> (row & 63U)) & 1ULL) != 0ULL;
}

static void test_valid8_decode() {
    std::vector<unsigned char> bytes = {1, 0, 1, 0, 1};
    auto validity = decodeValidityByteBitmap(bytes, bytes.size(), true);
    assert(validity.size() == 1);
    assert(bit_at(validity, 0));
    assert(!bit_at(validity, 1));
    assert(bit_at(validity, 2));
    assert(!bit_at(validity, 3));
    assert(bit_at(validity, 4));
    assert((validity[0] & ~0x1FULL) == 0ULL && "unused high bits must be masked");
}

static void test_null8_decode() {
    std::vector<unsigned char> bytes = {0, 1, 0, 1, 0};
    auto validity = decodeValidityByteBitmap(bytes, bytes.size(), false);
    assert(validity.size() == 1);
    assert(bit_at(validity, 0));
    assert(!bit_at(validity, 1));
    assert(bit_at(validity, 2));
    assert(!bit_at(validity, 3));
    assert(bit_at(validity, 4));
}

static void test_null64_decode_and_mask() {
    // Rows 1 and 3 are NULL; rows 0,2,4 are valid. Unused bits must be zero.
    std::vector<uint64_t> null_words = {0b01010ULL};
    auto validity = decodeValidityWordBitmap(null_words, 5, false);
    assert(validity.size() == 1);
    assert(bit_at(validity, 0));
    assert(!bit_at(validity, 1));
    assert(bit_at(validity, 2));
    assert(!bit_at(validity, 3));
    assert(bit_at(validity, 4));
    assert((validity[0] & ~0x1FULL) == 0ULL && "unused high bits must be masked after inversion");
}

static void test_invalid_byte_count_throws() {
    bool thrown = false;
    try {
        (void)decodeValidityByteBitmap(std::vector<unsigned char>{1, 0}, 3, true);
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    assert(thrown);
}

int main() {
    std::cout << "=== test_nullable_semantics ===\n";
    test_valid8_decode();
    test_null8_decode();
    test_null64_decode_and_mask();
    test_invalid_byte_count_throws();
    std::cout << "All nullable synthetic tests passed!\n";
    return 0;
}
