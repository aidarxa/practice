#include "core/content_hash.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace db {
namespace {
constexpr uint32_t constants[] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
uint32_t rotate(uint32_t value, unsigned bits) { return (value >> bits) | (value << (32 - bits)); }
class Sha256 {
public:
    void append(std::string_view bytes) {
        if (bytes.size() > (UINT64_MAX / 8) - length_) throw std::overflow_error("SHA-256 input too large");
        length_ += bytes.size();
        for (const auto byte : bytes) push(static_cast<uint8_t>(byte));
    }
    std::string finish() {
        const auto bits = length_ * 8;
        push(0x80);
        while (used_ != 56) push(0);
        for (int shift = 56; shift >= 0; shift -= 8) push(static_cast<uint8_t>(bits >> shift));
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state_) out << std::setw(8) << word;
        return out.str();
    }
private:
    std::array<uint32_t, 8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t length_ = 0;
    void push(uint8_t byte) {
        block_[used_++] = byte;
        if (used_ == block_.size()) { compress(); used_ = 0; }
    }
    std::array<uint32_t, 64> schedule() const {
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16; ++i) {
            for (size_t j = 0; j < 4; ++j) words[i] = (words[i] << 8) | block_[i * 4 + j];
        }
        for (size_t i = 16; i < words.size(); ++i) {
            const auto a = words[i - 15], b = words[i - 2];
            const auto s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3);
            const auto s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        return words;
    }
    void compress() {
        const auto words = schedule();
        auto work = state_;
        for (size_t i = 0; i < words.size(); ++i) {
            const auto s1 = rotate(work[4], 6) ^ rotate(work[4], 11) ^ rotate(work[4], 25);
            const auto choice = (work[4] & work[5]) ^ (~work[4] & work[6]);
            const auto first = work[7] + s1 + choice + constants[i] + words[i];
            const auto s0 = rotate(work[0], 2) ^ rotate(work[0], 13) ^ rotate(work[0], 22);
            const auto majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
            for (size_t j = 7; j > 0; --j) work[j] = work[j - 1];
            work[4] += first;
            work[0] = first + s0 + majority;
        }
        for (size_t i = 0; i < state_.size(); ++i) state_[i] += work[i];
    }
};
} // namespace
std::string contentHash(std::string_view bytes) { Sha256 hash; hash.append(bytes); return hash.finish(); }
std::string fileContentHash(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read JIT input: " + path.string());
    Sha256 hash;
    std::array<char, 65536> buffer;
    while (file.read(buffer.data(), buffer.size()) || file.gcount())
        hash.append(std::string_view(buffer.data(), static_cast<size_t>(file.gcount())));
    if (!file.eof()) throw std::runtime_error("Cannot finish reading JIT input: " + path.string());
    return hash.finish();
}
} // namespace db
