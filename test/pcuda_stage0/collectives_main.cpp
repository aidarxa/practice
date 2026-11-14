#include "collectives.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <vector>

namespace {
using namespace stage0;

std::size_t groupCount(std::size_t count, unsigned block) {
    return count / block + (count % block != 0);
}

struct Level {
    std::unique_ptr<Buffer<Word>> prefix;
    std::unique_ptr<Buffer<Word>> sums;
};

// Runtime-owned high-water buffers; no allocations or host reads between launches.
class ScanWorkspace {
public:
    ScanWorkspace(pcudaStream_t stream, std::size_t capacity, unsigned block)
        : stream_(stream), block_(block), capacity_(capacity) {
        if (!capacity || block < 2 || block > maxBlock)
            throw std::runtime_error("invalid scan workspace dimensions");
        for (auto count = capacity;;) {
            const auto groups = groupCount(count, block);
            levels_.push_back({std::make_unique<Buffer<Word>>(stream, count),
                               std::make_unique<Buffer<Word>>(stream, groups)});
            if (groups == 1) break;
            count = groups;
        }
        counts_.resize(levels_.size());
    }

    unsigned enqueue(const Word* input, std::size_t count, ScanLaunch scan, AddLaunch add) {
        if (count > capacity_) throw std::runtime_error("scan exceeds workspace capacity");
        if (!count) return 0;
        std::size_t used = 0;
        for (;;) {
            counts_[used++] = count;
            const auto groups = groupCount(count, block_);
            if (groups == 1) break;
            count = groups;
        }
        for (std::size_t index = 0; index < used; ++index) {
            check(scan(stream_, input, levels_[index].prefix->data, levels_[index].sums->data,
                       counts_[index], block_), "block scan");
            input = levels_[index].sums->data;
        }
        for (auto index = used - 1; index > 0; --index)
            check(add(stream_, levels_[index - 1].prefix->data, levels_[index].prefix->data,
                      counts_[index - 1], block_), "add offsets");
        return static_cast<unsigned>(used);
    }

    Word* output() const { return levels_[0].prefix->data; }
    Word* firstSums() const { return levels_[0].sums->data; }
private:
    pcudaStream_t stream_;
    unsigned block_;
    std::size_t capacity_;
    std::vector<Level> levels_;
    std::vector<std::size_t> counts_;
};

Word scanValue(std::size_t i, unsigned pattern) {
    if (pattern == 0) return 0;
    if (pattern == 1) return i % 7 == 0 ? 1 : 0;
    // Result exceeds 2^53 without overflowing uint64, even for the largest fixture.
    return i % 97 == 0 ? (Word{1} << 42) + 1 : static_cast<Word>(i % 11);
}

SignedWord reduceValue(std::size_t i, unsigned pattern) {
    if (pattern == 0) return 0;
    const auto small = static_cast<SignedWord>(i % 251) - 125;
    return pattern == 1 ? small : (i % 2 ? -(SignedWord{1} << 54) : (SignedWord{1} << 54)) + small;
}

template <class T>
void verifyReduction(const T* input, const T* result,
                     std::size_t count, unsigned block) {
    for (std::size_t group = 0; group < groupCount(count, block); ++group) {
        T sum = 0;
        T min = std::numeric_limits<T>::max();
        T max = std::numeric_limits<T>::lowest();
        const auto end = std::min(count, group * block + block);
        for (auto i = group * block; i < end; ++i) {
            sum += input[i];
            min = std::min(min, input[i]);
            max = std::max(max, input[i]);
        }
        if (result[3 * group] != sum || result[3 * group + 1] != min || result[3 * group + 2] != max)
            throw std::runtime_error("SUM/MIN/MAX mismatch in block " + std::to_string(group));
    }
}

void checkInvalidLaunches(pcudaStream_t stream, ScanLaunch scan, AddLaunch add, ReduceLaunch reduce) {
    if (scan(nullptr, nullptr, nullptr, nullptr, 0, 32) != pcudaErrorInvalidValue ||
        scan(stream, nullptr, nullptr, nullptr, 1, 32) != pcudaErrorInvalidValue ||
        scan(stream, nullptr, nullptr, nullptr, 0, 0) != pcudaErrorInvalidValue ||
        add(stream, nullptr, nullptr, 0, maxBlock + 1) != pcudaErrorInvalidValue ||
        reduce(stream, nullptr, nullptr, 1, 32) != pcudaErrorInvalidValue ||
        scan(stream, nullptr, nullptr, nullptr, std::numeric_limits<std::size_t>::max(), 32) != pcudaErrorInvalidValue)
        throw std::runtime_error("invalid launch accepted");
    bool overflowRejected = false;
    try {
        checkedBytes(std::numeric_limits<std::size_t>::max(), sizeof(Word));
    } catch (const std::overflow_error&) {
        overflowRejected = true;
    }
    if (!overflowRejected) throw std::runtime_error("size overflow accepted");
}

void verifyAtomics(pcudaStream_t stream, AtomicLaunch launch, bool atomic64) {
    Buffer<AtomicState> device(stream, 1);
    Buffer<AtomicState> host(stream, 1, true);
    constexpr unsigned count = 10003;
    *host.data = {};
    host.data->sum64 = Word{1} << 54;
    host.data->min64 = std::numeric_limits<Word>::max();
    check(pcudaMemcpyAsync(device.data, host.data, sizeof(AtomicState),
                          pcudaMemcpyHostToDevice, stream), "atomic H2D");
    check(launch(stream, device.data, count, 128, atomic64), "atomics");
    check(pcudaMemcpyAsync(host.data, device.data, sizeof(AtomicState),
                          pcudaMemcpyDeviceToHost, stream), "atomic D2H");
    check(pcudaStreamSynchronize(stream), "await atomic result");
    if (host.data->count32 != count || host.data->claims32 != 1 ||
        host.data->key32 != 77 || host.data->exchange32 != 91)
        throw std::runtime_error("32-bit atomic mismatch");
    if (atomic64 && (host.data->sum64 != (Word{1} << 54) + count ||
        host.data->min64 != (Word{1} << 54) || host.data->max64 != (Word{1} << 54) + count - 1 ||
        host.data->bits64 != std::numeric_limits<Word>::max() || host.data->cas64 != 99))
        throw std::runtime_error("64-bit atomic mismatch");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 7 || !*argv[6])
            throw std::runtime_error("usage: collectives scan.so add.so reduce.so reduce_unsigned.so atomics.so DEVICE");
        const auto prop = selectDevice(argv[6]);
        constexpr std::size_t capacity = 1000003;
        // Bound every sum before any allocation or kernel launch, including tree intermediates.
        if (capacity > std::numeric_limits<Word>::max() / ((Word{1} << 42) + 11) ||
            maxBlock > std::numeric_limits<SignedWord>::max() / ((SignedWord{1} << 54) + 125))
            throw std::overflow_error("fixture arithmetic bound exceeded");
        if (prop.maxThreadsPerBlock < static_cast<int>(maxBlock) ||
            prop.maxThreadsDim[0] < static_cast<int>(maxBlock) ||
            prop.sharedMemPerBlock < maxBlock * sizeof(Word) ||
            prop.maxGridSize[0] < static_cast<int>(groupCount(capacity, 32)))
            throw std::runtime_error("device collective limits insufficient");
        Stream stream;
        const auto loadStart = Clock::now();
        Module<ScanLaunch> scan(argv[1], stream.value, "launch_block_scan");
        Module<AddLaunch> add(argv[2], stream.value, "launch_add_offsets");
        Module<ReduceLaunch> reduce(argv[3], stream.value, "launch_block_reduce");
        Module<UnsignedReduceLaunch> reduceUnsigned(argv[4], stream.value, "launch_block_reduce_unsigned");
        Module<AtomicLaunch> atomics(argv[5], stream.value, "launch_atomics");
        const auto loadMs = elapsed(loadStart);
        checkInvalidLaunches(stream.value, scan.launch, add.launch, reduce.launch);
        Buffer<Word> hostInput(stream.value, capacity, true);
        Buffer<Word> hostOutput(stream.value, capacity, true);
        Buffer<Word> input(stream.value, capacity);
        Buffer<SignedWord> signedHost(stream.value, capacity, true);
        Buffer<SignedWord> signedInput(stream.value, capacity);
        const auto reduceCapacity = checkedBytes(groupCount(capacity, 32), std::size_t{3});
        Buffer<SignedWord> reduced(stream.value, reduceCapacity);
        Buffer<SignedWord> reducedHost(stream.value, reduceCapacity, true);
        Buffer<Word> reducedUnsigned(stream.value, reduceCapacity);
        Buffer<Word> reducedUnsignedHost(stream.value, reduceCapacity, true);
        Buffer<Word> firstSumsHost(stream.value, groupCount(capacity, 32), true);

        std::cout << std::setprecision(9) << "{\"load_ms\":" << loadMs << ",\"fixtures\":[";
        unsigned fixture = 0;
        for (unsigned block : {32u, 48u, 64u, 128u, 256u}) {
            ScanWorkspace workspace(stream.value, capacity, block);
            const auto square = static_cast<std::size_t>(block) * block;
            const auto cube = square * block; // bounded: block <= 256
            for (auto count : {capacity, capacity, std::size_t{0}, std::size_t{1},
                               std::size_t{block - 1}, std::size_t{block}, std::size_t{block + 1},
                               square - 1, square, square + 1, std::min(capacity, cube + 1)}) {
                for (unsigned pattern : {0u, 1u, 2u}) {
                    for (std::size_t i = 0; i < count; ++i) {
                        hostInput.data[i] = scanValue(i, pattern);
                        signedHost.data[i] = reduceValue(i, pattern);
                    }
                    const auto bytes = checkedBytes(count, sizeof(Word));
                    // Poison outputs to catch stale buffers and writes on empty input.
                    check(pcudaMemsetAsync(workspace.output(), 0xA5,
                                          checkedBytes(capacity, sizeof(Word)), stream.value), "poison scan");
                    check(pcudaMemsetAsync(reduced.data, 0xA5,
                                          checkedBytes(reduceCapacity, sizeof(SignedWord)), stream.value), "poison reduction");
                    check(pcudaMemsetAsync(reducedUnsigned.data, 0xA5,
                                          checkedBytes(reduceCapacity, sizeof(Word)), stream.value), "poison unsigned reduction");
                    if (count) {
                        check(pcudaMemcpyAsync(input.data, hostInput.data, bytes,
                                               pcudaMemcpyHostToDevice, stream.value), "scan H2D");
                        check(pcudaMemcpyAsync(signedInput.data, signedHost.data, bytes,
                                               pcudaMemcpyHostToDevice, stream.value), "reduce H2D");
                    }
                    // Isolate device-resident execution from fixture copies for timing.
                    check(pcudaStreamSynchronize(stream.value), "await fixture upload");
                    const auto start = Clock::now();
                    const auto levels = workspace.enqueue(input.data, count, scan.launch, add.launch);
                    check(reduce.launch(stream.value, signedInput.data, reduced.data, count, block), "reduce");
                    check(reduceUnsigned.launch(stream.value, input.data, reducedUnsigned.data, count, block), "unsigned reduce");
                    check(pcudaStreamSynchronize(stream.value), "await collectives");
                    const auto executionMs = elapsed(start);
                    // Also fetch a guard element beyond the valid output (if available).
                    const auto outputCount = count < capacity ? count + 1 : count;
                    const auto reducedCount = groupCount(count, block) * 3 + 1;
                    check(pcudaMemcpyAsync(hostOutput.data, workspace.output(),
                                           checkedBytes(outputCount, sizeof(Word)),
                                           pcudaMemcpyDeviceToHost, stream.value), "scan D2H");
                    check(pcudaMemcpyAsync(reducedHost.data, reduced.data,
                                           checkedBytes(std::min(reducedCount, reduceCapacity), sizeof(SignedWord)),
                                           pcudaMemcpyDeviceToHost, stream.value), "reduce D2H");
                    check(pcudaMemcpyAsync(reducedUnsignedHost.data, reducedUnsigned.data,
                                           checkedBytes(std::min(reducedCount, reduceCapacity), sizeof(Word)),
                                           pcudaMemcpyDeviceToHost, stream.value), "unsigned reduce D2H");
                    if (count)
                        check(pcudaMemcpyAsync(firstSumsHost.data, workspace.firstSums(),
                                               checkedBytes(groupCount(count, block), sizeof(Word)),
                                               pcudaMemcpyDeviceToHost, stream.value), "block sums D2H");
                    check(pcudaStreamSynchronize(stream.value), "await verification copies");
                    Word total = 0;
                    for (std::size_t i = 0; i < count; ++i) {
                        if (hostOutput.data[i] != total)
                            throw std::runtime_error("scan mismatch at " + std::to_string(i));
                        if (hostInput.data[i] > std::numeric_limits<Word>::max() - total)
                            throw std::overflow_error("fixture scan sum overflow");
                        total += hostInput.data[i];
                    }
                    verifyReduction(signedHost.data, reducedHost.data, count, block);
                    verifyReduction(hostInput.data, reducedUnsignedHost.data, count, block);
                    for (std::size_t group = 0; group < groupCount(count, block); ++group) {
                        Word sum = 0;
                        for (auto i = group * block; i < std::min(count, group * block + block); ++i)
                            sum += hostInput.data[i];
                        if (firstSumsHost.data[group] != sum)
                            throw std::runtime_error("block scan total mismatch");
                    }
                    constexpr Word poison = 0xA5A5A5A5A5A5A5A5ULL;
                    if ((count < capacity && hostOutput.data[count] != poison) ||
                        (reducedCount <= reduceCapacity && (static_cast<Word>(reducedHost.data[reducedCount - 1]) != poison ||
                         reducedUnsignedHost.data[reducedCount - 1] != poison)))
                        throw std::runtime_error("collective output guard overwritten");
                    if (fixture++) std::cout << ',';
                    std::cout << "{\"count\":" << count << ",\"block\":" << block
                              << ",\"pattern\":" << pattern << ",\"levels\":" << levels
                              << ",\"execute_ms\":" << executionMs << '}';
                }
            }
        }
        verifyAtomics(stream.value, atomics.launch, false);
        if (prop.pcudaHasAtomic64) verifyAtomics(stream.value, atomics.launch, true);
        std::cout << "],\"atomic32_verified\":true,\"atomic64_verified\":"
                  << (prop.pcudaHasAtomic64 ? "true" : "false") << ",\"verified\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
