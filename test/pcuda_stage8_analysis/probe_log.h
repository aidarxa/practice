#pragma once
/** @file @brief Research-only bounded host observations; never part of the engine ABI. */
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

namespace db { namespace s2probe {
/** @brief Absolute steady-clock interval; nested events must not be added as disjoint work. */
struct Event {
    std::string kind;
    std::uint64_t begin_ns=0, end_ns=0, compilation=0, bytes=0;
    int status=0;
    std::string name;
    std::vector<std::string> arguments;
};
/** @brief One process-owned bounded recorder; overflow remains visible in the output. */
class EventLog {
public:
    explicit EventLog(std::size_t limit=1000000) : limit_(limit) {}
    void add(Event event) noexcept;
    std::uint64_t nextCompilation();
    void write(std::ostream&) const;
private:
    const std::size_t limit_;
    mutable std::mutex mutex_;
    std::vector<Event> events_;
    std::uint64_t dropped_=0, compilation_=0;
};
std::uint64_t nowNs();
EventLog& log();
/** @brief TLS grouping restores its parent and does not mutate process-global environment. */
class CompilationScope {
public:
    CompilationScope();
    ~CompilationScope();
private:
    std::uint64_t previous_;
};
/** @brief A direct host interval; event serialization occurs after the measured call. */
class Span {
public:
    explicit Span(std::string kind, std::uint64_t bytes=0, std::string name={}, std::vector<std::string> args={});
    void finish(int status);
private:
    Event event_;
};
}} // namespace db::s2probe
