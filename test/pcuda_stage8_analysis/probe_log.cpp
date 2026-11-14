/** @file @brief CPU recorder and JSON export for isolated S2 experiments. */
#include "probe_log.h"
#include "cli/text_format.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <ostream>
#include <unistd.h>

namespace db { namespace s2probe {
namespace {
thread_local std::uint64_t current_compilation=0;
void writeArguments(std::ostream& out, const std::vector<std::string>& arguments) {
    const char* separator="";
    for (const auto& argument:arguments) {out << separator << cli::quoteText(argument);separator=",";}
}
void writeEvent(std::ostream& out, const Event& event) {
    out << "{\"kind\":" << cli::quoteText(event.kind) << ",\"begin_ns\":" << event.begin_ns
        << ",\"end_ns\":" << event.end_ns << ",\"compilation\":" << event.compilation
        << ",\"bytes\":" << event.bytes << ",\"status\":" << event.status
        << ",\"name\":" << cli::quoteText(event.name) << ",\"arguments\":[";
    writeArguments(out,event.arguments);
    out << "]}";
}
void dumpAtExit() {
    const char* path=std::getenv("CRYSTAL_S2_EVENTS");
    if (!path) return;
    std::ofstream out(path);
    log().write(out);
    out.close();
    if (!out) {std::fputs("S2 event export failed\n",stderr);std::_Exit(125);}
}
void increment(std::uint64_t& value) {
    if (value==std::numeric_limits<std::uint64_t>::max()) std::abort();
    ++value;
}
}
void EventLog::add(Event event) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size()>=limit_) {increment(dropped_);return;}
    try {events_.push_back(std::move(event));}
    catch (...) {increment(dropped_);}
}
std::uint64_t EventLog::nextCompilation() {
    std::lock_guard<std::mutex> lock(mutex_);
    increment(compilation_);
    return compilation_;
}
void EventLog::write(std::ostream& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out << "{\"clock\":\"steady_ns\",\"pid\":" << getpid() << ",\"event_count\":" << events_.size()
        << ",\"dropped\":" << dropped_ << ",\"compilations\":" << compilation_ << ",\"events\":[";
    const char* separator="";
    for (const auto& event:events_) {out << separator;writeEvent(out,event);separator=",";}
    out << "]}\n";
}
std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
EventLog& log() {
    static EventLog recorder;
    static const auto registered=std::atexit(dumpAtExit);
    if (registered) std::abort();
    return recorder;
}
CompilationScope::CompilationScope() : previous_(current_compilation) {current_compilation=log().nextCompilation();}
CompilationScope::~CompilationScope() {current_compilation=previous_;}
Span::Span(std::string kind,std::uint64_t bytes,std::string name,std::vector<std::string> args)
    : event_{std::move(kind),0,0,current_compilation,bytes,0,std::move(name),std::move(args)} {
    (void)log();
    event_.begin_ns=nowNs();
}
void Span::finish(int status) {
    event_.end_ns=nowNs();
    event_.status=status;
    log().add(std::move(event_));
}
}} // namespace db::s2probe
