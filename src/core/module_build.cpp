/** @file @brief Process-only S1 readiness with private worker counters and explicit ownership. */
#include "core/module_build.h"
#include <algorithm>
#include <chrono>
#include <future>
#include <thread>

namespace db {
namespace {
/** @brief Same steady-clock epoch as the isolated S2 observer. */
uint64_t steadyNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
/** @brief Validate the agreed resource bound before starting any asynchronous work. */
size_t waveSize(size_t jobs) {
    if (!jobs || jobs>4) throw std::runtime_error("overlap jobs must be between 1 and 4");
    return jobs;
}
/** @brief Merge counters transactionally, including the less common error-path events. */
void mergeCounters(ModuleCounters& target,const ModuleCounters& source) {
    const auto fields={&ModuleCounters::plan_codegen,&ModuleCounters::module_codegen,&ModuleCounters::compiler,
        &ModuleCounters::linker,&ModuleCounters::hits,&ModuleCounters::misses,&ModuleCounters::loads,&ModuleCounters::symbols};
    auto next=target;
    for (const auto field:fields) recordModuleEvent(next.*field,source.*field);
    target=next;
}
/** @brief The build interval is the direct worker span, not the sum of wave elapsed times. */
void mergeTiming(ModuleBuildTiming& target,const ModuleBuildTiming& source) {
    const auto fields={&ModuleBuildTiming::compile_wall_ms,&ModuleBuildTiming::compile_user_ms,&ModuleBuildTiming::compile_system_ms,
        &ModuleBuildTiming::link_wall_ms,&ModuleBuildTiming::link_user_ms,&ModuleBuildTiming::link_system_ms,
        &ModuleBuildTiming::build_elapsed_ms,&ModuleBuildTiming::load_ms};
    for (const auto field:fields) target.*field+=source.*field;
    target.waves.insert(target.waves.end(),source.waves.begin(),source.waves.end());
}
}
struct ModuleBuildSequence::Impl {
    ICompiler& compiler;
    const std::vector<KernelModuleSource> sources;
    const size_t jobs;
    ModuleCounters& counters;
    ModuleBuildTiming& timing;
    ModuleCounters events;
    ModuleBuildTiming observations;
    std::vector<std::promise<CompiledModule>> promises;
    std::vector<std::shared_future<CompiledModule>> ready;
    std::thread worker;
    std::exception_ptr failure;
    size_t published=0;
    bool merged=false;

    Impl(ICompiler& tool,std::vector<KernelModuleSource> input,size_t count,ModuleCounters& output,ModuleBuildTiming& times)
        : compiler(tool),sources(std::move(input)),jobs(waveSize(count)),counters(output),timing(times),promises(sources.size()) {
        for (auto& promise:promises) ready.push_back(promise.get_future().share());
    }
    /** @brief Publish a whole successful wave; compiler count mismatches cannot create dangling futures. */
    void wave(size_t offset) {
        const auto count=std::min(jobs,sources.size()-offset);
        const auto end=checkedSizeAdd(offset,count);
        const std::vector<KernelModuleSource> batch(sources.begin()+offset,sources.begin()+end);
        const auto begin=steadyNs();
        const auto binaries=compiler.compile(batch,events,observations);
        observations.waves.push_back({offset,count,begin,steadyNs()});
        if (binaries.size()!=count) throw std::runtime_error("Compiler returned an invalid PCUDA wave count");
        for (const auto& binary:binaries) {promises[published].set_value(binary);++published;}
    }
    /** @brief Resolve every future on both success and failure; no caller action is needed for progress. */
    void run() noexcept {
        const auto start=std::chrono::steady_clock::now();
        try {while (published<sources.size()) wave(published);}
        catch (...) {
            failure=std::current_exception();
            for (;published<promises.size();++published) promises[published].set_exception(failure);
        }
        observations.build_elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    /** @brief Joining precedes publication of private worker diagnostics and destruction of its inputs. */
    void join() {
        if (worker.joinable()) worker.join();
        if (merged) return;
        mergeCounters(counters,events);
        mergeTiming(timing,observations);
        merged=true;
    }
    ~Impl() noexcept {try {join();} catch (...) {std::terminate();}}
};
ModuleBuildSequence::ModuleBuildSequence(ICompiler& compiler,std::vector<KernelModuleSource> sources,size_t jobs,ModuleCounters& counters,ModuleBuildTiming& timing)
    : impl_(std::make_unique<Impl>(compiler,std::move(sources),jobs,counters,timing)) {
    impl_->worker=std::thread([state=impl_.get()] {state->run();});
}
ModuleBuildSequence::~ModuleBuildSequence() noexcept=default;
CompiledModule ModuleBuildSequence::module(size_t index) {return impl_->ready.at(index).get();}
void ModuleBuildSequence::finish() {impl_->join();if (impl_->failure) std::rethrow_exception(impl_->failure);}
}
