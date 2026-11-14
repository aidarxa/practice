#include "core/module_cache.h"
#include "core/module_build.h"
#include "core/module_contract_v2.h"
#include <algorithm>
#include <chrono>
#include <dlfcn.h>
#include <stdexcept>
#include <unordered_set>

namespace db {
namespace {
template<class Function> Function symbol(void* handle, const char* name, ModuleCounters& counters) {
    recordModuleEvent(counters.symbols);
    dlerror();
    auto* address = dlsym(handle,name);
    const auto* error = dlerror();
    if (error) throw std::runtime_error(std::string("dlsym ")+name+": "+error);
    return reinterpret_cast<Function>(address);
}
void closeModule(void* handle) noexcept { if (handle && dlclose(handle)) std::terminate(); }
bool hasParameters(const KernelContract& contract) {
    return std::any_of(contract.arguments.begin(),contract.arguments.end(),[](const auto& arg) {return arg.schema.kind==DB_ARG_PARAMETER;});
}
struct PendingModule {
    KernelModuleSource recipe;
    size_t kernel;
    bool tagged;
};
void validateLoadedSchema(const KernelModuleDescriptorV2& module, const std::vector<KernelArgumentSchemaV2>& expected) {
    if (module.argument_count != expected.size() || (module.argument_count && !module.arguments))
        throw std::runtime_error("Loaded PCUDA module argument count differs from its recipe");
    for (size_t i=0;i<expected.size();++i) {
        if (!abi2::validSchema(module.arguments[i]) || !compatibleArgumentSchema(expected[i],module.arguments[i]) ||
            !compatibleArgumentSchema(module.arguments[i],expected[i]))
            throw std::runtime_error("Loaded PCUDA module argument schema differs from its recipe");
    }
}
void validateLoadedHeader(const KernelModuleDescriptorV2* descriptor) {
    if (!descriptor || descriptor->abi_version!=DB_MODULE_ABI_V2_VERSION || descriptor->struct_size!=sizeof(KernelModuleDescriptorV2) ||
        descriptor->backend!=DB_BACKEND_PCUDA || descriptor->reserved)
        throw std::runtime_error("Incompatible loaded PCUDA module ABI/backend/layout");
}
} // namespace

LoadedKernelModule::LoadedKernelModule(CompiledModule binary, const std::string& identity, ModuleCounters& counters,
                                     const std::vector<KernelArgumentSchemaV2>& schema)
    : binary_(std::move(binary)) {
    recordModuleEvent(counters.loads);
    handle_ = dlopen(binary_.path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!handle_) throw std::runtime_error("dlopen failed: "+std::string(dlerror()));
    try {
        const auto actual = symbol<const char*(*)()>(handle_,"db_kernel_identity_v2",counters)();
        if (!actual || identity != actual) throw std::runtime_error("Loaded PCUDA module identity differs from its structural key");
        descriptor_ = symbol<DbKernelDescriptorV2>(handle_,"db_kernel_descriptor_v2",counters)();
        launch_ = symbol<DbKernelLaunchV2>(handle_,"db_kernel_launch_v2",counters);
        validateLoadedHeader(descriptor_);
        validateLoadedSchema(*descriptor_,schema);
    } catch (...) { closeModule(handle_); handle_=nullptr; throw; }
}
LoadedKernelModule::~LoadedKernelModule() noexcept {
    for (const auto& completion : completions_) completion.second();
    closeModule(handle_);
}
void LoadedKernelModule::retainCompletion(const void* identity, std::function<void()> drain) {
    completions_.emplace(identity,std::move(drain));
}

/** @brief A miss transaction owns reservations until every compiler output has passed load/ABI validation. */
struct ProcessModuleCache::Preparation {
    ProcessModuleCache& cache;
    const PlannedQuery& query;
    ICompiler& compiler;
    ModuleCounters& counters;
    ModuleBuildTiming& timing;
    std::vector<std::string> keys;
    std::vector<PendingModule> pending;
    std::unordered_set<std::string> requested;
    std::unordered_map<std::string,size_t> reserved;

    Preparation(ProcessModuleCache& owner,const PlannedQuery& plan,ICompiler& tool,ModuleCounters& events,ModuleBuildTiming& times)
        : cache(owner),query(plan),compiler(tool),counters(events),timing(times),reserved(owner.typed_) {}
    std::string family(const KernelModuleSource& recipe) const { return recipe.structure+compiler.fingerprint(); }
    void missing(KernelModuleSource recipe,size_t i,bool tagged) {
        if (!requested.insert(recipe.key).second) return;
        recordModuleEvent(counters.misses);
        if (!tagged && hasParameters(query.plan.kernels[i])) ++reserved[family(recipe)];
        const auto old=cache.identities_.find(recipe.identity);
        if (old!=cache.identities_.end() && old->second!=recipe.key) throw std::runtime_error("PCUDA module identity hash collision");
        recipe.source=generatePcudacode(query.plan.kernels[i],query.bodies[i],compiler.fingerprint(),tagged).source;
        recordModuleEvent(counters.module_codegen);
        pending.push_back({std::move(recipe),i,tagged});
    }
    void select(size_t i) {
        auto recipe=generatePcudacode(query.plan.kernels[i],query.bodies[i],compiler.fingerprint(),false,false);
        bool tagged=false;
        if (!cache.modules_.count(recipe.key) && hasParameters(query.plan.kernels[i]) && reserved[family(recipe)]>=8) {
            tagged=true;
            recipe=generatePcudacode(query.plan.kernels[i],query.bodies[i],compiler.fingerprint(),true,false);
        }
        keys.push_back(recipe.key);
        if (cache.modules_.count(recipe.key)) {recordModuleEvent(counters.hits); return;}
        missing(std::move(recipe),i,tagged);
    }
    LoadedKernels build() {
        std::vector<KernelModuleSource> sources;
        for (const auto& module:pending) sources.push_back(module.recipe);
        const auto binaries=compiler.compile(sources,counters,timing);
        if (binaries.size()!=pending.size()) throw std::runtime_error("Compiler returned an invalid PCUDA module count");
        const auto load_start=std::chrono::steady_clock::now();
        LoadedKernels loaded;
        for (size_t i=0;i<pending.size();++i)
            loaded.push_back(std::make_shared<LoadedKernelModule>(binaries[i],pending[i].recipe.identity,counters,pending[i].recipe.schema));
        timing.load_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_start).count();
        return loaded;
    }
    void publish(const LoadedKernels& loaded) {
        auto modules=cache.modules_;
        auto identities=cache.identities_;
        for (size_t i=0;i<pending.size();++i) {
            modules.emplace(pending[i].recipe.key,loaded[i]);
            identities.emplace(pending[i].recipe.identity,pending[i].recipe.key);
        }
        cache.modules_.swap(modules);
        cache.identities_.swap(identities);
        cache.typed_.swap(reserved);
    }
    /** @brief Retain each loaded candidate once; duplicate plan indices share the same staged owner. */
    std::shared_ptr<LoadedKernelModule> ready(size_t index,ModuleBuildSequence& build,LoadedKernels& staged) {
        const auto& key=keys.at(index);
        const auto cached=cache.modules_.find(key);
        if (cached!=cache.modules_.end()) return cached->second;
        const auto found=std::find_if(pending.begin(),pending.end(),[&](const auto& item){return item.recipe.key==key;});
        const auto slot=static_cast<size_t>(found-pending.begin());
        auto& module=staged.at(slot);
        if (!module) {
            const auto binary=build.module(slot); // No load timer includes waiting for S1 readiness.
            const auto start=std::chrono::steady_clock::now();
            module=std::make_shared<LoadedKernelModule>(binary,found->recipe.identity,counters,found->recipe.schema);
            timing.load_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        return module;
    }
    /** @brief Execution and cache commit are separate; even unlaunched branch modules must validate. */
    void overlap(size_t jobs,const std::function<void(const KernelLookup&)>& execute) {
        std::vector<KernelModuleSource> sources;
        for (const auto& module:pending) sources.push_back(module.recipe);
        ModuleBuildSequence build(compiler,std::move(sources),jobs,counters,timing);
        LoadedKernels staged(pending.size());
        execute([&](size_t index){return ready(index,build,staged);});
        build.finish();
        for (size_t index=0;index<keys.size();++index) ready(index,build,staged);
        publish(staged);
    }
    /** @brief Source selection is shared by synchronous and experimental paths. */
    void selectAll() {
        if (query.bodies.size()!=query.plan.kernels.size()) throw std::runtime_error("Physical kernel body/contract count mismatch");
        for (size_t i=0;i<query.bodies.size();++i) select(i);
    }
    LoadedKernels result() const {
        LoadedKernels loaded;
        for (const auto& key:keys) loaded.push_back(cache.modules_.at(key));
        return loaded;
    }
};

LoadedKernels ProcessModuleCache::prepare(const PlannedQuery& query,ICompiler& compiler,ModuleCounters& counters,ModuleBuildTiming& timing,
                                         const std::function<void()>& sources_ready) {
    Preparation preparation(*this,query,compiler,counters,timing);
    preparation.selectAll();
    if (sources_ready) sources_ready();
    if (!preparation.pending.empty()) preparation.publish(preparation.build());
    return preparation.result();
}
LoadedKernels ProcessModuleCache::prepareOverlapped(const PlannedQuery& query,ICompiler& compiler,ModuleCounters& counters,ModuleBuildTiming& timing,
        size_t jobs,const std::function<void(const KernelLookup&)>& execute,const std::function<void()>& sources_ready) {
    if (!jobs || jobs>4 || !execute) throw std::runtime_error("overlap requires an execution callback and one to four jobs");
    Preparation preparation(*this,query,compiler,counters,timing);
    preparation.selectAll();
    if (sources_ready) sources_ready();
    if (preparation.pending.empty()) {
        const auto result=preparation.result();
        execute([&](size_t index){return result.at(index);});
        return result;
    }
    preparation.overlap(jobs,execute);
    return preparation.result();
}
} // namespace db
