/** @file @brief Host-only cache/compiler/identity contracts. Fake C modules do not claim GPU coverage. */
#include "core/module_cache.h"
#include "core/module_build.h"
#include "core/content_hash.h"
#include "core/code_symbols.h"
#include "core/tagged_numeric.h"
#include "core/inline_math.h"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <future>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>

using namespace db;
namespace fs=std::filesystem;

template<class F> void rejects(F f) {bool failed=false;try {f();}catch(const std::exception&){failed=true;}assert(failed);}
static void write(const fs::path& path,const std::string& text) {std::ofstream file(path);file<<text;file.close();assert(file);}
static std::string quote(const std::string& text) {
    std::string out="'";
    for(char c:text) out+=c=='\''?"'\\''":std::string(1,c);
    return out+"'";
}

/** @brief A real dlopen/dlsym fixture with independently compiled C ABI stubs, not fake GPU execution. */
class HostCompiler final : public ICompiler {
public:
    std::shared_ptr<ModuleArtifacts> artifacts=std::make_shared<ModuleArtifacts>(fs::path{},false);
    std::string toolchain="independent-c-abi-fixture";
    unsigned batches=0;
    bool wrong_identity=false,wrong_abi=false,wrong_schema=false,missing_symbol=false,fail=false;
    bool wide_schema=false,narrow_tagged=false,wrong_backend=false;
    const std::string& fingerprint() const override {return toolchain;}
    std::vector<CompiledModule> compile(const std::vector<KernelModuleSource>& modules,ModuleCounters& events,ModuleBuildTiming&) override {
        if(modules.empty()) return {};
        ++batches;
        if(fail) throw std::runtime_error("injected native compiler failure");
        const auto directory=artifacts->path()/std::to_string(batches);
        fs::create_directory(directory);
        std::vector<CompiledModule> result;
        for(const auto& module:modules) {
            const auto base=directory/module.identity;
            std::ostringstream source;
            source<<"#include \"core/module_abi_v2.h\"\nstatic KernelArgumentSchemaV2 slots[]={\n";
            if(module.schema.empty()) source<<"{1,1,1,0},\n";
            for(auto slot:module.schema) {
                if(wrong_schema) slot.access=DB_ACCESS_WRITE;
                if(wide_schema && slot.kind==DB_ARG_PARAMETER) slot.type=DB_TYPE_TAGGED_NUMERIC;
                if(narrow_tagged && slot.type==DB_TYPE_TAGGED_NUMERIC) slot.type=DB_TYPE_INT64;
                source<<'{'<<slot.kind<<','<<slot.type<<','<<slot.access<<','<<slot.flags<<"},\n";
            }
            source<<"};\nstatic KernelModuleDescriptorV2 descriptor={"<<(wrong_abi?1:2)
                  <<",sizeof(KernelModuleDescriptorV2),"<<(wrong_backend?DB_BACKEND_LEGACY_SYCL:DB_BACKEND_PCUDA)<<",0,slots,"<<module.schema.size()<<"};\n"
                  <<"const char* db_kernel_identity_v2(void){return \""<<(wrong_identity?"wrong":module.identity)<<"\";}\n"
                  <<"const KernelModuleDescriptorV2* db_kernel_descriptor_v2(void){return &descriptor;}\n";
            if(!missing_symbol) source<<"int32_t db_kernel_launch_v2(const KernelInvocationV2* p,KernelErrorV2* e){return DB_KERNEL_OK;}\n";
            write(base.string()+".c",source.str());
            const auto command="cc -std=c11 -shared -fPIC -I"+quote(std::string(DB_TEST_SOURCE_DIR)+"/include")+" "+quote(base.string()+".c")+" -o "+quote(base.string()+".so");
            assert(std::system(command.c_str())==0);
            recordModuleEvent(events.compiler);recordModuleEvent(events.linker);
            result.push_back({base.string()+".so",artifacts});
        }
        return result;
    }
};

static PlannedQuery plan(unsigned mask=0,const std::string& suffix="",unsigned threads=48) {
    PlannedQuery result;
    KernelContract contract{"fixture",{}};
    contract.arguments.push_back({"output"+suffix,{DB_ARG_BUFFER,DB_TYPE_UINT64,DB_ACCESS_WRITE,0},"buffer-binding"});
    std::string body="output"+suffix+"[threadIdx.x]=0;\n";
    for(unsigned i=0;i<4;++i) {
        const auto name="parameter"+std::to_string(i)+suffix;
        const auto type=(mask&(1U<<i))?DB_TYPE_FLOAT64:DB_TYPE_INT64;
        contract.arguments.push_back({name,{DB_ARG_PARAMETER,static_cast<uint32_t>(type),DB_ACCESS_READ,DB_ARG_NULLABLE},ParameterSlot{i}});
        body+="if("+name+"_valid) output"+suffix+"[threadIdx.x]+=static_cast<unsigned long long>("+name+");\n";
    }
    result.plan.kernels.push_back(contract);
    result.bodies.push_back({KernelBody::Kind::NdRange,"fixture","it",body,{},{},threads});
    return result;
}
static void hashesAndLexing() {
    assert(contentHash("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(contentHash("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(contentHash(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    const std::string source="alpha+alphabet;\"alpha\\\"\"; 'a'; /* alpha */ // alpha\nalpha";
    assert(codeIdentifiers(source)==std::set<std::string>({"alpha","alphabet"}));
    assert(replaceCodeIdentifiers(source,{{"alpha","x"}})=="x+alphabet;\"alpha\\\"\"; 'a'; /* alpha */ // alpha\nx");
    assert(codeIdentifiers("/* unfinished").empty());
    assert(codeIdentifiers("// end").empty());
    assert(codeIdentifiers("\"unfinished").empty());
    const auto a=plan(),b=plan(0,"_another_binding");
    const auto first=generatePcudacode(a.plan.kernels[0],a.bodies[0],"flags");
    const auto second=generatePcudacode(b.plan.kernels[0],b.bodies[0],"flags");
    assert(first.key==second.key && first.identity==second.identity);
    assert(first.source.find("pcudaParallelFor")!=std::string::npos && first.source.find("sycl::")==std::string::npos);
    assert(first.source.find("namespace db_module_"+first.identity)!=std::string::npos);
    assert(first.key!=generatePcudacode(a.plan.kernels[0],a.bodies[0],"changed flags").key);
    auto different=plan(0,"",31);
    assert(first.key!=generatePcudacode(different.plan.kernels[0],different.bodies[0],"flags").key);
    different.bodies[0].code="sycl::queue forbidden;";
    rejects([&]{generatePcudacode(different.plan.kernels[0],different.bodies[0],"flags");});
    for(const auto type:{DB_TYPE_INT32,DB_TYPE_INT64,DB_TYPE_FLOAT64,DB_TYPE_UINT8,DB_TYPE_NULL_ONLY}) {
        auto typed=plan();typed.plan.kernels[0].arguments[1].schema.type=type;
        assert(!generatePcudacode(typed.plan.kernels[0],typed.bodies[0],"flags").source.empty());
    }
    auto invalid=plan();invalid.plan.kernels[0].arguments[0].schema.type=999;
    rejects([&]{generatePcudacode(invalid.plan.kernels[0],invalid.bodies[0],"flags");});
}
static void boundedCache() {
    HostCompiler compiler;
    ProcessModuleCache cache;
    ModuleCounters counters;
    ModuleBuildTiming timing;
    std::shared_ptr<LoadedKernelModule> initial;
    for(unsigned mask=0;mask<16;++mask) {
        const auto loaded=cache.prepare(plan(mask),compiler,counters,timing);
        if(!mask) initial=loaded[0];
        assert(loaded.size()==1);
    }
    assert(cache.size()==9 && counters.compiler==9 && counters.linker==9 && counters.module_codegen==9 && counters.symbols==27);
    const auto before=counters;
    for(unsigned mask=0;mask<16;++mask) cache.prepare(plan(mask),compiler,counters,timing);
    assert(cache.prepare(plan(0,"_other_plan"),compiler,counters,timing)[0]==initial);
    assert(counters.compiler==before.compiler && counters.loads==before.loads && counters.symbols==before.symbols);
    auto changed=plan();changed.bodies[0].code+="output[threadIdx.x]+=1;";
    cache.prepare(changed,compiler,counters,timing);
    assert(counters.compiler==10);
    auto repeated=plan();
    repeated.plan.kernels.push_back(repeated.plan.kernels[0]);repeated.bodies.push_back(repeated.bodies[0]);
    assert(cache.prepare(repeated,compiler,counters,timing)[0]==initial);
}
/** @brief A native-ABI fixture whose second wave cannot finish before the client releases it. */
class GatedCompiler final : public ICompiler {
public:
    HostCompiler host;
    std::promise<void> entered,release;
    std::shared_future<void> released=release.get_future().share();
    unsigned calls=0;
    bool fail_second=false,bad_abi=false,bad_count=false;
    const std::string& fingerprint() const override {return host.fingerprint();}
    std::vector<CompiledModule> compile(const std::vector<KernelModuleSource>& modules,ModuleCounters& events,ModuleBuildTiming& timing) override {
        if (++calls==2) {
            entered.set_value();
            if (released.wait_for(std::chrono::seconds(15))!=std::future_status::ready) throw std::runtime_error("client failed to release the second wave");
            if (fail_second) throw std::runtime_error("late compiler failure");
            if (bad_count) return {};
            host.wrong_abi=bad_abi;
        }
        return host.compile(modules,events,timing);
    }
};
/** @brief Partial readiness is observable without publication; every late failure drains and remains retryable. */
static void partialReadiness() {
    auto query=plan();
    query.plan.kernels.push_back(query.plan.kernels[0]);
    query.bodies.push_back(query.bodies[0]);query.bodies[1].code+="output[threadIdx.x]+=1;";
    for (unsigned failure=0;failure<7;++failure) {
        unsigned drained=0,executed=0; // Completion observations must outlive the cache that retains them.
        GatedCompiler compiler;ProcessModuleCache cache;ModuleCounters counters;ModuleBuildTiming timing;
        compiler.fail_second=failure==1 || failure==5;compiler.bad_abi=failure==2 || failure==6;compiler.bad_count=failure==3;
        auto entered=compiler.entered.get_future();
        const auto run=[&](const KernelLookup& ready) {
            ++executed;
            const auto first=ready(0);
            assert(first && first==ready(0) && cache.size()==0);
            assert(entered.wait_for(std::chrono::seconds(5))==std::future_status::ready);
            first->retainCompletion(&drained,[&]{++drained;});
            compiler.release.set_value();
            if (failure==4) throw std::runtime_error("execution failed after the first submission");
            if (failure==5 || failure==6) return; // Unlaunched build/ABI failures must still prevent publication.
            const auto second=ready(1);
            assert(second && second!=first && cache.size()==0);
        };
        if (failure) {
            rejects([&]{cache.prepareOverlapped(query,compiler,counters,timing,1,run);});
            assert(cache.size()==0 && drained==1 && executed==1);
            HostCompiler retry;
            assert(cache.prepare(query,retry,counters,timing).size()==2 && cache.size()==2);
            if (failure==1) {
                LoadedKernels seventh;
                for (unsigned mask=1;mask<8;++mask) seventh=cache.prepare(plan(mask),retry,counters,timing);
                assert(cache.size()==9 && seventh[0]->descriptor()->arguments[1].type==DB_TYPE_FLOAT64);
            }
        } else {
            const auto loaded=cache.prepareOverlapped(query,compiler,counters,timing,1,run);
            assert(cache.size()==2 && loaded.size()==2 && counters.compiler==2 && counters.loads==2 && executed==1);
            assert(timing.waves.size()==2 && timing.waves[0].count==1 && timing.waves[1].first==1);
            assert(timing.waves[0].end_ns<=timing.waves[1].begin_ns && timing.waves[0].begin_ns<timing.waves[0].end_ns);
            const auto before=counters;
            cache.prepareOverlapped(query,compiler,counters,timing,1,[&](const KernelLookup& ready) {assert(ready(0)==loaded[0] && ready(1)==loaded[1]);});
            assert(counters.compiler==before.compiler && counters.loads==before.loads && counters.symbols==before.symbols);
        }
    }
    HostCompiler compiler;ProcessModuleCache cache;ModuleCounters counters;ModuleBuildTiming timing;
    for (size_t jobs:{size_t(0),size_t(5)}) rejects([&]{cache.prepareOverlapped(query,compiler,counters,timing,jobs,[](const KernelLookup&){});});
    rejects([&]{cache.prepareOverlapped(query,compiler,counters,timing,1,{});});
    assert(cache.size()==0 && counters.compiler==0 && counters.module_codegen==0);
    for (unsigned mask=0;mask<16;++mask)
        cache.prepareOverlapped(plan(mask),compiler,counters,timing,4,[](const KernelLookup& ready) {assert(ready(0));});
    assert(cache.size()==9 && counters.compiler==9); // The same eight typed + one tagged bound.
}
/** @brief Mixed hits/misses, duplicate indices and invalid lookups preserve the same cache identities. */
static void overlapReuse() {
    HostCompiler compiler;ProcessModuleCache cache;ModuleCounters counters;ModuleBuildTiming timing;
    auto query=plan();
    const auto initial=cache.prepare(query,compiler,counters,timing)[0];
    query.plan.kernels.push_back(query.plan.kernels[0]);query.bodies.push_back(query.bodies[0]);
    query.bodies[1].code+="output[threadIdx.x]+=9;";
    unsigned sources=0;
    const auto loaded=cache.prepareOverlapped(query,compiler,counters,timing,4,[&](const KernelLookup& ready) {
        assert(sources==1 && ready(0)==initial && ready(1)!=initial && cache.size()==1);
    },[&]{++sources;assert(counters.module_codegen==2 && counters.loads==1);});
    assert(loaded[0]==initial && cache.size()==2 && counters.compiler==2 && sources==1);
    ProcessModuleCache duplicate;
    query.bodies[1]=query.bodies[0];
    const auto before=counters;
    const auto aliases=duplicate.prepareOverlapped(query,compiler,counters,timing,4,[](const KernelLookup& ready){assert(ready(0)==ready(1));});
    assert(aliases[0]==aliases[1] && counters.compiler==before.compiler+1 && duplicate.size()==1);
    ProcessModuleCache rejected;
    rejects([&]{rejected.prepareOverlapped(query,compiler,counters,timing,4,[](const KernelLookup& ready){ready(2);});});
    assert(rejected.size()==0);
}
/** @brief A caller may join without consuming any future; finish must preserve failure and be idempotent. */
static void buildSequenceContract() {
    HostCompiler compiler;ModuleCounters counters;ModuleBuildTiming timing;
    const auto query=plan();
    const std::vector<KernelModuleSource> sources{generatePcudacode(query.plan.kernels[0],query.bodies[0],compiler.fingerprint())};
    for (size_t jobs:{size_t(0),size_t(5)}) rejects([&]{ModuleBuildSequence invalid(compiler,sources,jobs,counters,timing);});
    compiler.fail=true;
    ModuleBuildSequence build(compiler,sources,1,counters,timing);
    rejects([&]{build.finish();});
    const auto elapsed=timing.build_elapsed_ms;
    rejects([&]{build.finish();});
    rejects([&]{build.module(0);});
    rejects([&]{build.module(1);});
    assert(elapsed>0 && timing.build_elapsed_ms==elapsed && compiler.batches==1);
}
/** @brief The frontend boundary sees emitted sources before compiler/load work, including an all-hit plan. */
static void sourcesReadyBoundary() {
    HostCompiler compiler; ProcessModuleCache cache; ModuleCounters counters; ModuleBuildTiming timing;
    unsigned notifications = 0;
    cache.prepare(plan(), compiler, counters, timing, [&] {
        ++notifications;
        assert(counters.module_codegen == 1 && compiler.batches == 0 && counters.loads == 0);
        assert(cache.size() == 0);
    });
    assert(notifications == 1 && compiler.batches == 1);
    cache.prepare(plan(), compiler, counters, timing, [&] {
        ++notifications;
        assert(counters.module_codegen == 1 && compiler.batches == 1 && counters.loads == 1);
    });
    assert(notifications == 2 && compiler.batches == 1);
    auto invalid = plan(); invalid.bodies.clear();
    rejects([&] { cache.prepare(invalid, compiler, counters, timing, [&] { ++notifications; }); });
    assert(notifications == 2);
}
static void rejectAndRetry() {
    for(unsigned failure=0;failure<7;++failure) {
        HostCompiler compiler;ProcessModuleCache cache;ModuleCounters counters;ModuleBuildTiming timing;
        compiler.wrong_identity=failure==0;compiler.wrong_abi=failure==1;compiler.wrong_schema=failure==2;
        compiler.missing_symbol=failure==3;compiler.fail=failure==4;
        compiler.wide_schema=failure==5;compiler.wrong_backend=failure==6;
        rejects([&]{cache.prepare(plan(),compiler,counters,timing);});
        assert(cache.size()==0);
        compiler.wrong_identity=compiler.wrong_abi=compiler.wrong_schema=compiler.missing_symbol=compiler.fail=false;
        compiler.wide_schema=compiler.wrong_backend=false;
        const auto loaded=cache.prepare(plan(),compiler,counters,timing);
        assert(cache.size()==1 && loaded[0]->launch());
    }
    HostCompiler compiler;ProcessModuleCache cache;ModuleCounters counters;ModuleBuildTiming timing;
    auto invalid=plan();invalid.bodies.clear();rejects([&]{cache.prepare(invalid,compiler,counters,timing);});
    rejects([&]{LoadedKernelModule missing({"/nonexistent/pcuda-fixture.so",{}},"id",counters);});
    auto counter=UINT64_MAX;rejects([&]{recordModuleEvent(counter);});
    for(unsigned mask=0;mask<8;++mask) cache.prepare(plan(mask),compiler,counters,timing);
    compiler.narrow_tagged=true;rejects([&]{cache.prepare(plan(8),compiler,counters,timing);});
    assert(cache.size()==8);
    compiler.narrow_tagged=false;cache.prepare(plan(8),compiler,counters,timing);assert(cache.size()==9);
    unsigned drained=0;
    {
        HostCompiler owner;ProcessModuleCache lifetime;ModuleCounters events;ModuleBuildTiming times;
        auto loaded=lifetime.prepare(plan(),owner,events,times);
        loaded[0]->retainCompletion(&drained,[&]{++drained;});
        loaded[0]->retainCompletion(&drained,[&]{drained+=100;}); // one completion owner, not duplicate callbacks
    }
    assert(drained==1);
}
static void nativeCompiler() {
    ModuleArtifacts directory({},false);
    const auto root=directory.path()/"space and ' quote";fs::create_directory(root);
    const auto compiler=root/"compiler.sh",pch=root/"prebuilt.pch",manifest=root/"toolchain.manifest";
    write(compiler,"#!/bin/bash\nout=\nwhile (($#)); do if [[ $1 == -o ]]; then shift; out=$1; fi; shift; done\n[[ -n $out ]] || exit 9\nprintf fixture >\"$out\"\n");
    fs::permissions(compiler,fs::perms::owner_all);
    write(pch,"prebuilt fixture, not a GPU PCH");
    std::string text="DB-PCUDA-PCH-1\ncompiler\t"+compiler.string()+"\npch\t"+pch.string()+"\nheader\t"+pch.string()+"\n";
    for(const auto* flag:{"-O0","-fPIC","-std=c++17","--acpp-targets=generic","--acpp-pcuda"}) text+="compile_arg\t"+std::string(flag)+"\n";
    text+="file\t"+fileContentHash(compiler)+"\t"+compiler.string()+"\nfile\t"+fileContentHash(pch)+"\t"+pch.string()+"\n";
    text+="link_arg\t-fuse-ld=lld\ncompiler_env_unset\tDB_STAGE5_TEST_UNSET\ncompiler_env_set\tDB_STAGE5_TEST_SET\tvalue with ' quote\n";
    write(manifest,text);
    CrystalConfig config;config.pcuda_manifest=manifest.string();config.jit_artifact_dir=root.string();config.jit_jobs=2;
    ModuleCounters counters;ModuleBuildTiming timing;
    AdaptiveCppCompiler native(config);
    assert(fs::is_directory(native.artifactPath()));
    // Exercise the deployed lookup with a real executable path, not a mocked /proc value.
    const auto prefix=root/"installed";
    fs::create_directories(prefix/"bin");
    fs::create_directories(prefix/"lib/crystal/pcuda-jit");
    fs::copy_file(fs::read_symlink("/proc/self/exe"),prefix/"bin/checks");
    write(prefix/"lib/crystal/pcuda-jit/toolchain.manifest",text);
    const auto deployed="env -u CRYSTAL_PCUDA_MANIFEST "+quote((prefix/"bin/checks").string())+" --installed-pch "+quote(native.fingerprint());
    assert(std::system(deployed.c_str())==0);
    setenv("CRYSTAL_PCUDA_MANIFEST",manifest.c_str(),1);
    CrystalConfig overridden;overridden.pcuda_manifest="/nonexistent/overridden.manifest";
    assert(AdaptiveCppCompiler(overridden).fingerprint()==native.fingerprint());
    unsetenv("CRYSTAL_PCUDA_MANIFEST");
    const auto query=plan();
    auto module=generatePcudacode(query.plan.kernels[0],query.bodies[0],native.fingerprint());
    auto another=module;another.identity+="another";
    const auto compiled=native.compile({module,another},counters,timing);
    assert(compiled.size()==2 && fs::exists(compiled[0].path) && counters.compiler==2 && counters.linker==2);
    assert(timing.build_elapsed_ms>=0 && timing.compile_wall_ms>=0);
    assert(native.compile({},counters,timing).empty());
    for(const auto& invalid:std::vector<std::string>{"DB-PCUDA-PCH-1\nmalformed\n",text+"unknown\tx\n",text+"compiler_env_set\tmalformed\n",text+"file\tbad\n"}) {
        write(manifest,invalid);rejects([&]{AdaptiveCppCompiler bad(config);});
    }
    auto noflag=text;const auto flag=noflag.find("compile_arg\t-O0\n");noflag.erase(flag,16);
    write(manifest,noflag);rejects([&]{AdaptiveCppCompiler bad(config);});
    for(const auto* incompatible:{"-O3","-std=c++20","-ffast-math","-fno-validate-pch"}) {
        write(manifest,text+"compile_arg\t"+std::string(incompatible)+"\n");
        rejects([&]{AdaptiveCppCompiler bad(config);});
    }
    write(manifest,text);
    config.jit_jobs=0;rejects([&]{AdaptiveCppCompiler bad(config);});config.jit_jobs=4;
    const auto previous=fileContentHash(compiler);
    write(compiler,"#!/bin/bash\nexit 7\n");
    auto failed=text;failed.replace(failed.find(previous),previous.size(),fileContentHash(compiler));write(manifest,failed);
    AdaptiveCppCompiler failing(config);ModuleCounters failed_events;ModuleBuildTiming failed_timing;
    rejects([&]{failing.compile({module},failed_events,failed_timing);});
    assert(failed_events.compiler==1 && failed_events.linker==0);
    write(pch,"changed input");rejects([&]{AdaptiveCppCompiler bad(config);});
    write(manifest,"wrong format\n");rejects([&]{AdaptiveCppCompiler bad(config);});
    rejects([&]{fileContentHash(root/"absent");});
}
static void taggedArithmetic() {
    const TaggedNumeric large=INT64_C(9007199254740993),one=INT64_C(1),fraction=0.5;
    assert(static_cast<int64_t>(safe_add(large,one))==9007199254740994LL);
    assert(safe_neq(large,INT64_C(9007199254740992)));
    assert(static_cast<double>(safe_add(fraction,1))==1.5);
    assert(static_cast<int64_t>(safe_sub(large,one))==9007199254740992LL);
    assert(static_cast<int64_t>(safe_mul(TaggedNumeric(7),3))==21);
    assert(static_cast<int64_t>(safe_div(TaggedNumeric(21),3))==7);
    assert(safe_gt(large,one) && safe_lte(one,large) && safe_gte(large,large));
    KernelArgumentV1 argument{};argument.schema.type=DB_TYPE_FLOAT64;argument.valid=1;argument.value.f64=3.5;
    assert(static_cast<double>(taggedParameter(argument))==3.5);
    argument.valid=0;assert(static_cast<int64_t>(taggedParameter(argument))==0);
    const TaggedNumeric nan=std::numeric_limits<double>::quiet_NaN();
    assert(!(nan<=one) && !(nan>=one) && !(nan==one) && nan!=one);
}
#ifdef DB_ACCEPTANCE_COMBINED_TEST
int moduleChecks(int argc,char** argv) {
#else
int main(int argc,char** argv) {
#endif
    if(argc==3 && std::string(argv[1])=="--installed-pch") {
        assert(AdaptiveCppCompiler(CrystalConfig{}).fingerprint()==argv[2]);
        return 0;
    }
    hashesAndLexing();boundedCache();partialReadiness();overlapReuse();buildSequenceContract();sourcesReadyBoundary();rejectAndRetry();nativeCompiler();taggedArithmetic();
    std::cout<<"Host module identities, bounded cache, ABI rejection/retry and stock-compiler protocol passed\n";
    return 0;
}
