/** @file @brief Actual SQL PCUDA modules: values, type variants, cross-plan reuse and exact results. */
#include "core/execution.h"
#include "core/pcuda_plan_services.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <fstream>

using namespace db;
using Rows = std::vector<std::vector<std::optional<int64_t>>>;

class ModuleSqlFixture {
public:
    std::shared_ptr<Catalog> catalog = std::make_shared<Catalog>();
    std::shared_ptr<DeviceRuntime> runtime = std::make_shared<DeviceRuntime>();
    ExecutionContext context;
    QueryEngine engine;
    ModuleSqlFixture(bool overlap=false,std::unique_ptr<ICompiler> compiler=std::make_unique<AdaptiveCppCompiler>())
        : engine(catalog,std::make_unique<ProcessModuleCache>(),std::move(compiler),std::make_unique<DynamicLibraryExecutor>()) {
        context.config_=loadCrystalConfig();
        context.config_.jit_overlap_experiment=overlap;
        context.runtime_=runtime;
        context.result_buffer_=std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime,0);
        context.result_validity_buffer_=std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime,0);
        TableMetadata fact("LINEORDER",{"lo_quantity","lo_revenue","lo_suppkey"},9,true);
        fact.setColumnStats("lo_quantity",{0,8,9});
        catalog->pushTableMetadata(fact);
        TableMetadata dimension("SUPPLIER",{"s_suppkey","s_region"},4,false);
        dimension.setColumnStats("s_suppkey",{0,3,4}); dimension.setColumnPrimaryKey("s_suppkey");
        catalog->pushTableMetadata(dimension);
        input("lo_quantity",{0,1,2,3,4,5,6,7,8});
        input("lo_revenue",{10,20,30,40,50,60,70,80,90});
        input("lo_suppkey",{0,1,2,3,0,1,2,3,0});
        input("s_suppkey",{0,1,2,3}); input("s_region",{0,1,2,3});
    }
    void input(const std::string& name,const std::vector<int>& values) {
        LoadedDeviceColumn column;
        column.values=std::make_unique<DynamicDeviceBuffer<int>>(runtime,values.size());
        column.values->copyFromHost(values);
        context.base_columns_.emplace("d_"+name,std::move(column));
    }
    Rows query(const std::string& sql) {
        std::cout<<sql<<std::endl;
        engine.executeQuery(sql,&context);
        assert(context.timing_.sql_to_pcuda_ms > 0.0 && context.timing_.sql_to_pcuda_ms <= context.timing_.engine_ms);
        assert(context.result_is_columnar_);
        Rows rows(context.result_row_count_,std::vector<std::optional<int64_t>>(context.result_column_count_));
        for(size_t c=0;c<context.result_column_count_;++c) {
            auto& column=context.result_column_storage_[c];
            std::vector<uint64_t> valid;
            column.validity->copyToHost(valid,context.result_row_count_/64+(context.result_row_count_%64!=0));
            std::vector<int64_t> integers;
            std::vector<uint64_t> unsigneds;
            std::vector<double> floats;
            if(column.type==LogicalType::Int64) column.i64->copyToHost(integers,rows.size());
            else if(column.type==LogicalType::UInt64) column.u64->copyToHost(unsigneds,rows.size());
            else column.f64->copyToHost(floats,rows.size());
            for(size_t r=0;r<rows.size();++r) {
                if(context.result_columns_[c].nullable && !(valid[r/64]&(1ULL<<(r%64)))) continue;
                if(column.type==LogicalType::Int64) rows[r][c]=integers[r];
                else if(column.type==LogicalType::UInt64) rows[r][c]=static_cast<int64_t>(unsigneds[r]);
                else {
                    assert(floats[r]==static_cast<int64_t>(floats[r])); // only integral Float64 oracles in this fixture
                    rows[r][c]=static_cast<int64_t>(floats[r]);
                }
            }
        }
        return rows;
    }
    void noBuild() const {
        const auto& events=context.timing_.module_events;
        assert(events.plan_codegen==0 && events.module_codegen==0 && events.compiler==0 && events.linker==0 && events.loads==0 && events.symbols==0);
    }
};
/** @brief Fail a real native second wave, not the device launch route or the module ABI. */
class LateFailureCompiler final : public ICompiler {
public:
    AdaptiveCppCompiler native;
    unsigned waves=0;
    bool fail=true;
    const std::string& fingerprint() const override {return native.fingerprint();}
    std::vector<CompiledModule> compile(const std::vector<KernelModuleSource>& sources,ModuleCounters& counters,ModuleBuildTiming& timing) override {
        if (++waves==2 && fail) throw std::runtime_error("injected late S1 failure after the first ready wave");
        return native.compile(sources,counters,timing);
    }
};
/** @brief A failed partially submitted SQL cannot publish a ready plan; fresh native modules make retry correct. */
static void lateFailureRetry() {
    auto compiler=std::make_unique<LateFailureCompiler>();
    auto* control=compiler.get();
    ModuleSqlFixture fixture(true,std::move(compiler));
    fixture.context.config_.jit_jobs=1;
    const std::string sql="SELECT COUNT(*) FROM lineorder WHERE lo_quantity >= 0 ORDER BY 1";
    bool failed=false;
    try {fixture.query(sql);} catch (const std::runtime_error& error) {failed=std::string(error.what()).find("injected late S1")!=std::string::npos;}
    assert(failed && fixture.engine.moduleCounters().compiler==1);
    fixture.runtime->wait();
    control->fail=false;
    assert(fixture.query(sql)==Rows({{9}}));
    assert(fixture.context.timing_.module_events.compiler>=2);
    assert(fixture.query(sql)==Rows({{9}}));fixture.noBuild();
}

static void valueReuse(ModuleSqlFixture& fixture) {
    assert(fixture.query("SELECT lo_revenue + 9007199254740993 FROM lineorder WHERE lo_quantity >= 3 ORDER BY 1 LIMIT 2") == Rows({{9007199254741033LL},{9007199254741043LL}}));
    assert(fixture.query("SELECT lo_revenue + 9007199254740995 FROM lineorder WHERE lo_quantity >= 7 ORDER BY 1 LIMIT 1") == Rows({{9007199254741075LL}}));
    fixture.noBuild();
    assert(fixture.query("SELECT lo_revenue + 9007199254740995 FROM lineorder WHERE lo_quantity >= 100 ORDER BY 1 LIMIT 0").empty());
    fixture.noBuild();
    const auto grouped=[](int64_t a,int64_t b,int limit) {
        const auto expr="CASE WHEN lo_quantity < 4 THEN "+std::to_string(a)+" ELSE "+std::to_string(b)+" END";
        return "SELECT "+expr+",COUNT(*) FROM lineorder GROUP BY "+expr+" HAVING COUNT(*) > 0 ORDER BY 1 LIMIT "+std::to_string(limit);
    };
    assert(fixture.query(grouped(1,1000000000,9))==Rows({{1,4},{1000000000,5}}));
    assert(fixture.query(grouped(7,7,100))==Rows({{7,9}})); fixture.noBuild();
    assert(fixture.query(grouped(7,8,0)).empty()); fixture.noBuild();
    assert(fixture.query(grouped(7,8,9))==Rows({{7,4},{8,5}})); fixture.noBuild();
    const std::string nullable = "CASE WHEN lo_quantity < 4 THEN NULL ELSE NULL END";
    assert(fixture.query("SELECT "+nullable+",COUNT(*) FROM lineorder GROUP BY "+nullable+" HAVING COUNT(*) > 0 ORDER BY 1 LIMIT 9") == Rows({{std::nullopt,9}}));
    fixture.noBuild();
    const auto having=[](int increment,int threshold,int limit) {
        const auto aggregate="SUM(lo_revenue+"+std::to_string(increment)+")";
        return "SELECT lo_quantity,"+aggregate+" FROM lineorder GROUP BY lo_quantity HAVING "+aggregate+">"+std::to_string(threshold)+" ORDER BY 1 LIMIT "+std::to_string(limit);
    };
    assert(fixture.query(having(1,80,10))==Rows({{7,81},{8,91}}));
    assert(fixture.query(having(2,80,1))==Rows({{7,82}})); fixture.noBuild();
    fixture.context.config_.reuse_scratch_buffers=false;
    assert(fixture.query(having(2,80,1))==Rows({{7,82}})); fixture.noBuild();
    fixture.context.config_.reuse_scratch_buffers=true;
    assert(fixture.query("SELECT 9007199254740993 AS exact FROM lineorder LIMIT 1")==Rows({{9007199254740993LL}}));
    assert(fixture.query("SELECT NULL AS exact FROM lineorder LIMIT 1")==Rows({{std::nullopt}}));
    fixture.noBuild();
}
static void structuralReuse(ModuleSqlFixture& fixture) {
    const std::string from=" FROM lineorder,supplier WHERE lo_suppkey=s_suppkey AND s_region>0 ORDER BY 1";
    assert(fixture.query("SELECT COUNT(*)"+from)==Rows({{6}}));
    assert(fixture.query("SELECT SUM(lo_revenue)"+from)==Rows({{300}}));
    const auto& events=fixture.context.timing_.module_events;
    assert(events.hits>=1 && events.module_codegen>=1); // unchanged build module, changed fused probe/aggregate
    assert(events.module_codegen==events.compiler && events.compiler==events.linker && events.loads==events.compiler);
}
/** @brief A single build-predicate operator change builds exactly one module, not a whole-query bundle. */
static void singleKernelChange(ModuleSqlFixture& fixture) {
    const std::string base = "SELECT COUNT(*) FROM lineorder,supplier WHERE lo_suppkey=s_suppkey AND s_region";
    assert(fixture.query(base + ">0 ORDER BY 1") == Rows({{6}}));
    fixture.noBuild();
    assert(fixture.query(base + "<0 ORDER BY 1") == Rows({{0}}));
    const auto events = fixture.context.timing_.module_events;
    std::cout << "Single-kernel structural events: compiler=" << events.compiler << " linker=" << events.linker
              << " loads=" << events.loads << " hits=" << events.hits << '\n';
    assert(events.module_codegen == 1 && events.compiler == 1 && events.linker == 1 && events.loads == 1 && events.hits >= 1);
    assert(fixture.query(base + ">0 ORDER BY 1") == Rows({{6}}));
    fixture.noBuild();
}
static void boundedTypes(ModuleSqlFixture& fixture) {
    const auto sql=[](unsigned mask) {
        const std::string values[]={"0","9","0","100"};
        std::string numbers[4];
        for(unsigned i=0;i<4;++i) numbers[i]=values[i]+((mask&(1U<<i))?".0":"");
        return "SELECT COUNT(*) FROM lineorder WHERE lo_quantity >= "+numbers[0]+" AND lo_quantity < "+numbers[1]+
            " AND lo_revenue > "+numbers[2]+" AND lo_revenue < "+numbers[3]+" ORDER BY 1";
    };
    const auto before=fixture.engine.moduleCounters();
    for(unsigned mask=0;mask<16;++mask) assert(fixture.query(sql(mask))==Rows({{9}}));
    const auto built=fixture.engine.moduleCounters().compiler-before.compiler;
    assert(built==9); // exactly eight typed scans plus one tagged fallback; final materialization is already ready
    for(unsigned mask=0;mask<16;++mask) {assert(fixture.query(sql(mask))==Rows({{9}}));fixture.noBuild();}
    assert(fixture.query("SELECT COUNT(*) FROM lineorder WHERE lo_quantity >= 0 AND lo_quantity < 9 AND lo_revenue > 0 AND lo_revenue < NULL ORDER BY 1")==Rows({{0}}));
    fixture.noBuild();
}
static void pendingUnload() {
    const auto runtime=std::make_shared<DeviceRuntime>();
    DynamicDeviceBuffer<unsigned long long> output(runtime,1);
    auto compiler=std::make_unique<AdaptiveCppCompiler>();
    auto cache=std::make_unique<ProcessModuleCache>();
    PlannedQuery query;
    query.plan.kernels.push_back({"pending",{{"output",{DB_ARG_BUFFER,DB_TYPE_UINT64,DB_ACCESS_WRITE,0},"output"}}});
    query.bodies.push_back({KernelBody::Kind::NdRange,"pending","it","output[0]=123456789ULL;",{},{},1});
    ModuleCounters counters;ModuleBuildTiming timing;
    auto modules=cache->prepare(query,*compiler,counters,timing);
    modules[0]->retainCompletion(runtime.get(),[runtime]{runtime->waitNoThrow();});
    KernelArgumentV2 argument{{DB_ARG_BUFFER,DB_TYPE_UINT64,DB_ACCESS_WRITE,0},output.data(),1,{},1,0};
    KernelInvocationV2 call{DB_MODULE_ABI_V2_VERSION,sizeof(KernelInvocationV2),DB_BACKEND_PCUDA,0,runtime->stream(),&argument,1,{{1,1,1},{1,1,1},0}};
    KernelErrorV2 error{sizeof(KernelErrorV2),0,0,{}};
    assert(modules[0]->launch()(nullptr,&error)==DB_KERNEL_ARGUMENT_ERROR);
    call.abi_version=1;
    assert(modules[0]->launch()(&call,&error)==DB_KERNEL_ABI_ERROR);
    call.abi_version=DB_MODULE_ABI_V2_VERSION;
    call.backend=DB_BACKEND_LEGACY_SYCL;
    assert(modules[0]->launch()(&call,&error)==DB_KERNEL_ABI_ERROR);
    call.backend=DB_BACKEND_PCUDA;call.launch.block[0]=2;
    assert(modules[0]->launch()(&call,&error)==DB_KERNEL_LAUNCH_ERROR);
    call.launch.block[0]=1;call.stream=nullptr;
    assert(modules[0]->launch()(&call,&error)==DB_KERNEL_LAUNCH_ERROR);
    call.stream=runtime->stream();
    assert(modules[0]->launch()(&call,&error)==DB_KERNEL_OK);
    compiler.reset();cache.reset();modules.clear(); // the last module owner must wait before dlclose
    std::vector<unsigned long long> result;output.copyToHost(result,1);
    assert(result==std::vector<unsigned long long>{123456789ULL});
}
/** @brief Independent CPU prefix expectations cover partial blocks, empty/in-place and invalid Int32 counts. */
static void prefixServices() {
    const auto runtime = std::make_shared<DeviceRuntime>();
    PlanPrefixService service(runtime);
    assert(service.run(nullptr,nullptr,DB_TYPE_INT32,0) == 0);
    assert(service.run(nullptr,nullptr,DB_TYPE_UINT64,0) == 0);
    const auto rejects = [](const auto& action) {
        bool caught = false;
        try { action(); } catch (const std::exception&) { caught = true; }
        assert(caught);
    };
    std::vector<int> counts(65539),expected(counts.size());
    int sum = 0;
    for (size_t i = 0; i < counts.size(); ++i) { counts[i] = i % 4; expected[i] = sum; sum += counts[i]; }
    DynamicDeviceBuffer<int> data(runtime,counts.size());
    data.copyFromHost(counts);
    assert(service.run(data.data(),data.data(),DB_TYPE_INT32,counts.size()) == static_cast<uint64_t>(sum));
    std::vector<int> actual;
    data.copyToHost(actual,counts.size());
    assert(actual == expected);
    for (const auto& invalid : {std::vector<int>{-1},std::vector<int>{INT32_MAX,1}}) {
        data.copyFromHost(invalid);
        rejects([&] { service.run(data.data(),data.data(),DB_TYPE_INT32,invalid.size()); });
    }
    data.copyFromHost(std::vector<int>{INT32_MAX});
    assert(service.run(data.data(),data.data(),DB_TYPE_INT32,1) == INT32_MAX);
    data.copyToHost(actual,1); assert(actual == std::vector<int>{0});
    rejects([&] { service.run(nullptr,data.data(),DB_TYPE_INT32,1); });
    rejects([&] { service.run(data.data(),nullptr,DB_TYPE_INT32,1); });
    rejects([&] { service.run(nullptr,nullptr,DB_TYPE_FLOAT64,0); });
    DynamicDeviceBuffer<unsigned long long> wide(runtime,3);
    wide.copyFromHost(std::vector<unsigned long long>{9007199254740993ULL,0,7});
    assert(service.run(wide.data(),wide.data(),DB_TYPE_UINT64,3) == 9007199254741000ULL);
    std::vector<unsigned long long> result;
    wide.copyToHost(result,3);
    assert(result == std::vector<unsigned long long>({0,9007199254740993ULL,9007199254740993ULL}));
}
int main(int argc,char** argv) try {
    const bool overlap=argc==3 && std::string(argv[2])=="--overlap";
    if(argc==2 || overlap) {
        const auto root=std::filesystem::absolute(argv[1]);
        std::filesystem::create_directories(root);
        const auto config=root/"fixture.conf";
        std::ofstream file(config);file<<"jit_keep_artifacts=on\njit_artifact_dir="<<root.string()<<'\n';file.close();
        setenv("CRYSTAL_CONFIG",config.c_str(),1);
        setenv("ACPP_APPDB_DIR",(root/"device-cache").c_str(),1);
    }
    ModuleSqlFixture fixture(overlap);
    prefixServices();
    valueReuse(fixture);
    structuralReuse(fixture);
    singleKernelChange(fixture);
    boundedTypes(fixture);
    pendingUnload();
    if (overlap) lateFailureRetry();
    std::cout<<"PCUDA SQL module reuse, CASE equality, LIMIT zero, exact Int64, typed NULL and bounded fallback passed\n";
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
