/** @file @brief End-to-end tiny SQL checks of PHT zero/negative keys and occupancy reuse. */
#include "core/execution.h"
#include <cassert>
#include <iostream>

using namespace db;
class PhtFixture {
public:
    std::shared_ptr<DeviceRuntime> runtime=std::make_shared<DeviceRuntime>();
    ExecutionContext context;
    std::shared_ptr<Catalog> catalog=std::make_shared<Catalog>();
    QueryEngine engine{catalog,std::make_unique<FileBasedQueryCache>(),std::make_unique<AdaptiveCppCompiler>(),std::make_unique<DynamicLibraryExecutor>()};
    PhtFixture() {
        context.runtime_=runtime;
        context.result_buffer_=std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime,0);
        context.result_validity_buffer_=std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime,0);
        TableMetadata supplier("SUPPLIER",{"s_suppkey","s_nation"},8,false);
        supplier.setColumnPrimaryKey("s_suppkey"); supplier.setColumnStats("s_suppkey",{-3,4,8});
        catalog->pushTableMetadata(supplier);
        TableMetadata fact("LINEORDER",{"lo_suppkey","lo_revenue"},6,true);
        fact.setColumnNullable("lo_suppkey"); catalog->pushTableMetadata(fact);
        input("s_suppkey",{-3,-2,-1,0,1,2,3,4}); input("s_nation",{11,12,13,14,15,16,17,18});
        input("lo_suppkey",{-3,-1,0,1,4,INT32_MIN}); input("lo_revenue",{10,20,30,40,50,60});
        auto bitmap=std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime,1);
        bitmap->copyFromHost(std::vector<uint64_t>{0x3f});
        context.base_columns_.at("d_lo_suppkey").validity=std::move(bitmap);
    }
    void input(const std::string& name,const std::vector<int>& values) {
        LoadedDeviceColumn column;
        column.values=std::make_unique<DynamicDeviceBuffer<int>>(runtime,values.size());
        column.values->copyFromHost(values);
        context.base_columns_.emplace("d_"+name,std::move(column));
    }
    std::vector<int64_t> query(const std::string& sql) {
        const auto* nonce=std::getenv("DB_STAGE2_FIXTURE_NONCE");
        std::string spacing=" ";
        for (const char* p=nonce?nonce:"0";*p;++p) spacing.append(static_cast<unsigned char>(*p)%17+1,' ');
        std::cout<<sql<<std::endl;
        engine.executeQuery(sql.substr(0,6)+spacing+sql.substr(6),&context);
        assert(context.result_is_columnar_ && context.result_column_count_==1);
        auto& column=context.result_column_storage_[0];
        std::vector<int64_t> result;
        if (column.type==LogicalType::Int64) column.i64->copyToHost(result,context.result_row_count_);
        else {
            std::vector<uint64_t> raw; column.u64->copyToHost(raw,context.result_row_count_);
            for (auto value:raw) result.push_back(static_cast<int64_t>(value));
        }
        return result;
    }
};
int main() try {
    PhtFixture fixture;
    const std::string from=" FROM lineorder,supplier WHERE lo_suppkey=s_suppkey";
    // Independent nested-loop expectations: the INT32_MIN probe has no match.
    assert(fixture.query("SELECT s_nation"+from+" ORDER BY s_nation")==std::vector<int64_t>({11,13,14,15,18}));
    assert(fixture.query("SELECT SUM(lo_revenue)"+from+" ORDER BY 1")==std::vector<int64_t>({150}));
    assert(fixture.query("SELECT COUNT(*)"+from+" AND s_suppkey<>0 ORDER BY 1")==std::vector<int64_t>({4}));
    assert(fixture.query("SELECT s_nation"+from+" AND s_suppkey<>0 ORDER BY s_nation")==std::vector<int64_t>({11,13,15,18}));
    assert(fixture.query("SELECT s_nation"+from+" ORDER BY s_nation")==std::vector<int64_t>({11,13,14,15,18}));
    // A NULL foreign key is not key zero, even when zero is present in the PHT.
    fixture.context.base_columns_.at("d_lo_suppkey").validity->copyFromHost(std::vector<uint64_t>{0x3b});
    assert(fixture.query("SELECT s_nation"+from+" ORDER BY s_nation")==std::vector<int64_t>({11,13,15,18}));
    fixture.context.config_.reuse_scratch_buffers=false;
    assert(fixture.query("SELECT s_nation"+from+" ORDER BY s_nation")==std::vector<int64_t>({11,13,15,18}));
    std::cout<<"7 SQL PHT zero/negative/absent/NULL/reuse results passed\n";
} catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
