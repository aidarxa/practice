/** @file @brief Real SQL/PCUDA fixture, emitting typed rows for an independent SQLite oracle. */
#include "core/execution.h"
#include "core/text_fetch.h"
#include <iostream>
#include <iomanip>

using namespace db;
namespace {
class Fixture {
public:
    std::shared_ptr<DeviceRuntime> runtime = std::make_shared<DeviceRuntime>();
    std::shared_ptr<Catalog> catalog = std::make_shared<Catalog>();
    ExecutionContext context;
    QueryEngine engine{catalog, std::make_unique<ProcessModuleCache>(), std::make_unique<AdaptiveCppCompiler>(), std::make_unique<DynamicLibraryExecutor>()};
    TextFetch fetch{runtime};
    Fixture() {
        context.runtime_ = runtime;
        context.config_ = loadCrystalConfig();
        context.result_buffer_ = std::make_unique<DynamicDeviceBuffer<unsigned long long>>(runtime, 0);
        context.result_validity_buffer_ = std::make_unique<DynamicDeviceBuffer<uint64_t>>(runtime, 0);
        TableMetadata customers("CUSTOMER", {"c_custkey", "c_name", "c_city"}, 8, false);
        customers.setColumnStorage("c_name", {PhysicalEncoding::FixedText,25}); customers.setColumnNullable("c_name");
        catalog->pushTableMetadata(customers);
        TableMetadata suppliers("SUPPLIER", {"s_suppkey", "s_name", "s_city"}, 6, false);
        suppliers.setColumnStorage("s_name", {PhysicalEncoding::FixedText,25}); suppliers.setColumnNullable("s_name");
        catalog->pushTableMetadata(suppliers);
        TableMetadata part("PART", {"p_partkey","p_name"},2,false);
        // The declared upper bound requires 64-bit offsets without allocating a multi-GiB payload.
        part.setColumnStorage("p_name",{PhysicalEncoding::FixedText,INT32_MAX}); catalog->pushTableMetadata(part);
        numeric("p_partkey",{1,2}); text("p_name",{"wide",u8"🙂"},3,true);
        numeric("c_custkey", {1,2,3,4,5,6,7,8}); numeric("c_city", {1,1,2,2,3,3,4,4});
        numeric("s_suppkey", {3,4,1,2,5,6});
        numeric("s_city", {1,2,3,4,5,6});
        text("c_name", {"","","Alice","Alice","a ","a",u8"Я","1234567890123456789012345"}, 0xfd);
        text("s_name", {"Alice","Alice","","","a ","Z"}, 0x37);
        context.plan_scratch_buffers_.emplace("retired_fixture_buffer",std::make_unique<DynamicDeviceBuffer<uint8_t>>(runtime,32));
    }
    void numeric(const std::string& name, const std::vector<int>& values) {
        LoadedDeviceColumn column;
        column.values = std::make_unique<DynamicDeviceBuffer<int>>(runtime, values.size());
        column.values->copyFromHost(values);
        context.base_columns_.emplace("d_"+name, std::move(column));
    }
    void text(const std::string& name, const std::vector<std::string>& values, uint64_t valid, bool wide=false) {
        HostTextColumn column;
        column.offsets.push_back(0);
        for (const auto& value : values) {
            validateText(value);
            column.bytes.insert(column.bytes.end(), value.begin(), value.end());
            column.offsets.push_back(column.bytes.size());
        }
        context.text_columns_.emplace(name, uploadTextColumn(runtime, column, wide, {valid}));
    }
    void query(const std::string& sql) {
        engine.executeQuery(sql, &context);
        if (context.plan_scratch_buffers_.count("retired_fixture_buffer"))
            throw std::runtime_error("Retired physical plans must not pin unrelated scratch indefinitely");
        if (context.result_column_storage_.size()!=context.result_column_count_)
            throw std::runtime_error("Retired result columns must not pin unrelated buffers");
        for (const auto& column : context.result_column_storage_)
            if (bool(column.text)+bool(column.i64)+bool(column.u64)+bool(column.f64)!=1)
                throw std::runtime_error("One result column must own exactly one logical payload type");
        if (!context.result_is_columnar_) throw std::runtime_error("Text fixture expects typed columnar output");
        const auto count = context.result_row_count_;
        std::vector<std::vector<std::string>> rows(count, std::vector<std::string>(context.result_column_count_));
        for (size_t col=0;col<context.result_column_count_;++col) {
            auto& storage = context.result_column_storage_[col];
            std::vector<uint64_t> validity;
            storage.validity->copyToHost(validity, count/64+(count%64!=0));
            std::vector<std::string> text;
            std::vector<int64_t> integers;
            std::vector<uint64_t> unsigneds;
            std::vector<double> fractions;
            if (storage.type==LogicalType::Text) text = fetch.read(storage.text->data(), storage.validity->data(), count, 0, count);
            else if (storage.type==LogicalType::Int64) storage.i64->copyToHost(integers,count);
            else if (storage.type==LogicalType::Float64) storage.f64->copyToHost(fractions,count);
            else storage.u64->copyToHost(unsigneds,count);
            for (size_t row=0;row<count;++row) {
                if (!((validity[row/64]>>(row%64))&1U)) { rows[row][col]="N"; continue; }
                std::ostringstream cell;
                if (storage.type==LogicalType::Text) {
                    cell << 'T';
                    for (const unsigned char byte : text[row]) cell << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
                } else if (storage.type==LogicalType::Int64) cell << 'I' << integers[row];
                else if (storage.type==LogicalType::Float64) cell << 'D' << std::setprecision(17) << fractions[row];
                else cell << 'U' << unsigneds[row];
                rows[row][col]=cell.str();
            }
        }
        std::cout << "__TEXT_ROWS__ " << count << ' ' << context.result_column_count_ << '\n';
        for (const auto& row : rows) {
            for (size_t i=0;i<row.size();++i) std::cout << (i?"\t":"") << row[i];
            std::cout << '\n';
        }
        const auto& events = context.timing_.module_events;
        std::cout << "__TEXT_EVENTS__ " << events.plan_codegen << ' ' << events.module_codegen << ' ' << events.compiler << ' '
                  << events.linker << ' ' << events.loads << ' ' << events.symbols << '\n';
    }
};
}
int main() {
    try {
        Fixture fixture;
        std::string sql;
        while (std::getline(std::cin,sql)) {
            if (sql.empty()) continue;
            std::cout << "__TEXT_BEGIN__ " << sql << std::endl;
            try { fixture.query(sql); }
            catch (const std::exception& error) { std::cout << "__TEXT_ERROR__ " << error.what() << std::endl; }
            std::cout << "__TEXT_END__" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
