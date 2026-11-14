/** @file @brief Independent Text boundary/view/parameter/ABI behavior, active in Release. */
#include "storage/text_column.h"
#include "core/parameter_value.h"
#include "core/module_contract_v2.h"
#include "core/tagged_numeric.h"
#include "cli/text_format.h"
#include "cli/sql_input.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <climits>

namespace {
void require(bool ok) {
    static unsigned check = 0;
    ++check;
    if (!ok) throw std::runtime_error("Text contract mismatch at check " + std::to_string(check));
}
template<class F> void rejects(F operation) {
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    require(caught);
}
db::TextView view(const std::string& text) {
    return {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}
void values() {
    const std::string empty, a = "a", aa = "aa", z = "z", space = "a ", unicode = u8"Я🙂";
    require(db::compareText(view(a), view(aa)) < 0);
    require(db::cli::quoteText("NULL") == "\"NULL\"");
    require(db::cli::quoteText("") == "\"\"");
    require(db::cli::quoteText("a |\n\t\"\\") == "\"a |\\u000a\\u0009\\\"\\\\\"");
    require(db::cli::quoteText(unicode+" ") == "\""+unicode+" \"");
    require(!db::cli::hasSqlTerminator("SELECT 'a;\n\t"));
    require(db::cli::hasSqlTerminator("SELECT 'a;\n\t';"));
    require(!db::cli::hasSqlTerminator("SELECT 'it''s;'") );
    require(!db::cli::hasSqlTerminator("-- ;\n SELECT 1 /* ; */"));
    require(db::cli::hasSqlTerminator("SELECT \"a;\"; --comment"));
    require(db::compareText(view(z), view(aa)) > 0);
    require(db::compareText(view(a), view(space)) < 0);
    require(!db::equalText(view(a), view(space)));
    require(db::equalText(view(empty), {nullptr, 0}));
    const auto copy = unicode;
    require(db::equalText(view(unicode), view(copy)));
    require(db::hashText(view(unicode)) == db::hashText(view(copy)));
    require(db::ParameterValue::text(unicode).asText() == unicode);
    const auto null = db::ParameterValue::null(db::ParameterType::Text);
    require(null.isNull() && null.type() == db::ParameterType::Text);
    rejects([&] { null.asText(); });
    rejects([] { db::ParameterValue::text(std::string("a\0b",3)); });
    rejects([] { db::ParameterValue::text(std::string("\xc0\x80",2)); });
    rejects([] { db::commonParameterType(db::ParameterType::Text, db::ParameterType::Int64); });
    for (const auto& text : {empty, a, unicode, std::string(u8"é"), std::string(u8"é"),std::string("\xf4\x8f\xbf\xbf",4)}) db::validateText(text);
    for (const auto& text : {std::string("a\0b", 3), std::string("\xc0\x80", 2), std::string("\xe0\x80\x80", 3),
         std::string("\xed\xa0\x80", 3), std::string("\xf4\x90\x80\x80", 4), std::string("\xf0\x90", 2), std::string("\xe2x\x80", 3)})
        rejects([&] { db::validateText(text); });
    rejects([] { db::validateText(std::string_view(reinterpret_cast<const char*>(1),uint64_t(INT32_MAX)+1)); });
}
/** @brief All six comparisons use an independent unsigned-byte lexicographical oracle. */
void comparisons() {
    const std::vector<std::string> cases={"","A","a","aa","a ",u8"é",u8"é",u8"Я🙂",u8"\u0080",std::string(1,'\x7f')};
    for (const auto& a : cases) for (const auto& b : cases) {
        const bool less=std::lexicographical_compare(a.begin(),a.end(),b.begin(),b.end(),[](unsigned char a,unsigned char b){return a<b;});
        const auto left=view(a),right=view(b);
        require(db::safe_eq(left,right)==(a==b)); require(db::safe_neq(left,right)==(a!=b));
        require(db::safe_lt(left,right)==less); require(db::safe_lte(left,right)==(less || a==b));
        require(db::safe_gt(left,right)==(!less && a!=b)); require(db::safe_gte(left,right)==!less);
    }
    const uint8_t bytes[]={'a','b','c'};
    const int32_t narrow[]={0,1,1,3}; const uint64_t wide[]={0,1,1,3};
    for (size_t i=0;i<3;++i) {
        require(db::equalText(db::textAt(bytes,narrow,i),db::textAt(bytes,wide,i)));
        require(db::textAt(bytes,narrow,i).bytes==uint64_t(narrow[i+1]-narrow[i]));
    }
    require(db::textAt(nullptr,narrow,1).data==nullptr);
    require(db::textAt(nullptr,wide,1).data==nullptr);
}
void storage(const std::filesystem::path& path) {
    const std::string bytes = std::string("a \0JUNK!", 8) + "12345678" + std::string(8, '\xff') + std::string(8, '\0');
    { std::ofstream output(path, std::ios::binary); output.write(bytes.data(), bytes.size()); }
    const auto column = db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, 8}, {0xb});
    require(column.offsets == std::vector<std::uint64_t>({0,2,10,10,10}));
    require(std::string(column.bytes.begin(), column.bytes.end()) == "a 12345678");
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, 8}); });
    rejects([&] { db::loadTextColumn(path, 3, {db::PhysicalEncoding::FixedText, 8}); });
    rejects([&] { db::loadTextColumn(path, 1, {db::PhysicalEncoding::FixedText, 8}); });
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::Int32LE, 4}); });
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, 8}, {1,2}); });
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, 8}, {0,0}); });
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, 0}); });
    rejects([&] { db::loadTextColumn(path, 4, {db::PhysicalEncoding::FixedText, uint32_t(INT32_MAX)+1}); });
    rejects([&] { db::loadTextColumn(path, SIZE_MAX, {db::PhysicalEncoding::FixedText, 8}); });
}
void abi() {
    KernelArgumentSchemaV2 schema{DB_ARG_PARAMETER, DB_TYPE_TEXT, DB_ACCESS_READ, DB_ARG_NULLABLE};
    KernelArgumentV2 argument{schema, nullptr, 0, {}, 1, 0};
    KernelModuleDescriptorV2 descriptor{2, sizeof(descriptor), DB_BACKEND_PCUDA, 0, &schema, 1};
    KernelInvocationV2 call{2, sizeof(call), DB_BACKEND_PCUDA, 0, reinterpret_cast<void*>(8), &argument, 1, {{1,1,1},{1,1,1},0}};
    KernelErrorV2 error{sizeof(error),0,0,{}};
    const auto status = db::abi2::validateInvocation(descriptor, call, &error);
    if (status) throw std::runtime_error("Initial ABI v2 validation: " + std::to_string(status) + " " + error.message);
    require(status == 0);
    argument.value.text.bytes = 1;
    require(db::abi2::validateInvocation(descriptor, call, &error) == DB_KERNEL_ARGUMENT_ERROR);
    argument.value.text = {reinterpret_cast<const uint8_t*>(8), 1};
    require(db::abi2::validateInvocation(descriptor, call, &error) == 0);
    argument.valid = 0;
    require(db::abi2::validateInvocation(descriptor, call, &error) == DB_KERNEL_ARGUMENT_ERROR);
    argument.value = {};
    require(db::abi2::validateInvocation(descriptor, call, &error) == 0);
    call.abi_version = 1;
    require(db::abi2::validateInvocation(descriptor, call, &error) == DB_KERNEL_ABI_ERROR);
}
/** @brief V2 negative cases check fail-closed layout/schema/payload before any launch. */
void invalidArguments() {
    KernelArgumentSchemaV2 text{DB_ARG_PARAMETER,DB_TYPE_TEXT,DB_ACCESS_READ,DB_ARG_NULLABLE};
    KernelArgumentV2 argument{text,nullptr,0,{},1,0};
    auto expect=[&](int status){require(db::abi2::validateArgument(text,argument)==status);};
    argument.value.text={reinterpret_cast<const uint8_t*>(8),uint64_t(INT32_MAX)+1}; expect(DB_KERNEL_ARGUMENT_ERROR);
    argument.value.text.bytes=INT32_MAX; expect(DB_KERNEL_OK); // header validation does not dereference a borrowed range
    argument.value={}; argument.valid=2; expect(DB_KERNEL_ARGUMENT_ERROR);
    argument.valid=0; expect(DB_KERNEL_OK);
    argument.reserved=1; expect(DB_KERNEL_ARGUMENT_ERROR); argument.reserved=0;
    argument.schema.type=DB_TYPE_INT64; expect(DB_KERNEL_SCHEMA_ERROR);
    text.kind=DB_ARG_SCALAR; argument.schema=text; expect(DB_KERNEL_SCHEMA_ERROR);
    text.kind=DB_ARG_PARAMETER; text.access=DB_ACCESS_WRITE; argument.schema=text; expect(DB_KERNEL_SCHEMA_ERROR);
    text={DB_ARG_BUFFER,DB_TYPE_TEXT,DB_ACCESS_READ_WRITE,0}; argument={text,reinterpret_cast<void*>(8),1,{},1,0}; expect(DB_KERNEL_OK);
    argument.data=reinterpret_cast<void*>(9); expect(DB_KERNEL_ARGUMENT_ERROR);
    argument.data=nullptr; expect(DB_KERNEL_ARGUMENT_ERROR);
    argument.element_count=0; text.flags=DB_ARG_NULLABLE; argument.schema=text; expect(DB_KERNEL_OK);
    argument.data=reinterpret_cast<void*>(8); argument.element_count=SIZE_MAX/16+1; expect(DB_KERNEL_ARGUMENT_ERROR);
    KernelArgumentSchemaV2 number{DB_ARG_PARAMETER,DB_TYPE_INT64,DB_ACCESS_READ,DB_ARG_NULLABLE};
    argument={number,nullptr,0,{},1,0}; argument.value.i64=-9007199254740993LL;
    require(db::abi2::validateArgument(number,argument)==DB_KERNEL_OK);
    reinterpret_cast<uint8_t*>(&argument.value)[8]=1;
    require(db::abi2::validateArgument(number,argument)==DB_KERNEL_ARGUMENT_ERROR);
    require(!db::compatibleArgumentSchema({DB_ARG_PARAMETER,DB_TYPE_TAGGED_NUMERIC,DB_ACCESS_READ,DB_ARG_NULLABLE},
                                         {DB_ARG_PARAMETER,DB_TYPE_TEXT,DB_ACCESS_READ,DB_ARG_NULLABLE}));
}
/** @brief V2's larger scalar union must copy exact numeric members into the bounded fallback. */
void taggedParameters() {
    KernelArgumentV2 argument{{DB_ARG_PARAMETER,DB_TYPE_INT64,DB_ACCESS_READ,DB_ARG_NULLABLE},nullptr,0,{},1,0};
    argument.value.i64=-9007199254740993LL;
    require(static_cast<int64_t>(db::taggedParameter(argument))==-9007199254740993LL);
    argument.schema.type=DB_TYPE_FLOAT64; argument.value.f64=-0.5;
    require(static_cast<double>(db::taggedParameter(argument))==-0.5);
    argument.valid=0;
    require(static_cast<int64_t>(db::taggedParameter(argument))==0);
}
}
#ifdef TEXT_COMBINED_TEST
int textValueChecks(int argc, char** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 2) throw std::runtime_error("provide owned fixture file path");
        values(); comparisons(); storage(argv[1]); abi(); invalidArguments(); taggedParameters();
        std::cout << "Text value/storage/ABI contracts passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
