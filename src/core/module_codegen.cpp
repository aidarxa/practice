#include "core/module_codegen.h"
#include "core/content_hash.h"
#include "core/code_symbols.h"
#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>

namespace db {
namespace {
std::string storageType(uint32_t type) {
    if (type == DB_TYPE_INT32) return "int";
    if (type == DB_TYPE_UINT64) return "unsigned long long";
    if (type == DB_TYPE_INT64) return "std::int64_t";
    if (type == DB_TYPE_FLOAT64) return "double";
    if (type == DB_TYPE_UINT8) return "std::uint8_t";
    if (type == DB_TYPE_NULL_ONLY) return "std::int64_t";
    if (type == DB_TYPE_TEXT) return "db::TextView";
    throw std::runtime_error("Unsupported module storage type");
}
void token(std::ostream& out, const std::string& text) { out << text.size() << ':' << text; }
std::map<std::string, std::string> argumentNames(const KernelContract& contract) {
    std::map<std::string, std::string> result;
    for (size_t i = 0; i < contract.arguments.size(); ++i) {
        const auto& arg = contract.arguments[i];
        const auto local = "db_arg_" + std::to_string(i);
        result.emplace(arg.name, local);
        if (arg.schema.kind == DB_ARG_PARAMETER) result.emplace(arg.name + "_valid", local + "_valid");
    }
    return result;
}
std::string canonical(const KernelContract& contract, const KernelBody& body, bool schema, bool tagged) {
    std::ostringstream out;
    token(out, "pcuda-module-v2");
    out << static_cast<unsigned>(body.kind) << ',' << body.threads << ','
        << contract.requires_atomic64 << ',' << contract.static_shared_bytes << ';';
    for (const auto& local : body.locals) {
        token(out, local.name); out << local.type << ',' << local.elements << ';';
    }
    for (const auto& arg : contract.arguments) {
        const auto parameter = arg.schema.kind == DB_ARG_PARAMETER && arg.schema.type != DB_TYPE_TEXT;
        out << arg.schema.kind << ',' << (parameter && (!schema || tagged) ? DB_TYPE_TAGGED_NUMERIC : arg.schema.type)
            << ',' << arg.schema.access << ',' << arg.schema.flags << ';';
        token(out, arg.legacy_cpp_type);
    }
    token(out, replaceCodeIdentifiers(body.code, argumentNames(contract)));
    return out.str();
}
void emitSchema(std::ostream& out, const KernelContract& contract, bool tagged) {
    out << "static const KernelArgumentSchemaV2 schema[] = {\n";
    if (contract.arguments.empty()) out << "{DB_ARG_SCALAR,DB_TYPE_UINT64,DB_ACCESS_READ,0},\n";
    for (const auto& arg : contract.arguments) {
        const auto type = tagged && arg.schema.kind == DB_ARG_PARAMETER && arg.schema.type != DB_TYPE_TEXT ? DB_TYPE_TAGGED_NUMERIC : arg.schema.type;
        out << '{' << arg.schema.kind << ',' << type << ',' << arg.schema.access << ',' << arg.schema.flags << "},\n";
    }
    out << "};\nstatic const KernelModuleDescriptorV2 descriptor = {2,sizeof(KernelModuleDescriptorV2),DB_BACKEND_PCUDA,0,schema,"
        << contract.arguments.size() << "};\n";
}
void emitBindings(std::ostream& out, const KernelContract& contract, bool tagged) {
    for (size_t i = 0; i < contract.arguments.size(); ++i) {
        const auto& arg = contract.arguments[i];
        const auto& name = arg.name;
        const auto access = "call->arguments[" + std::to_string(i) + "]";
        const auto type = arg.legacy_cpp_type.empty() ? storageType(arg.schema.type) : arg.legacy_cpp_type;
        if (arg.schema.kind == DB_ARG_BUFFER) out << type << "* " << name << " = static_cast<" << type << "*>(" << access << ".data);\n";
        else if (arg.schema.kind == DB_ARG_PARAMETER) {
            out << "const bool " << name << "_valid = " << access << ".valid != 0;\n";
            if (arg.schema.type == DB_TYPE_TEXT) out << "const db::TextView " << name << " = " << access << ".value.text;\n";
            else if (tagged) out << "const db::TaggedNumeric " << name << " = db::taggedParameter(" << access << ");\n";
            else out << "const " << type << ' ' << name << " = " << name << "_valid ? " << access
                     << (arg.schema.type == DB_TYPE_FLOAT64 ? ".value.f64" : ".value.i64") << " : 0;\n";
        } else out << "const unsigned long long " << name << " = " << access << ".value.u64;\n";
    }
}
void emitBody(std::ostream& out, const KernelContract& contract, const KernelBody& body) {
    if (body.kind == KernelBody::Kind::Range) {
        out << "const std::uint64_t gid[1] = {static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x};\n"
            << "if(gid[0] >= db_range_items) return;\n";
    } else out << "const db::KernelIndex it;\n";
    for (const auto& local : body.locals)
        out << "__shared__ " << storageType(local.type) << ' ' << local.name << '[' << local.elements << "];\n";
    out << body.code << '\n';
}
} // namespace

KernelModuleSource generatePcudacode(const KernelContract& contract, const KernelBody& body,
                                   const std::string& toolchain, bool tagged, bool emit_source) {
    if (body.code.find("sycl::") != std::string::npos) throw std::runtime_error("SYCL device code reached PCUDA module generator");
    KernelModuleSource result;
    result.structure = canonical(contract, body, false, false);
    result.key = canonical(contract, body, true, tagged) + toolchain;
    result.identity = contentHash(result.key);
    for (const auto& arg : contract.arguments) {
        auto schema = arg.schema;
        if (tagged && schema.kind == DB_ARG_PARAMETER && schema.type != DB_TYPE_TEXT) schema.type = DB_TYPE_TAGGED_NUMERIC;
        result.schema.push_back(schema);
    }
    if (!emit_source) return result;
    std::ostringstream out;
    out << "#include \"jit_headers.h\"\nnamespace db_module_" << result.identity << " {\n";
    out << "constexpr int BLOCK_THREADS=128;\nconstexpr int ITEMS_PER_THREAD=4;\nconstexpr int TILE_SIZE=512;\n";
    emitSchema(out, contract, tagged);
    out << "int32_t launch(const KernelInvocationV2* call, KernelErrorV2* error) {\n"
        << "if(!call) return db::kernelError(error,DB_KERNEL_ARGUMENT_ERROR,0,\"null invocation\");\n"
        << "const auto status=db::abi2::validateInvocation(db_module_" << result.identity << "::descriptor,*call,error); if(status) return status;\n"
        << "if(call->launch.block[0]!=" << body.threads << " || call->launch.grid[0]>UINT32_MAX || call->launch.grid[1]!=1 || call->launch.grid[2]!=1 || call->launch.block[1]!=1 || call->launch.block[2]!=1 || call->launch.dynamic_shared_bytes)\n"
        << "return db::kernelError(error,DB_KERNEL_LAUNCH_ERROR,0,\"incompatible 1D/static-shared module launch\");\ntry {\n";
    emitBindings(out, contract, tagged);
    out << "const auto submitted=pcudaParallelFor(dim3(static_cast<unsigned>(call->launch.grid[0])),dim3(" << body.threads
        << "),0,static_cast<pcudaStream_t>(call->stream),[=] {\n";
    emitBody(out, contract, body);
    out << "});\nif(submitted!=pcudaSuccess) return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,pcudaGetErrorString(submitted));\n"
        << "return DB_KERNEL_OK;\n} catch(const std::exception& exception) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,exception.what());}\n"
        << "catch(...) {return db::kernelError(error,DB_KERNEL_BACKEND_ERROR,0,\"unknown PCUDA launch failure\");}\n}\n}\n"
        << "extern \"C\" const char* db_kernel_identity_v2() {return \"" << result.identity << "\";}\n"
        << "extern \"C\" const KernelModuleDescriptorV2* db_kernel_descriptor_v2() {return &db_module_" << result.identity << "::descriptor;}\n"
        << "extern \"C\" int32_t db_kernel_launch_v2(const KernelInvocationV2* call, KernelErrorV2* error) {return db_module_"
        << result.identity << "::launch(call,error);}\n";
    result.source = out.str();
    return result;
}
std::string describePcudamodules(const PlannedQuery& query) {
    std::string result="// Inspection: one independent translation unit per module below.\n"
                       "// Typed candidates; execution may reuse an already loaded tagged fallback.\n";
    for (size_t i = 0; i < query.bodies.size(); ++i) {
        const auto module = generatePcudacode(query.plan.kernels.at(i), query.bodies[i], "inspection");
        result += "// ===== Separate module " + std::to_string(i) + ": " + query.bodies[i].name + " =====\n" + module.source;
    }
    return result;
}
} // namespace db
