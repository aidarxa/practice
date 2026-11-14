#include "core/physical_planner.h"
#include "core/module_contract_v2.h"
#include "core/expression_codegen.h"
#include "core/code_symbols.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace db {
namespace {
bool identifierStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool identifierPart(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }
bool validIdentifier(const std::string& name) {
    return !name.empty() && identifierStart(name.front()) && std::all_of(name.begin(), name.end(), identifierPart);
}
bool readOnly(BufferRole role) {
    return role == BufferRole::External || role == BufferRole::NullBitmap ||
           role == BufferRole::TextBytes || role == BufferRole::TextOffsets || role == BufferRole::TextValidity;
}
bool sameBuffer(const PlanBuffer& a, const PlanBuffer& b) {
    return a.type == b.type && a.role == b.role && a.external_name == b.external_name && a.column == b.column &&
           a.optional == b.optional && a.external_elements == b.external_elements && a.legacy_cpp_type == b.legacy_cpp_type;
}
bool sameKernelDefinition(const KernelBody& a, const KernelBody& b) {
    const auto same_local = [](const KernelLocalArray& left, const KernelLocalArray& right) {
        return left.name == right.name && left.type == right.type && left.elements == right.elements;
    };
    return a.code == b.code && a.kind == b.kind && a.item == b.item && a.threads == b.threads && a.scalar_captures == b.scalar_captures &&
           std::equal(a.locals.begin(), a.locals.end(), b.locals.begin(), b.locals.end(), same_local);
}
KernelArgumentBinding scalarBinding(const std::string& name) {
    return {name, {DB_ARG_SCALAR, DB_TYPE_UINT64, DB_ACCESS_READ, 0}, variable(name)};
}
KernelArgumentBinding bufferBinding(const PlanBuffer& buffer) {
    std::string cpp_type = buffer.legacy_cpp_type;
    if (buffer.role == BufferRole::NullBitmap || buffer.role == BufferRole::ResultValidity || buffer.role == BufferRole::ResultColumnValidity)
        cpp_type = "std::uint64_t";
    if (buffer.role == BufferRole::ResultColumn && buffer.type == DB_TYPE_UINT64) cpp_type = "std::uint64_t";
    return {buffer.name, {DB_ARG_BUFFER, buffer.type, readOnly(buffer.role) ? DB_ACCESS_READ : DB_ACCESS_READ_WRITE,
                         buffer.optional ? DB_ARG_NULLABLE : 0u}, buffer.name, cpp_type};
}
bool needsAtomic64(const std::set<std::string>& names) {
    static const char* helpers[] = {"atomic_add_ull", "atomic_fetch_add_ull", "atomic_min_ull", "atomic_max_ull", "atomic_set_valid_bit", "atomicExch", "atomicAggregate", "atomicCAS", "atomicAdd"};
    return std::any_of(std::begin(helpers), std::end(helpers), [&](const auto* name) { return names.count(name) != 0; });
}
void deviceRequirements(KernelContract& contract, const KernelBody& body, const std::set<std::string>& names) {
    contract.requires_fp64 = names.count("double") || names.count("safe_div");
    for (const auto& arg : contract.arguments) contract.requires_fp64 |= arg.schema.type == DB_TYPE_FLOAT64;
    contract.requires_atomic64 = needsAtomic64(names);
    for (const auto& local : body.locals)
        contract.static_shared_bytes = checkedSizeAdd(contract.static_shared_bytes, checkedSizeMultiply(abi2::kernelTypeBytes(local.type), local.elements));
}
void bindKernel(KernelContract& contract, const KernelBody& body, const std::vector<PlanBuffer>& buffers,
                 const std::map<std::string, SizeExpr>& common_scalars, const ParameterSchema& parameters) {
    const auto names = codeIdentifiers(body.code);
    for (const auto& buffer : buffers) if (names.count(buffer.name)) contract.arguments.push_back(bufferBinding(buffer));
    std::set<std::string> scalars(body.scalar_captures.begin(), body.scalar_captures.end());
    for (const auto& scalar : common_scalars) if (names.count(scalar.first)) scalars.insert(scalar.first);
    for (const auto& name : scalars) contract.arguments.push_back(scalarBinding(name));
    for (size_t slot = 0; slot < parameters.size(); ++slot) {
        const auto name = parameterName(slot);
        if (!names.count(name) && !names.count(name + "_valid")) continue;
        const auto type = parameters[slot] == ParameterType::Float64 ? DB_TYPE_FLOAT64 :
                          parameters[slot] == ParameterType::Int64 ? DB_TYPE_INT64 :
                          parameters[slot] == ParameterType::Text ? static_cast<int>(DB_TYPE_TEXT) : DB_TYPE_NULL_ONLY;
        contract.arguments.push_back({name, {DB_ARG_PARAMETER, static_cast<uint32_t>(type), DB_ACCESS_READ, DB_ARG_NULLABLE}, ParameterSlot{slot}});
    }
    if (body.kind == KernelBody::Kind::Range)
        contract.arguments.push_back({"db_range_items", {DB_ARG_SCALAR, DB_TYPE_UINT64, DB_ACCESS_READ, 0}, body.range_items});
    deviceRequirements(contract, body, names);
}
} // namespace

bool PhysicalPlanBuilder::declareBuffer(PlanBuffer buffer) {
    if (!validIdentifier(buffer.name)) throw std::runtime_error("invalid physical buffer identifier: " + buffer.name);
    const auto found = std::find_if(plan.buffers.begin(), plan.buffers.end(), [&](const auto& old) { return old.name == buffer.name; });
    if (found == plan.buffers.end()) { plan.buffers.push_back(std::move(buffer)); return true; }
    if (!sameBuffer(*found, buffer))
        throw std::runtime_error("inconsistent physical buffer declaration: " + buffer.name);
    return false;
}
void PhysicalPlanBuilder::scratch(PlanBlock& block, const std::string& name, uint32_t type, SizeExpr elements, bool clear, const std::string& cpp_type) {
    declareBuffer({name, type, BufferRole::Scratch, {}, 0, false, 0, cpp_type});
    block.emplace_back(EnsureBuffer{name, elements});
    if (clear) block.emplace_back(FillBuffer{name, 0, maximum(elements, 1), 0});
}
void PhysicalPlanBuilder::kernel(PlanBlock& block, KernelBody body, SizeExpr grid, uint32_t threads) {
    body.threads = threads;
    if (!validIdentifier(body.name)) throw std::runtime_error("invalid kernel identifier: " + body.name);
    const auto existing = std::find_if(bodies_.begin(), bodies_.end(), [&](const auto& old) { return old.name == body.name; });
    if (existing != bodies_.end()) {
        if (!sameKernelDefinition(*existing, body))
            throw std::runtime_error("conflicting kernel definition: " + body.name);
        block.emplace_back(LaunchKernel{static_cast<size_t>(existing - bodies_.begin()), std::move(grid), threads, 0});
        return;
    }
    const auto index = bodies_.size();
    plan.kernels.push_back({body.name, {}});
    bodies_.push_back(std::move(body));
    block.emplace_back(LaunchKernel{index, std::move(grid), threads, 0});
}
void PhysicalPlanBuilder::columns(PlanBlock& block, const std::vector<uint32_t>& types, SizeExpr rows) {
    for (size_t i = 0; i < types.size(); ++i) {
        declareBuffer({"d_result_col_" + std::to_string(i), types[i], BufferRole::ResultColumn, {}, i, false});
        declareBuffer({"d_result_validity_col_" + std::to_string(i), DB_TYPE_UINT64, BufferRole::ResultColumnValidity, {}, i, false});
    }
    block.emplace_back(EnsureColumns{types, std::move(rows)});
}
void PhysicalPlanBuilder::range(PlanBlock& block, const std::string& name, const std::string& code, SizeExpr items, std::vector<std::string> captures) {
    KernelBody body{KernelBody::Kind::Range, name, "gid", code, {}, std::move(captures)};
    const auto count = "range_items_" + name;
    block.emplace_back(SetScalar{count,items});
    body.range_items = variable(count);
    kernel(block, std::move(body), ceilDiv(items, 128), 128);
}
void PhysicalPlanBuilder::ndRange(PlanBlock& block, const std::string& name, const std::string& code, SizeExpr groups,
                                  uint32_t threads, std::vector<std::string> captures, std::vector<KernelLocalArray> locals) {
    if (code.find("db_collective_u64") != std::string::npos) locals.push_back({"db_collective_u64", DB_TYPE_UINT64, threads});
    if (code.find("db_collective_i32") != std::string::npos) locals.push_back({"db_collective_i32", DB_TYPE_INT32, threads});
    if (code.find("db_collective_i64") != std::string::npos) locals.push_back({"db_collective_i64", DB_TYPE_INT64, threads});
    if (code.find("db_collective_f64") != std::string::npos) locals.push_back({"db_collective_f64", DB_TYPE_FLOAT64, threads});
    kernel(block, {KernelBody::Kind::NdRange, name, "it", code, std::move(locals), std::move(captures)}, std::move(groups), threads);
}
void PhysicalPlanBuilder::commonScalar(const std::string& name, SizeExpr value) { common_scalars_.insert_or_assign(name, std::move(value)); }

PlannedQuery PhysicalPlanBuilder::finish(size_t tuple_size, SizeExpr result_storage_elements, const ParameterSchema& parameters) const {
    PlannedQuery result{plan, bodies_};
    result.plan.tuple_size = tuple_size;
    PlanBlock prelude;
    for (const auto& scalar : common_scalars_) prelude.emplace_back(SetScalar{scalar.first, scalar.second});
    prelude.emplace_back(FillBuffer{"d_result", 0, result_storage_elements, 0});
    for (const auto& buffer : result.plan.buffers) {
        if (buffer.role == BufferRole::ResultValidity)
            prelude.emplace_back(FillBuffer{buffer.name, 0, maximum(ceilDiv(result_storage_elements, 64), 1), 0});
    }
    prelude.insert(prelude.end(), result.plan.steps.begin(), result.plan.steps.end());
    prelude.insert(prelude.end(), post.begin(), post.end());
    result.plan.steps = std::move(prelude);
    for (size_t i = 0; i < result.bodies.size(); ++i) bindKernel(result.plan.kernels[i], result.bodies[i], result.plan.buffers, common_scalars_, parameters);
    return result;
}
} // namespace db
