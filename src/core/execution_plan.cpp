#include "core/execution_plan.h"
#include "core/module_contract_v2.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace db {
namespace {
/** @brief Keep a view, its storage type and its accessible byte range together. */
struct Slice { void* data; size_t bytes; };

/** @brief Drain before a borrowed host destination/upload snapshot leaves scope, including errors. */
class PlanCompletion {
public:
    explicit PlanCompletion(PlanBackend& backend) : backend_(backend) {}
    ~PlanCompletion() noexcept {
        if (!pending_) return;
        try { backend_.wait(); } catch (...) { std::terminate(); }
    }
    void finish() { backend_.wait(); pending_ = false; }
private:
    PlanBackend& backend_;
    bool pending_ = true;
};

KernelArgumentV2 parameterArgument(const KernelArgumentSchemaV2& schema, const ParameterValue& value) {
    KernelArgumentV2 result{schema, nullptr, 0, {}, value.isNull() ? 0u : 1u, 0};
    const auto types = std::array<uint32_t, 4>{DB_TYPE_NULL_ONLY, DB_TYPE_INT64, DB_TYPE_FLOAT64, DB_TYPE_TEXT};
    result.schema.type = types.at(static_cast<size_t>(value.type()));
    if (!value.isNull() && value.type() == ParameterType::Int64) result.value.i64 = value.asInt64();
    if (!value.isNull() && value.type() == ParameterType::Float64) result.value.f64 = value.asFloat64();
    if (!value.isNull() && value.type() == ParameterType::Text)
        result.value.text = {reinterpret_cast<const uint8_t*>(value.asText().data()), value.asText().size()};
    return result;
}

/** @brief An execution owns scalar state, aliases, argument snapshots and pending upload sources. */
class PlanExecution {
public:
    PlanExecution(const ExecutionPlan& plan, PlanBackend& backend, const ParameterValues& parameters)
        : plan_(plan), backend_(backend), parameters_(parameters) {
        for (const auto slot : plan.size_parameters) {
            const auto value = parameters_.at(slot).asInt64();
            if (value < 0) throw std::runtime_error("LIMIT is outside the supported size range");
            scalars_["db_size_parameter_" + std::to_string(slot)] = static_cast<uint64_t>(value);
        }
        for (const auto& buffer : plan.buffers) {
            if (!abi2::kernelTypeBytes(buffer.type) || !buffers_.emplace(buffer.name, &buffer).second)
                throw std::runtime_error("invalid or duplicate physical buffer: " + buffer.name);
        }
    }
    void validate();
    void run(const PlanBlock& block);
private:
    const ExecutionPlan& plan_;
    PlanBackend& backend_;
    const ParameterValues parameters_;
    PlanScalars scalars_;
    std::unordered_map<std::string, const PlanBuffer*> buffers_;
    std::unordered_map<std::string, BufferView> scratch_;
    std::vector<std::vector<uint64_t>> pending_uploads_;

    const PlanBuffer& buffer(const std::string& name) const { return *buffers_.at(name); }
    BufferView view(const std::string& name);
    Slice slice(const std::string& name, uint64_t offset, uint64_t elements);
    std::vector<uint64_t> read(const std::string& name, uint64_t offset, uint64_t count);
    KernelArgumentV2 argument(const KernelArgumentBinding& binding);
    void validateBinding(const KernelArgumentBinding& binding) const;
    void validateBindingValue(const KernelArgumentBinding& binding, const std::string& name) const;
    void validateBindingValue(const KernelArgumentBinding& binding, const ParameterSlot& slot) const;
    void validateBindingValue(const KernelArgumentBinding& binding, const SizeExpr&) const;
    void apply(const SetScalar& step) { scalars_[step.name] = step.value.evaluate(scalars_); }
    void apply(const EnsureBuffer& step);
    void apply(const FillBuffer& step);
    void apply(const CopyBuffer& step);
    void apply(const AliasBuffer& step);
    void apply(const LaunchKernel& step);
    void apply(const ReadScalar& step);
    void apply(const HostPrefix& step);
    void apply(const DevicePrefix& step);
    void apply(const HostReduce& step);
    void apply(const SelectRadixBucket& step);
    void apply(const EnsureColumns& step) { backend_.ensureColumns(step.types, step.rows.evaluate(scalars_)); }
    void apply(const PublishResult& step);
    void apply(const Require& step);
    void apply(const Wait&) { backend_.wait(); }
    void apply(const Branch& step) { run(step.condition.evaluate(scalars_) ? step.yes : step.no); }
    void apply(const Loop& step);
};

void PlanExecution::validateBinding(const KernelArgumentBinding& binding) const {
    if (!abi2::validSchema(binding.schema)) throw std::runtime_error("invalid plan argument schema: " + binding.name);
    std::visit([&](const auto& value) { validateBindingValue(binding, value); }, binding.value);
}
void PlanExecution::validateBindingValue(const KernelArgumentBinding& binding, const std::string& name) const {
    if (binding.schema.kind != DB_ARG_BUFFER || buffer(name).type != binding.schema.type)
        throw std::runtime_error("plan buffer binding type mismatch: " + binding.name);
}
void PlanExecution::validateBindingValue(const KernelArgumentBinding& binding, const ParameterSlot& slot) const {
    if (binding.schema.kind != DB_ARG_PARAMETER || slot.index >= parameters_.size())
        throw std::runtime_error("missing physical-plan parameter: " + binding.name);
    const auto argument = parameterArgument(binding.schema, parameters_[slot.index]);
    if (abi2::validateArgument(binding.schema, argument) != DB_KERNEL_OK) throw std::runtime_error("incompatible physical-plan parameter: " + binding.name);
}
void PlanExecution::validateBindingValue(const KernelArgumentBinding& binding, const SizeExpr&) const {
    const auto valid_size_scalar = binding.schema.kind == DB_ARG_SCALAR && binding.schema.type == DB_TYPE_UINT64;
    if (!valid_size_scalar)
        throw std::runtime_error("size expression must bind a UInt64 scalar: " + binding.name);
}

void PlanExecution::validate() {
    if (!plan_.tuple_size) throw std::runtime_error("zero result tuple width");
    for (size_t i = 0; i < plan_.kernels.size(); ++i) {
        for (const auto& binding : plan_.kernels[i].arguments) validateBinding(binding);
        backend_.validateKernel(i, plan_.kernels[i]);
    }
}
BufferView PlanExecution::view(const std::string& name) {
    const auto& spec = buffer(name);
    if (spec.role == BufferRole::Scratch || spec.role == BufferRole::Alias) return scratch_.at(name);
    return backend_.resolve(spec);
}
Slice PlanExecution::slice(const std::string& name, uint64_t offset, uint64_t elements) {
    const auto data = view(name);
    if (offset > data.elements || elements > data.elements - offset)
        throw std::out_of_range("physical buffer range exceeds capacity: " + name);
    const auto bytes = abi2::kernelTypeBytes(buffer(name).type);
    const auto start = checkedSizeMultiply(offset, bytes);
    const auto size = checkedSizeMultiply(elements, bytes);
    if (size > std::numeric_limits<size_t>::max() || start > std::numeric_limits<size_t>::max())
        throw std::overflow_error("physical buffer byte range overflow");
    if (size && !data.data) throw std::runtime_error("null physical buffer: " + name);
    return {data.data ? static_cast<uint8_t*>(data.data) + start : nullptr, static_cast<size_t>(size)};
}
void PlanExecution::apply(const EnsureBuffer& step) {
    const auto& spec = buffer(step.buffer);
    if (spec.role != BufferRole::Scratch) throw std::runtime_error("only runtime scratch may use EnsureBuffer");
    const auto count = maximum(step.elements, 1).evaluate(scalars_);
    const auto bytes = checkedSizeMultiply(count, abi2::kernelTypeBytes(spec.type));
    if (bytes > std::numeric_limits<size_t>::max()) throw std::overflow_error("scratch byte size overflow");
    scratch_[step.buffer] = backend_.ensure(spec, count);
}
void PlanExecution::apply(const FillBuffer& step) {
    const auto range = slice(step.buffer, step.offset.evaluate(scalars_), step.elements.evaluate(scalars_));
    if (range.bytes) backend_.fill(range.data, step.value, range.bytes);
}
void PlanExecution::apply(const CopyBuffer& step) {
    if (buffer(step.source).type != buffer(step.destination).type) throw std::runtime_error("service copy storage type mismatch");
    const auto count = step.elements.evaluate(scalars_);
    const auto src = slice(step.source, step.source_offset.evaluate(scalars_), count);
    const auto dst = slice(step.destination, step.destination_offset.evaluate(scalars_), count);
    if (src.bytes) backend_.copy(dst.data, src.data, src.bytes);
}
void PlanExecution::apply(const AliasBuffer& step) {
    if (buffer(step.destination).role != BufferRole::Alias || buffer(step.destination).type != buffer(step.source).type)
        throw std::runtime_error("invalid physical buffer alias");
    if (step.source == step.destination) throw std::runtime_error("self-referencing physical buffer alias");
    // Resolve now: later ping-pong rebinding must not change an earlier selection.
    const auto selected = view(step.source);
    scratch_[step.destination] = selected;
    // Aliases hold snapshots just like launch arguments, not mutable name chains.
}

KernelArgumentV2 PlanExecution::argument(const KernelArgumentBinding& binding) {
    KernelArgumentV2 result{binding.schema, nullptr, 0, {}, 1, 0};
    if (const auto* name = std::get_if<std::string>(&binding.value)) {
        const auto data = view(*name);
        result.data = data.data;
        result.element_count = data.elements;
    } else if (const auto* expression = std::get_if<SizeExpr>(&binding.value)) {
        result.value.u64 = expression->evaluate(scalars_);
    } else {
        const auto slot = std::get<ParameterSlot>(binding.value).index;
        const auto& value = parameters_.at(slot);
        result = parameterArgument(binding.schema, value);
        if (value.type() == ParameterType::Text && !value.isNull()) result.value.text = backend_.textParameter(slot, value);
        return result;
    }
    return result;
}
void PlanExecution::apply(const LaunchKernel& step) {
    const auto grid = step.grid.evaluate(scalars_);
    if (!grid) return;
    const auto& contract = plan_.kernels.at(step.kernel);
    std::vector<KernelArgumentV2> args;
    std::vector<KernelArgumentSchemaV2> schema;
    for (const auto& binding : contract.arguments) { args.push_back(argument(binding)); schema.push_back(binding.schema); }
    KernelInvocationV2 call{DB_MODULE_ABI_V2_VERSION, sizeof(KernelInvocationV2), backend_.backend(), 0,
                            backend_.stream(), args.data(), args.size(), {{grid, 1, 1}, {step.block, 1, 1}, step.shared_bytes}};
    KernelModuleDescriptorV2 module{DB_MODULE_ABI_V2_VERSION, sizeof(KernelModuleDescriptorV2), backend_.backend(), 0, schema.data(), schema.size()};
    KernelErrorV2 error{sizeof(KernelErrorV2), 0, 0, {}};
    if (abi2::validateInvocation(module, call, &error) != DB_KERNEL_OK)
        throw std::runtime_error(contract.name + " argument " + std::to_string(error.argument_index) + ": " + error.message);
    backend_.launch(step.kernel, call); // wrappers copy values into their kernel closure before returning
}
std::vector<uint64_t> PlanExecution::read(const std::string& name, uint64_t offset, uint64_t count) {
    if (buffer(name).type != DB_TYPE_UINT64) throw std::runtime_error("host counter service requires UInt64");
    const auto src = slice(name, offset, count);
    std::vector<uint64_t> result(static_cast<size_t>(count));
    PlanCompletion transfer(backend_);
    if (src.bytes) backend_.copy(result.data(), src.data, src.bytes);
    transfer.finish(); // required before reading the copied CPU counters
    return result;
}
void PlanExecution::apply(const ReadScalar& step) {
    const auto source = slice(step.buffer, step.offset.evaluate(scalars_), 1);
    if (buffer(step.buffer).type == DB_TYPE_INT32) {
        int32_t value = 0;
        PlanCompletion transfer(backend_);
        backend_.copy(&value, source.data, source.bytes);
        transfer.finish();
        if (value < 0) throw std::overflow_error("negative/overflowed Int32 device counter");
        scalars_[step.destination] = static_cast<uint64_t>(value);
    } else {
        scalars_[step.destination] = read(step.buffer, step.offset.evaluate(scalars_), 1).front();
    }
}
void PlanExecution::apply(const HostPrefix& step) {
    const auto counts = read(step.counts, 0, step.elements.evaluate(scalars_));
    pending_uploads_.emplace_back(counts.size());
    auto& offsets = pending_uploads_.back();
    uint64_t total = 0;
    for (size_t i = 0; i < counts.size(); ++i) { offsets[i] = total; total = checkedSizeAdd(total, counts[i]); }
    const auto dst = slice(step.offsets, 0, counts.size());
    if (dst.bytes) backend_.copy(dst.data, offsets.data(), dst.bytes);
    scalars_[step.total] = total;
}
void PlanExecution::apply(const DevicePrefix& step) {
    const auto type=buffer(step.counts).type;
    if(type!=buffer(step.offsets).type) throw std::runtime_error("device prefix storage type mismatch");
    const auto count=step.elements.evaluate(scalars_);
    const auto source=slice(step.counts,0,count);
    const auto destination=slice(step.offsets,0,count);
    scalars_[step.total]=backend_.prefix(destination.data,source.data,type,count);
}
void PlanExecution::apply(const HostReduce& step) {
    const auto values = read(step.buffer, step.offset.evaluate(scalars_), step.elements.evaluate(scalars_));
    uint64_t result = step.kind == HostReduce::Kind::Minimum ? std::numeric_limits<uint64_t>::max() : 0;
    for (const auto value : values) {
        if (step.kind == HostReduce::Kind::Sum) result = checkedSizeAdd(result, value);
        else if (step.kind == HostReduce::Kind::Minimum) result = std::min(result, value);
        else result = std::max(result, value);
    }
    scalars_[step.destination] = result;
}
void PlanExecution::apply(const SelectRadixBucket& step) {
    const auto counts = read(step.buffer, 0, 256);
    auto& rank = scalars_.at(step.rank);
    if (!rank) throw std::runtime_error("radix rank must be positive");
    for (size_t index = counts.size(); index > 0; --index) {
        if (rank > counts[index - 1]) rank -= counts[index - 1];
        else {
            scalars_.at(step.prefix) = checkedSizeAdd(checkedSizeMultiply(scalars_.at(step.prefix), 256), index - 1);
            return;
        }
    }
    throw std::runtime_error("radix histogram does not contain requested rank");
}
void PlanExecution::apply(const PublishResult& step) {
    const auto shape = PlanResultShape{step.rows.evaluate(scalars_), step.values.evaluate(scalars_), step.dense, step.validity_words.evaluate(scalars_)};
    backend_.publish(shape);
    scalars_["result_rows"] = shape.rows;
}
void PlanExecution::apply(const Require& step) {
    if (!step.condition.evaluate(scalars_)) throw std::runtime_error(step.message);
}
void PlanExecution::apply(const Loop& step) {
    uint64_t iterations = 0;
    while (step.condition.evaluate(scalars_)) {
        if (iterations++ == step.maximum_iterations) throw std::runtime_error("physical plan loop iteration bound exceeded");
        run(step.body);
    }
}
void PlanExecution::run(const PlanBlock& block) {
    for (const auto& step : block) std::visit([&](const auto& operation) { apply(operation); }, step.operation);
}

void describeBlock(std::ostringstream& out, const PlanBlock& block, size_t depth) {
    static constexpr const char* names[] = {"set", "ensure", "fill", "copy", "alias", "launch", "read-counter", "host-prefix", "host-reduce", "radix-bucket", "ensure-columns", "publish", "require", "wait", "branch", "loop", "device-prefix"};
    for (const auto& step : block) {
        out << std::string(depth * 2, ' ') << names[step.operation.index()];
        if (const auto* launch = std::get_if<LaunchKernel>(&step.operation)) out << " kernel=" << launch->kernel << " grid=" << launch->grid.describe() << " block=" << launch->block;
        out << '\n';
        if (const auto* branch = std::get_if<Branch>(&step.operation)) { describeBlock(out, branch->yes, depth + 1); describeBlock(out, branch->no, depth + 1); }
        if (const auto* loop = std::get_if<Loop>(&step.operation)) describeBlock(out, loop->body, depth + 1);
    }
}
} // namespace

void executePlan(const ExecutionPlan& plan, PlanBackend& backend, const ParameterValues& parameters) {
    PlanExecution execution(plan, backend, parameters);
    execution.validate();
    PlanCompletion completion(backend); // destroyed before execution's pending upload sources
    execution.run(plan.steps);
    completion.finish();
}
std::string describePlan(const ExecutionPlan& plan) {
    std::ostringstream out;
    for (size_t i = 0; i < plan.kernels.size(); ++i) {
        out << "kernel " << i << ' ' << plan.kernels[i].name << '\n';
        for (const auto& arg : plan.kernels[i].arguments) out << "  argument " << arg.name << " kind=" << arg.schema.kind << " type=" << arg.schema.type << " access=" << arg.schema.access << '\n';
    }
    describeBlock(out, plan.steps, 0);
    return out.str();
}
} // namespace db
