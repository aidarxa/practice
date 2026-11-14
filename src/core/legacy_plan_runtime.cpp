#include "core/legacy_plan_runtime.h"
#include "core/module_contract.h"
#include "core/legacy_queue.h"
#include <pcuda.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace db {

struct LegacyQueue::Impl {
    std::shared_ptr<DeviceRuntime> runtime;
    sycl::queue queue;
    explicit Impl(std::shared_ptr<DeviceRuntime> owner)
        : runtime(std::move(owner)), queue(sycl::AdaptiveCpp_pcuda::make_queue(runtime->stream())) {
        if (!queue.has_property<sycl::property::queue::in_order>() || !queue.AdaptiveCpp_extract_inorder_executor())
            throw std::runtime_error("legacy bridge requires the PCUDA dedicated in-order executor");
    }
};
LegacyQueue::LegacyQueue(std::shared_ptr<DeviceRuntime> runtime) : impl_(std::make_unique<Impl>(std::move(runtime))) {}
void* LegacyQueue::handle() const { return &impl_->queue; }
void LegacyQueue::wait() const {
    impl_->queue.wait(); // flush an unexpected non-instant node before checked native completion
    impl_->runtime->wait();
}
LegacyQueue::~LegacyQueue() noexcept {
    try { wait(); }
    catch (...) { std::terminate(); } // never destroy a queue that might still reference borrowed resources
}
namespace {
using ScratchMap = std::unordered_map<std::string, std::unique_ptr<DynamicDeviceBuffer<uint8_t>>>;
size_t hostSize(uint64_t value) {
    if (value > std::numeric_limits<size_t>::max()) throw std::overflow_error("runtime value exceeds size_t");
    return static_cast<size_t>(value);
}
LogicalType logicalType(uint32_t type) {
    if (type == DB_TYPE_INT64) return LogicalType::Int64;
    if (type == DB_TYPE_FLOAT64) return LogicalType::Float64;
    if (type == DB_TYPE_UINT64) return LogicalType::UInt64;
    throw std::runtime_error("unsupported physical result column type");
}
bool sameSchema(const KernelArgumentSchemaV1& left, const KernelArgumentSchemaV1& right) {
    return left.kind == right.kind && left.type == right.type && left.access == right.access && left.flags == right.flags;
}
bool compatibleLegacyLayout(const KernelModuleDescriptorV1& module) {
    return module.abi_version == DB_MODULE_ABI_VERSION && module.struct_size == sizeof(KernelModuleDescriptorV1) &&
           module.backend == DB_BACKEND_LEGACY_SYCL && module.reserved == 0;
}
void validateModuleShape(const KernelModuleDescriptorV1& module, const KernelContract& contract) {
    if (!compatibleLegacyLayout(module)) throw std::runtime_error("incompatible module ABI/backend/layout: " + contract.name);
    if (module.argument_count != contract.arguments.size() || (module.argument_count && !module.arguments))
        throw std::runtime_error("incompatible module argument count: " + contract.name);
}
void validateModuleArguments(const KernelModuleDescriptorV1& module, const KernelContract& contract) {
    for (size_t i = 0; i < contract.arguments.size(); ++i) {
        if (!sameSchema(module.arguments[i], contract.arguments[i].schema))
            throw std::runtime_error("incompatible module argument schema: " + contract.name + " slot " + std::to_string(i));
    }
}

/** @brief All backend-specific operations are confined to this removable host adapter. */
class LegacyPlanBackend final : public PlanBackend {
public:
    LegacyPlanBackend(ExecutionContext& context, const std::vector<LegacyKernelEntry>& kernels)
        : context_(context), kernels_(kernels) {
        if (!context_.runtime_) throw std::runtime_error("physical plan requires a PCUDA runtime");
        context_.runtime_->setMemoryPolicy(context_.config_);
        if (!context_.legacy_queue_) context_.legacy_queue_ = std::make_shared<LegacyQueue>(context_.runtime_);
    }
    BufferView resolve(const PlanBuffer& buffer) override;
    BufferView ensure(const PlanBuffer& buffer, uint64_t elements) override;
    void fill(void* pointer, uint8_t value, size_t bytes) override { context_.runtime_->fillAsync(pointer, value, bytes); }
    void copy(void* destination, const void* source, size_t bytes) override { context_.runtime_->copyAsync(destination, source, bytes); }
    void wait() override { context_.legacy_queue_->wait(); }
    void* stream() override { return context_.legacy_queue_->handle(); }
    uint32_t backend() const override { return DB_BACKEND_LEGACY_SYCL; }
    void validateKernel(size_t index, const KernelContract& contract) override;
    void launch(size_t index, const KernelInvocationV2& call) override;
    void ensureColumns(const std::vector<uint32_t>& types, uint64_t rows) override;
    void publish(const PlanResultShape& shape) override;
private:
    ExecutionContext& context_;
    const std::vector<LegacyKernelEntry>& kernels_;
    ScratchMap temporary_;
    BufferView resultColumn(const PlanBuffer& buffer);
};

BufferView LegacyPlanBackend::resultColumn(const PlanBuffer& buffer) {
    const auto& storage = context_.result_column_storage_.at(buffer.column);
    if (buffer.role == BufferRole::ResultColumnValidity) return {storage.validity->data(), storage.validity->capacity()};
    if (logicalType(buffer.type) != storage.type) throw std::runtime_error("runtime/result column type mismatch");
    if (storage.type == LogicalType::Int64) return {storage.i64->data(), storage.i64->capacity()};
    if (storage.type == LogicalType::Float64) return {storage.f64->data(), storage.f64->capacity()};
    return {storage.u64->data(), storage.u64->capacity()};
}
BufferView LegacyPlanBackend::resolve(const PlanBuffer& buffer) {
    switch (buffer.role) {
        case BufferRole::External: return {context_.getBuffer<int>(buffer.external_name), buffer.external_elements};
        case BufferRole::NullBitmap: {
            auto* pointer = context_.getNullBitmap(buffer.external_name);
            return {pointer, pointer ? buffer.external_elements : 0};
        }
        case BufferRole::Result: return {context_.getResultPointer(), context_.result_buffer_->capacity()};
        case BufferRole::ResultValidity: return {context_.getResultValidityPointer(), context_.result_validity_buffer_->capacity()};
        case BufferRole::ResultColumn:
        case BufferRole::ResultColumnValidity: return resultColumn(buffer);
        default: throw std::runtime_error("scratch/alias must be bound by a physical-plan operation");
    }
}
BufferView LegacyPlanBackend::ensure(const PlanBuffer& buffer, uint64_t elements) {
    const auto bytes = hostSize(checkedSizeMultiply(elements, kernelTypeBytes(buffer.type)));
    auto& map = context_.config_.reuse_scratch_buffers ? context_.plan_scratch_buffers_ : temporary_;
    auto& allocation = map[buffer.name];
    if (!allocation) allocation = std::make_unique<DynamicDeviceBuffer<uint8_t>>(context_.runtime_, 0);
    allocation->ensureCapacity(bytes);
    return {allocation->data(), allocation->capacity() / kernelTypeBytes(buffer.type)};
}
void LegacyPlanBackend::validateKernel(size_t index, const KernelContract& contract) {
    const auto& entry = kernels_.at(index);
    const auto* module = entry.descriptor;
    if (!module || !entry.launch) throw std::runtime_error("missing legacy module entry: " + contract.name);
    validateModuleShape(*module, contract);
    validateModuleArguments(*module, contract);
    context_.runtime_->requireCapabilities(contract.requires_fp64, contract.requires_atomic64, contract.static_shared_bytes);
}
void LegacyPlanBackend::launch(size_t, const KernelInvocationV2&) {
    throw std::runtime_error("Historical ABI v1 adapter cannot submit a native ABI v2 invocation");
}
void LegacyPlanBackend::ensureColumns(const std::vector<uint32_t>& types, uint64_t rows) {
    if (types.size() != context_.result_columns_.size()) throw std::runtime_error("physical result width differs from inferred schema");
    for (size_t i = 0; i < types.size(); ++i) {
        const auto type = logicalType(types[i]);
        if (type != context_.result_columns_[i].type) throw std::runtime_error("physical result type differs from inferred schema");
    }
    context_.ensureColumnarResultCapacity(types.size(), hostSize(rows));
}
void LegacyPlanBackend::publish(const PlanResultShape& shape) {
    context_.result_row_count_ = hostSize(shape.rows);
    context_.expected_result_size_ = hostSize(shape.values);
    context_.expected_result_validity_words_ = hostSize(shape.validity_words);
    context_.result_is_dense_ = shape.dense;
}
} // namespace

void prepareLegacyPlanResult(ExecutionContext& context, size_t elements) {
    if (!context.runtime_ || !context.result_buffer_ || !context.result_validity_buffer_)
        throw std::runtime_error("missing PCUDA runtime/result storage");
    context.runtime_->setMemoryPolicy(context.config_);
    context.result_buffer_->ensureCapacity(elements);
    context.ensureResultValidityCapacity(elements, false);
}
void executeLegacyPlan(const ExecutionPlan& plan, const ParameterValues& values,
                        const std::vector<LegacyKernelEntry>& kernels, ExecutionContext& context) {
    LegacyPlanBackend backend(context, kernels);
    context.tuple_size_ = plan.tuple_size;
    executePlan(plan, backend, values);
}
} // namespace db
