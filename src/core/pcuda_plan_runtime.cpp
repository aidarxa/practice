#include "core/pcuda_plan_runtime.h"
#include "core/module_contract_v2.h"
#include "core/pcuda_plan_services.h"
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace db {
namespace {
size_t hostSize(uint64_t value) {
    if (value > std::numeric_limits<size_t>::max()) throw std::overflow_error("runtime value exceeds size_t");
    return static_cast<size_t>(value);
}
LogicalType logicalType(uint32_t type) {
    if (type == DB_TYPE_INT64) return LogicalType::Int64;
    if (type == DB_TYPE_FLOAT64) return LogicalType::Float64;
    if (type == DB_TYPE_UINT64) return LogicalType::UInt64;
    if (type == DB_TYPE_TEXT) return LogicalType::Text;
    throw std::runtime_error("unsupported physical result column type");
}
void validateModule(const KernelModuleDescriptorV2& module, const KernelContract& contract) {
    if (module.abi_version!=DB_MODULE_ABI_V2_VERSION || module.struct_size!=sizeof(module) || module.backend!=DB_BACKEND_PCUDA || module.reserved)
        throw std::runtime_error("incompatible PCUDA module ABI/backend/layout: "+contract.name);
    if (module.argument_count!=contract.arguments.size() || (module.argument_count && !module.arguments))
        throw std::runtime_error("incompatible PCUDA module argument count: "+contract.name);
    for (size_t i=0;i<contract.arguments.size();++i) {
        if (!compatibleArgumentSchema(module.arguments[i],contract.arguments[i].schema))
            throw std::runtime_error("incompatible PCUDA module argument schema: "+contract.name+" slot "+std::to_string(i));
    }
}
/** @brief All resources stay with the session. No SYCL view or module-owned query allocation exists. */
class PcudaplanBackend final : public PlanBackend {
public:
    PcudaplanBackend(ExecutionContext& context,const ExecutionPlan& plan,const ParameterValues& values,
                     const LoadedKernels* kernels,const KernelLookup* ready=nullptr)
        : context_(context),plan_(plan),kernels_(kernels),ready_(ready),parameters_(values) {
        if (!context_.runtime_) throw std::runtime_error("physical plan requires a PCUDA runtime");
        context_.runtime_->activate();
        context_.runtime_->setMemoryPolicy(context_.config_);
    }
    BufferView resolve(const PlanBuffer&) override;
    BufferView ensure(const PlanBuffer&,uint64_t) override;
    void fill(void* p,uint8_t value,size_t bytes) override {context_.runtime_->fillAsync(p,value,bytes);}
    void copy(void* destination,const void* source,size_t bytes) override {context_.runtime_->copyAsync(destination,source,bytes);}
    void wait() override {context_.runtime_->wait();}
    void* stream() override {return context_.runtime_->stream();}
    uint32_t backend() const override {return DB_BACKEND_PCUDA;}
    void validateKernel(size_t,const KernelContract&) override;
    void launch(size_t,const KernelInvocationV2&) override;
    DbTextViewV2 textParameter(size_t, const ParameterValue&) override;
    void ensureColumns(const std::vector<uint32_t>&,uint64_t) override;
    uint64_t prefix(void* destination,const void* source,uint32_t type,uint64_t count) override;
    void publish(const PlanResultShape& shape) override {
        context_.result_row_count_=hostSize(shape.rows);
        context_.expected_result_size_=hostSize(shape.values);
        context_.expected_result_validity_words_=hostSize(shape.validity_words);
        context_.result_is_dense_=shape.dense;
    }
private:
    ExecutionContext& context_;
    const ExecutionPlan& plan_;
    const LoadedKernels* kernels_;
    const KernelLookup* ready_;
    const ParameterValues& parameters_;
    std::vector<uint64_t> parameter_offsets_;
    bool text_uploaded_ = false;
    std::unordered_map<std::string,std::unique_ptr<DynamicDeviceBuffer<uint8_t>>> temporary_;
    std::shared_ptr<PlanPrefixService> temporary_prefix_;
    BufferView resultColumn(const PlanBuffer&);
    BufferView textColumn(const PlanBuffer&);
    void validateReady(const std::shared_ptr<LoadedKernelModule>&,const KernelContract&);
    void submit(const LoadedKernelModule&,const KernelInvocationV2&);
};
BufferView PcudaplanBackend::resultColumn(const PlanBuffer& buffer) {
    const auto& storage=context_.result_column_storage_.at(buffer.column);
    if (buffer.role==BufferRole::ResultColumnValidity) return {storage.validity->data(),storage.validity->capacity()};
    if (logicalType(buffer.type)!=storage.type) throw std::runtime_error("runtime/result column type mismatch");
    if (storage.type==LogicalType::Int64) return {storage.i64->data(),storage.i64->capacity()};
    if (storage.type==LogicalType::Float64) return {storage.f64->data(),storage.f64->capacity()};
    if (storage.type==LogicalType::Text) return {storage.text->data(),storage.text->capacity()};
    return {storage.u64->data(),storage.u64->capacity()};
}
BufferView PcudaplanBackend::resolve(const PlanBuffer& buffer) {
    switch(buffer.role) {
        case BufferRole::TextBytes:
        case BufferRole::TextOffsets:
        case BufferRole::TextValidity: return textColumn(buffer);
        case BufferRole::External: return {context_.getBuffer<int>(buffer.external_name),buffer.external_elements};
        case BufferRole::NullBitmap: {
            auto* pointer=context_.getNullBitmap(buffer.external_name);
            return {pointer,pointer?buffer.external_elements:0};
        }
        case BufferRole::Result: return {context_.getResultPointer(),context_.result_buffer_->capacity()};
        case BufferRole::ResultValidity: return {context_.getResultValidityPointer(),context_.result_validity_buffer_->capacity()};
        case BufferRole::ResultColumn:
        case BufferRole::ResultColumnValidity: return resultColumn(buffer);
        default: throw std::runtime_error("scratch/alias must be bound by a physical-plan operation");
    }
}
BufferView PcudaplanBackend::ensure(const PlanBuffer& buffer,uint64_t elements) {
    const auto bytes=hostSize(checkedSizeMultiply(elements,abi2::kernelTypeBytes(buffer.type)));
    auto& map=context_.config_.reuse_scratch_buffers?context_.plan_scratch_buffers_:temporary_;
    auto& allocation=map[buffer.name];
    if (!allocation) allocation=std::make_unique<DynamicDeviceBuffer<uint8_t>>(context_.runtime_,0);
    allocation->ensureCapacity(bytes);
    return {allocation->data(),allocation->capacity()/abi2::kernelTypeBytes(buffer.type)};
}
void PcudaplanBackend::validateKernel(size_t index,const KernelContract& contract) {
    if (ready_) {
        context_.runtime_->requireCapabilities(contract.requires_fp64,contract.requires_atomic64,contract.static_shared_bytes);
        return;
    }
    validateReady(kernels_->at(index),contract);
}
/** @brief Loaded ABI, tagged-FP64 requirements and completion ownership precede every staged submission. */
void PcudaplanBackend::validateReady(const std::shared_ptr<LoadedKernelModule>& module,const KernelContract& contract) {
    if (!module || !module->descriptor() || !module->launch()) throw std::runtime_error("missing PCUDA module entry");
    validateModule(*module->descriptor(),contract);
    bool fp64=contract.requires_fp64;
    for (size_t i=0;i<module->descriptor()->argument_count;++i)
        fp64 |= module->descriptor()->arguments[i].type==DB_TYPE_TAGGED_NUMERIC;
    context_.runtime_->requireCapabilities(fp64,contract.requires_atomic64,contract.static_shared_bytes);
    const auto owner=context_.runtime_;
    module->retainCompletion(owner.get(),[owner] {owner->waitNoThrow();});
}
void PcudaplanBackend::launch(size_t index,const KernelInvocationV2& call) {
    if (ready_) {
        const auto module=(*ready_)(index);
        validateReady(module,plan_.kernels.at(index));
        submit(*module,call);
    } else submit(*kernels_->at(index),call);
}
/** @brief Keep the ordinary ready path's borrowing behavior; no extra shared-owner copy per warm launch. */
void PcudaplanBackend::submit(const LoadedKernelModule& module,const KernelInvocationV2& call) {
    context_.runtime_->validateLaunch(call.launch);
    KernelErrorV1 error{sizeof(KernelErrorV1),0,0,{}};
    const auto status=module.launch()(&call,&error);
    if (status!=DB_KERNEL_OK) throw std::runtime_error("PCUDA kernel launch failed: "+std::string(error.message));
}
BufferView PcudaplanBackend::textColumn(const PlanBuffer& buffer) {
    const auto& column = context_.text_columns_.at(buffer.external_name);
    if (buffer.role == BufferRole::TextBytes) return {column.bytes->data(), column.bytes->capacity()};
    if (buffer.role == BufferRole::TextValidity)
        return {column.validity ? column.validity->data() : nullptr, column.validity ? column.validity->capacity() : 0};
    if (buffer.type == DB_TYPE_INT32 && column.offsets32) return {column.offsets32->data(), column.offsets32->capacity()};
    if (buffer.type == DB_TYPE_UINT64 && column.offsets64) return {column.offsets64->data(), column.offsets64->capacity()};
    throw std::runtime_error("Text offset width differs from catalog schema");
}
DbTextViewV2 PcudaplanBackend::textParameter(size_t slot, const ParameterValue& value) {
    if (!text_uploaded_) {
        std::vector<uint8_t> bytes;
        parameter_offsets_.reserve(parameters_.size());
        for (const auto& parameter : parameters_) {
            parameter_offsets_.push_back(bytes.size());
            if (parameter.type() != ParameterType::Text || parameter.isNull()) continue;
            const auto& text = parameter.asText();
            checkedSizeAdd(bytes.size(), text.size());
            bytes.insert(bytes.end(), text.begin(), text.end());
        }
        auto& arena = context_.text_parameter_arena_;
        if (!arena) arena = std::make_unique<DynamicDeviceBuffer<uint8_t>>(context_.runtime_, 0);
        arena->ensureCapacity(std::max<size_t>(bytes.size(), 1));
        arena->copyFromHost(bytes);
        text_uploaded_ = true;
    }
    return {context_.text_parameter_arena_->data() + parameter_offsets_.at(slot), value.asText().size()};
}
void PcudaplanBackend::ensureColumns(const std::vector<uint32_t>& types,uint64_t rows) {
    if (types.size()!=context_.result_columns_.size()) throw std::runtime_error("physical result width differs from inferred schema");
    for (size_t i=0;i<types.size();++i) {
        if (logicalType(types[i])!=context_.result_columns_[i].type) throw std::runtime_error("physical result type differs from inferred schema");
    }
    context_.ensureColumnarResultCapacity(types.size(),hostSize(rows));
}
uint64_t PcudaplanBackend::prefix(void* destination,const void* source,uint32_t type,uint64_t count) {
    auto& service=context_.config_.reuse_scratch_buffers?context_.prefix_service_:temporary_prefix_;
    if(!service) service=std::make_shared<PlanPrefixService>(context_.runtime_);
    return service->run(destination,source,type,count);
}
/** @brief Same buffer spelling in another plan is not the same retained workspace owner. */
void retainWorkspace(ExecutionContext& context,const std::string& identity) {
    if (context.active_plan_identity_==identity) return;
    // Previous execution/fetch is complete. RAII also drains before release on an
    // exceptional earlier exit; no outstanding view escapes into CPU QueryResult.
    context.plan_scratch_buffers_.clear();
    context.result_column_storage_.clear();
    context.result_buffer_=std::make_unique<DynamicDeviceBuffer<unsigned long long>>(context.runtime_,0);
    context.result_validity_buffer_=std::make_unique<DynamicDeviceBuffer<uint64_t>>(context.runtime_,0);
    context.active_plan_identity_=identity;
}
} // namespace
void preparePcudaplanResult(ExecutionContext& context,size_t elements,const std::string& plan_identity) {
    if (!context.runtime_ || !context.result_buffer_ || !context.result_validity_buffer_) throw std::runtime_error("missing PCUDA runtime/result storage");
    context.runtime_->setMemoryPolicy(context.config_);
    retainWorkspace(context,plan_identity);
    context.result_buffer_->ensureCapacity(elements);
    context.ensureResultValidityCapacity(elements,false);
}
void executePcudaplan(const ExecutionPlan& plan,const ParameterValues& values,const LoadedKernels& kernels,ExecutionContext& context) {
    PcudaplanBackend backend(context,plan,values,&kernels);
    context.tuple_size_=plan.tuple_size;
    executePlan(plan,backend,values);
}
void executePcudaplan(const ExecutionPlan& plan,const ParameterValues& values,const KernelLookup& ready,ExecutionContext& context) {
    PcudaplanBackend backend(context,plan,values,nullptr,&ready);
    context.tuple_size_=plan.tuple_size;
    executePlan(plan,backend,values);
}
} // namespace db
