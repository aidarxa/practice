#pragma once
/** @file @brief Fail-closed native v2 validation; Text is never a tagged numeric. */
#include "core/module_abi_v2.h"
#include "core/module_contract.h"
#include <climits>
#include <cstring>

namespace db { namespace abi2 {
using db::kernelError;
using db::numericParameterType;
using db::compatibleArgumentSchema;
using db::validLaunchConfig;
/** @brief V2 adds a 16-byte view array with 8-byte alignment, not character storage. */
inline uint64_t kernelTypeBytes(uint32_t type) noexcept {
    return type == DB_TYPE_TEXT ? 16 : db::kernelTypeBytes(type);
}
/** @brief Text parameters carry a length-delimited device range; validity is separate. */
inline bool validTextScalar(const KernelArgumentV2& value) noexcept {
    const auto text = value.value.text;
    if (text.bytes > INT32_MAX) return false;
    if (!value.valid) return !text.data && !text.bytes;
    return !text.bytes || text.data;
}
/** @brief Reject unsupported schema bits and scalar/buffer role confusion. */
inline bool validSchema(const KernelArgumentSchemaV2& schema) noexcept {
    if (schema.type != DB_TYPE_TEXT) return db::validArgumentSchema(schema);
    if (schema.kind != DB_ARG_BUFFER && schema.kind != DB_ARG_PARAMETER) return false;
    if (schema.access < DB_ACCESS_READ || schema.access > DB_ACCESS_READ_WRITE) return false;
    if (schema.kind == DB_ARG_PARAMETER && schema.access != DB_ACCESS_READ) return false;
    return !(schema.flags & ~static_cast<uint32_t>(DB_ARG_NULLABLE));
}
inline bool validBuffer(const KernelArgumentV2& value) noexcept {
    const auto bytes = kernelTypeBytes(value.schema.type);
    if (!bytes || value.valid != 1 || value.element_count > SIZE_MAX / bytes) return false;
    if (!value.data) return (value.schema.flags & DB_ARG_NULLABLE) && !value.element_count;
    const auto alignment = value.schema.type == DB_TYPE_TEXT ? 8 : bytes;
    return reinterpret_cast<uintptr_t>(value.data) % alignment == 0;
}
inline bool validScalar(const KernelArgumentV2& value) noexcept {
    if (value.data || value.element_count || value.valid > 1) return false;
    if (!value.valid && !(value.schema.flags & DB_ARG_NULLABLE)) return false;
    if (value.schema.type == DB_TYPE_TEXT) return validTextScalar(value);
    uint64_t upper = 0;
    std::memcpy(&upper, reinterpret_cast<const uint8_t*>(&value.value) + 8, sizeof(upper));
    if (upper) return false;
    if (value.schema.type == DB_TYPE_NULL_ONLY) return !value.valid;
    return db::kernelTypeBytes(value.schema.type) != 0;
}
inline int32_t validateArgument(const KernelArgumentSchemaV2& schema, const KernelArgumentV2& value) noexcept {
    if (!validSchema(schema) || !compatibleArgumentSchema(schema, value.schema)) return DB_KERNEL_SCHEMA_ERROR;
    if (value.reserved) return DB_KERNEL_ARGUMENT_ERROR;
    return (value.schema.kind == DB_ARG_BUFFER ? validBuffer(value) : validScalar(value)) ? DB_KERNEL_OK : DB_KERNEL_ARGUMENT_ERROR;
}
/** @brief Validate header before accessing an array; v1 and legacy backends fail closed. */
inline int32_t validateHeader(const KernelModuleDescriptorV2& descriptor,
                              const KernelInvocationV2& call, KernelErrorV2* error) noexcept {
    if (descriptor.abi_version != DB_MODULE_ABI_V2_VERSION || call.abi_version != DB_MODULE_ABI_V2_VERSION)
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "native ABI v2 required");
    if (descriptor.struct_size != sizeof(descriptor) || call.struct_size != sizeof(call))
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "invalid v2 layout");
    if (descriptor.backend != DB_BACKEND_PCUDA || call.backend != DB_BACKEND_PCUDA || descriptor.reserved || call.reserved)
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "invalid native backend/reserved fields");
    return DB_KERNEL_OK;
}
/** @brief Check borrowed array shape without dereferencing it. */
inline int32_t validateArrays(const KernelModuleDescriptorV2& descriptor,
                              const KernelInvocationV2& call, KernelErrorV2* error) noexcept {
    if (descriptor.argument_count != call.argument_count)
        return kernelError(error, DB_KERNEL_SCHEMA_ERROR, 0, "v2 argument count mismatch");
    if (call.argument_count && (!descriptor.arguments || !call.arguments))
        return kernelError(error, DB_KERNEL_ARGUMENT_ERROR, 0, "null v2 argument arrays");
    return DB_KERNEL_OK;
}
/** @brief Validate all structural contracts before a wrapper submits device work. */
inline int32_t validateInvocation(const KernelModuleDescriptorV2& descriptor,
                                  const KernelInvocationV2& call, KernelErrorV2* error) noexcept {
    const auto header = validateHeader(descriptor,call,error);
    if (header) return header;
    const auto arrays = validateArrays(descriptor,call,error);
    if (arrays) return arrays;
    if (!call.stream || !validLaunchConfig(call.launch))
        return kernelError(error, DB_KERNEL_LAUNCH_ERROR, 0, "invalid explicit stream/launch");
    for (uint64_t i = 0; i < call.argument_count; ++i) {
        const auto status = validateArgument(descriptor.arguments[i], call.arguments[i]);
        if (status) return kernelError(error, status, i, "invalid v2 argument");
    }
    return kernelError(error, DB_KERNEL_OK, 0, "");
}
}} // namespace db::abi2
