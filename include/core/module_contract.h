#pragma once
/** @file
 * @brief Header-only validation shared by the host loader and module wrappers.
 */
#include "core/module_abi.h"
#include <limits>

namespace db {
/** @brief Populate a correctly sized diagnostic without allocating or throwing. */
inline int32_t kernelError(KernelErrorV1* error, int32_t status,
                           uint64_t argument, const char* message) noexcept {
    if (!error || error->struct_size != sizeof(KernelErrorV1)) return status;
    error->status = status;
    error->argument_index = argument;
    size_t i = 0;
    for (; i + 1 < sizeof(error->message) && message[i]; ++i) error->message[i] = message[i];
    error->message[i] = '\0';
    return status;
}

/** @brief Return internal buffer element size, or zero for a non-storage type. */
inline uint64_t kernelTypeBytes(uint32_t type) noexcept {
    if (type == DB_TYPE_UINT8) return 1;
    if (type == DB_TYPE_INT32) return 4;
    if (type >= DB_TYPE_UINT64 && type <= DB_TYPE_FLOAT64) return 8;
    return 0;
}

/** @brief Tagged fallback accepts only agreed numeric/NULL types, not arbitrary storage. */
inline bool numericParameterType(uint32_t type) noexcept {
    return type == DB_TYPE_INT64 || type == DB_TYPE_FLOAT64 || type == DB_TYPE_NULL_ONLY;
}

/** @brief Unknown backend tags cannot become compatible merely by matching each other. */
inline bool knownKernelBackend(uint32_t backend) noexcept {
    return backend == DB_BACKEND_PCUDA || backend == DB_BACKEND_LEGACY_SYCL;
}

/** @brief Compare a descriptor slot with an invocation slot, excluding validity. */
inline bool compatibleArgumentSchema(const KernelArgumentSchemaV1& expected,
                                     const KernelArgumentSchemaV1& actual) noexcept {
    if (expected.kind != actual.kind || expected.access != actual.access || expected.flags != actual.flags) return false;
    if (expected.type == actual.type) return true;
    return expected.kind == DB_ARG_PARAMETER && expected.type == DB_TYPE_TAGGED_NUMERIC && numericParameterType(actual.type);
}

/** @brief Validate header identity before accessing the borrowed argument arrays. */
inline int32_t validateInvocationHeader(const KernelModuleDescriptorV1& module,
                                        const KernelInvocationV1& call, KernelErrorV1* error) noexcept {
    if (module.abi_version != DB_MODULE_ABI_VERSION || call.abi_version != DB_MODULE_ABI_VERSION)
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "unsupported module ABI version");
    if (module.struct_size != sizeof(module) || call.struct_size != sizeof(call))
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "incompatible module ABI layout");
    if (!knownKernelBackend(module.backend) || module.backend != call.backend || module.reserved || call.reserved)
        return kernelError(error, DB_KERNEL_ABI_ERROR, 0, "incompatible backend or reserved ABI fields");
    return DB_KERNEL_OK;
}

/** @brief Validate buffer extent before pointer arithmetic, allocation or launch. */
inline bool validBufferArgument(const KernelArgumentV1& argument) noexcept {
    const auto bytes = kernelTypeBytes(argument.schema.type);
    if (!bytes || argument.valid != 1) return false;
    if (argument.element_count > std::numeric_limits<size_t>::max() / bytes) return false;
    if (argument.data) return reinterpret_cast<uintptr_t>(argument.data) % bytes == 0;
    return (argument.schema.flags & DB_ARG_NULLABLE) && argument.element_count == 0;
}

/** @brief Scalars cannot carry a buffer; NULL-only can never be a valid number. */
inline bool validScalarArgument(const KernelArgumentV1& argument) noexcept {
    if (argument.data || argument.element_count || argument.valid > 1) return false;
    if (argument.valid == 0 && !(argument.schema.flags & DB_ARG_NULLABLE)) return false;
    if (argument.schema.type == DB_TYPE_NULL_ONLY) return argument.valid == 0;
    return kernelTypeBytes(argument.schema.type) != 0;
}

/** @brief Validate nonempty launch dimensions and multiplication before backend conversion. */
inline bool validLaunchConfig(const KernelLaunchConfigV1& launch) noexcept {
    size_t volume = 1;
    for (unsigned i = 0; i < 3; ++i) {
        if (!launch.grid[i] || !launch.block[i]) return false;
        if (launch.grid[i] > std::numeric_limits<size_t>::max() / launch.block[i]) return false;
        const auto dimension = static_cast<size_t>(launch.grid[i] * launch.block[i]);
        if (volume > std::numeric_limits<size_t>::max() / dimension) return false;
        volume *= dimension;
    }
    return true;
}

/** @brief Validate array shape before any argument dereference. */
inline int32_t validateInvocationArrays(const KernelModuleDescriptorV1& module,
                                        const KernelInvocationV1& call, KernelErrorV1* error) noexcept {
    if (module.argument_count != call.argument_count)
        return kernelError(error, DB_KERNEL_SCHEMA_ERROR, 0, "argument count differs from module schema");
    if (call.argument_count && (!module.arguments || !call.arguments))
        return kernelError(error, DB_KERNEL_ARGUMENT_ERROR, 0, "null argument/schema array");
    return DB_KERNEL_OK;
}

/** @brief Reject invalid public kinds/access/flag bits instead of trusting a matching corrupt schema. */
inline bool validArgumentSchema(const KernelArgumentSchemaV1& schema) noexcept {
    if (schema.kind < DB_ARG_BUFFER || schema.kind > DB_ARG_PARAMETER) return false;
    if (schema.access < DB_ACCESS_READ || schema.access > DB_ACCESS_READ_WRITE) return false;
    if (schema.flags & ~static_cast<uint32_t>(DB_ARG_NULLABLE)) return false;
    if (schema.kind == DB_ARG_PARAMETER) return numericParameterType(schema.type) || schema.type == DB_TYPE_TAGGED_NUMERIC;
    return kernelTypeBytes(schema.type) != 0;
}

/** @brief Validate one slot; the caller adds its exact slot index to diagnostics. */
inline int32_t validateArgument(const KernelArgumentSchemaV1& schema, const KernelArgumentV1& argument) noexcept {
    if (!validArgumentSchema(schema) || !compatibleArgumentSchema(schema, argument.schema)) return DB_KERNEL_SCHEMA_ERROR;
    if (argument.reserved) return DB_KERNEL_ARGUMENT_ERROR;
    const bool valid = argument.schema.kind == DB_ARG_BUFFER ? validBufferArgument(argument) : validScalarArgument(argument);
    return valid ? DB_KERNEL_OK : DB_KERNEL_ARGUMENT_ERROR;
}

/** @brief Validate a complete invocation without submitting work or changing caller values. */
inline int32_t validateInvocation(const KernelModuleDescriptorV1& module,
                                  const KernelInvocationV1& call, KernelErrorV1* error) noexcept {
    const auto header = validateInvocationHeader(module, call, error);
    if (header != DB_KERNEL_OK) return header;
    const auto arrays = validateInvocationArrays(module, call, error);
    if (arrays != DB_KERNEL_OK) return arrays;
    if (!call.stream || !validLaunchConfig(call.launch))
        return kernelError(error, DB_KERNEL_LAUNCH_ERROR, 0, "invalid explicit stream or launch dimensions");
    for (uint64_t i = 0; i < call.argument_count; ++i) {
        const auto status = validateArgument(module.arguments[i], call.arguments[i]);
        if (status != DB_KERNEL_OK) return kernelError(error, status, i, "incompatible or invalid kernel argument");
    }
    return kernelError(error, DB_KERNEL_OK, 0, "");
}
} // namespace db
