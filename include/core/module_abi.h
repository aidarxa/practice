#ifndef DB_MODULE_ABI_H
#define DB_MODULE_ABI_H
/** @file
 * @brief Version 1 of the C-compatible, 64-bit kernel-module contract.
 * @details A module borrows all pointers. It allocates no query memory and submits
 * exactly one kernel to the supplied stream. Scalar values are copied into device
 * launch arguments; kernels must never dereference the host argument array.
 * Changes to any field, offset or meaning require another ABI version. There is
 * deliberately no STL, GPU SDK, SQL parser or engine execution context here.
 */
#include <stddef.h>
#include <stdint.h>

#define DB_MODULE_ABI_VERSION 1u

/** @brief Backend identity; a legacy queue is never interpreted as a PCUDA stream. */
enum DbKernelBackend { DB_BACKEND_PCUDA = 1, DB_BACKEND_LEGACY_SYCL = 2 };
/** @brief Storage role, independent of the numeric type of a value. */
enum DbArgumentKind { DB_ARG_BUFFER = 1, DB_ARG_SCALAR = 2, DB_ARG_PARAMETER = 3 };
/** @brief Internal storage types; this does not extend SQL's type system. */
enum DbKernelType {
    DB_TYPE_INT32 = 1, DB_TYPE_UINT64 = 2, DB_TYPE_INT64 = 3,
    DB_TYPE_FLOAT64 = 4, DB_TYPE_UINT8 = 5, DB_TYPE_NULL_ONLY = 6,
    DB_TYPE_TAGGED_NUMERIC = 7
};
/** @brief Conservative memory effects used by the physical plan. */
enum DbArgumentAccess { DB_ACCESS_READ = 1, DB_ACCESS_WRITE = 2, DB_ACCESS_READ_WRITE = 3 };
/** @brief Nullable parameters or optional bitmap pointers; all other bits are reserved. */
enum DbArgumentFlags { DB_ARG_NULLABLE = 1 };
/** @brief Stable return codes; exceptions may not cross the C boundary. */
enum DbKernelStatus {
    DB_KERNEL_OK = 0, DB_KERNEL_ABI_ERROR = 1, DB_KERNEL_SCHEMA_ERROR = 2,
    DB_KERNEL_ARGUMENT_ERROR = 3, DB_KERNEL_LAUNCH_ERROR = 4, DB_KERNEL_BACKEND_ERROR = 5
};

/** @brief Four 32-bit fields, at offsets 0/4/8/12; sizeof is 16. */
typedef struct KernelArgumentSchemaV1 {
    uint32_t kind;
    uint32_t type;
    uint32_t access;
    uint32_t flags;
} KernelArgumentSchemaV1;

/** @brief Exact payload bits; validity and runtime type tags are separate. */
typedef union KernelScalarV1 {
    uint64_t u64;
    int64_t i64;
    double f64;
} KernelScalarV1;

/**
 * @brief A borrowed argument: schema@0, data@16, elements@24, value@32,
 * valid@40, reserved@44; sizeof is 48.
 * @details Buffers use data/element_count and valid=1; scalars/parameters use
 * value/valid and have null data/zero element_count. A null optional bitmap has
 * zero elements. NULL-only must have valid=0. Reserved fields must be zero.
 */
typedef struct KernelArgumentV1 {
    KernelArgumentSchemaV1 schema;
    void* data;
    uint64_t element_count;
    KernelScalarV1 value;
    uint32_t valid;
    uint32_t reserved;
} KernelArgumentV1;

/** @brief Blocks@0, threads/block@24, dynamic shared bytes@36; sizeof is 40. */
typedef struct KernelLaunchConfigV1 {
    uint64_t grid[3];
    uint32_t block[3];
    uint32_t dynamic_shared_bytes;
} KernelLaunchConfigV1;

/**
 * @brief Version@0, size@4, backend@8, reserved@12, stream@16, arguments@24,
 * argument_count@32, launch@40; sizeof is 80.
 * @details The stream is an explicit PCUDA stream handle. Only the temporary
 * legacy adapter uses a borrowed queue address, with LEGACY_SYCL backend identity.
 * Neither representation is a public SYCL object or an implicit default stream.
 */
typedef struct KernelInvocationV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend;
    uint32_t reserved;
    void* stream;
    const KernelArgumentV1* arguments;
    uint64_t argument_count;
    KernelLaunchConfigV1 launch;
} KernelInvocationV1;

/** @brief Immutable module-owned schema: fields@0..12, arguments@16, count@24; 32 bytes. */
typedef struct KernelModuleDescriptorV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t backend;
    uint32_t reserved;
    const KernelArgumentSchemaV1* arguments;
    uint64_t argument_count;
} KernelModuleDescriptorV1;

/** @brief Caller-owned diagnostic: size@0, status@4, index@8, message@16; 208 bytes. */
typedef struct KernelErrorV1 {
    uint32_t struct_size;
    int32_t status;
    uint64_t argument_index;
    char message[192];
} KernelErrorV1;

#ifdef __cplusplus
extern "C" {
#endif
/** @brief Return the immutable descriptor; its lifetime is that of the loaded module. */
typedef const KernelModuleDescriptorV1* (*DbKernelDescriptorV1)(void);
/** @brief Submit one kernel, returning a stable status without throwing. */
typedef int32_t (*DbKernelLaunchV1)(const KernelInvocationV1*, KernelErrorV1*);
/** @brief Fixed entry names for a single-kernel module (module generation is stage 5). */
const KernelModuleDescriptorV1* db_kernel_descriptor_v1(void);
int32_t db_kernel_launch_v1(const KernelInvocationV1*, KernelErrorV1*);
#ifdef __cplusplus
}
#define DB_ABI_ASSERT(condition) static_assert(condition, #condition)
#else
#define DB_ABI_ASSERT(condition) _Static_assert(condition, #condition)
#endif
DB_ABI_ASSERT(sizeof(void*) == 8);
DB_ABI_ASSERT(sizeof(KernelArgumentSchemaV1) == 16);
DB_ABI_ASSERT(offsetof(KernelArgumentSchemaV1, kind) == 0);
DB_ABI_ASSERT(offsetof(KernelArgumentSchemaV1, type) == 4);
DB_ABI_ASSERT(offsetof(KernelArgumentSchemaV1, access) == 8);
DB_ABI_ASSERT(offsetof(KernelArgumentSchemaV1, flags) == 12);
DB_ABI_ASSERT(sizeof(KernelScalarV1) == 8);
DB_ABI_ASSERT(sizeof(KernelArgumentV1) == 48);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, schema) == 0);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, data) == 16);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, element_count) == 24);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, value) == 32);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, valid) == 40);
DB_ABI_ASSERT(offsetof(KernelArgumentV1, reserved) == 44);
DB_ABI_ASSERT(sizeof(KernelLaunchConfigV1) == 40);
DB_ABI_ASSERT(offsetof(KernelLaunchConfigV1, grid) == 0);
DB_ABI_ASSERT(offsetof(KernelLaunchConfigV1, block) == 24);
DB_ABI_ASSERT(offsetof(KernelLaunchConfigV1, dynamic_shared_bytes) == 36);
DB_ABI_ASSERT(sizeof(KernelInvocationV1) == 80);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, abi_version) == 0);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, struct_size) == 4);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, backend) == 8);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, reserved) == 12);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, stream) == 16);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, arguments) == 24);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, argument_count) == 32);
DB_ABI_ASSERT(offsetof(KernelInvocationV1, launch) == 40);
DB_ABI_ASSERT(sizeof(KernelModuleDescriptorV1) == 32);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, abi_version) == 0);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, struct_size) == 4);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, backend) == 8);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, reserved) == 12);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, arguments) == 16);
DB_ABI_ASSERT(offsetof(KernelModuleDescriptorV1, argument_count) == 24);
DB_ABI_ASSERT(sizeof(KernelErrorV1) == 208);
DB_ABI_ASSERT(offsetof(KernelErrorV1, struct_size) == 0);
DB_ABI_ASSERT(offsetof(KernelErrorV1, status) == 4);
DB_ABI_ASSERT(offsetof(KernelErrorV1, argument_index) == 8);
DB_ABI_ASSERT(offsetof(KernelErrorV1, message) == 16);
#undef DB_ABI_ASSERT
#endif
