#ifndef DB_MODULE_ABI_V2_H
#define DB_MODULE_ABI_V2_H
/** @file
 * @brief Native SQL module ABI v2: exact numeric values and borrowed Text ranges.
 * @details V1 is included only for unchanged enum/schema/launch/error definitions;
 * its layout and entry points are not reinterpreted. A v2 argument is 56 bytes,
 * not a v1 argument (48). All pointers are borrowed, device-readable where a
 * kernel dereferences them, and remain alive through stream completion and fetch.
 */
#include "core/module_abi.h"
#define DB_MODULE_ABI_V2_VERSION 2u

enum DbKernelTextTypeV2 { DB_TYPE_TEXT = 8 };
/** @brief Device address@0 and byte count@8; empty Text is distinct from NULL. */
typedef struct DbTextViewV2 {
    const uint8_t* data;
    uint64_t bytes;
} DbTextViewV2;
typedef KernelArgumentSchemaV1 KernelArgumentSchemaV2;
typedef KernelLaunchConfigV1 KernelLaunchConfigV2;
typedef KernelErrorV1 KernelErrorV2;
/** @brief Sixteen bytes; unused upper numeric payload bits must remain zero. */
typedef union KernelScalarV2 {
    DbTextViewV2 text;
    uint64_t u64;
    int64_t i64;
    double f64;
} KernelScalarV2;
/** @brief Schema@0, data@16, count@24, value@32, valid@48, reserved@52; 56 bytes. */
typedef struct KernelArgumentV2 {
    KernelArgumentSchemaV2 schema;
    void* data;
    uint64_t element_count;
    KernelScalarV2 value;
    uint32_t valid;
    uint32_t reserved;
} KernelArgumentV2;
/** @brief V2 invocation retains the 80-byte header but points only to v2 arguments. */
typedef struct KernelInvocationV2 {
    uint32_t abi_version, struct_size, backend, reserved;
    void* stream;
    const KernelArgumentV2* arguments;
    uint64_t argument_count;
    KernelLaunchConfigV2 launch;
} KernelInvocationV2;
/** @brief Immutable module-owned schema; no memory ownership crosses the C boundary. */
typedef struct KernelModuleDescriptorV2 {
    uint32_t abi_version, struct_size, backend, reserved;
    const KernelArgumentSchemaV2* arguments;
    uint64_t argument_count;
} KernelModuleDescriptorV2;
#ifdef __cplusplus
extern "C" {
#endif
typedef const KernelModuleDescriptorV2* (*DbKernelDescriptorV2)(void);
typedef int32_t (*DbKernelLaunchV2)(const KernelInvocationV2*, KernelErrorV2*);
const KernelModuleDescriptorV2* db_kernel_descriptor_v2(void);
int32_t db_kernel_launch_v2(const KernelInvocationV2*, KernelErrorV2*);
const char* db_kernel_identity_v2(void);
#ifdef __cplusplus
}
#define DB_V2_ASSERT(x) static_assert(x, #x)
#else
#define DB_V2_ASSERT(x) _Static_assert(x, #x)
#endif
DB_V2_ASSERT(sizeof(DbTextViewV2) == 16);
DB_V2_ASSERT(offsetof(DbTextViewV2, bytes) == 8);
DB_V2_ASSERT(sizeof(KernelScalarV2) == 16);
DB_V2_ASSERT(sizeof(KernelArgumentV2) == 56);
DB_V2_ASSERT(offsetof(KernelArgumentV2, data) == 16);
DB_V2_ASSERT(offsetof(KernelArgumentV2, element_count) == 24);
DB_V2_ASSERT(offsetof(KernelArgumentV2, value) == 32);
DB_V2_ASSERT(offsetof(KernelArgumentV2, valid) == 48);
DB_V2_ASSERT(offsetof(KernelArgumentV2, reserved) == 52);
DB_V2_ASSERT(sizeof(KernelInvocationV2) == 80);
DB_V2_ASSERT(offsetof(KernelInvocationV2, arguments) == 24);
DB_V2_ASSERT(offsetof(KernelInvocationV2, argument_count) == 32);
DB_V2_ASSERT(offsetof(KernelInvocationV2, launch) == 40);
DB_V2_ASSERT(sizeof(KernelModuleDescriptorV2) == 32);
DB_V2_ASSERT(offsetof(KernelModuleDescriptorV2, arguments) == 16);
DB_V2_ASSERT(offsetof(KernelModuleDescriptorV2, argument_count) == 24);
#undef DB_V2_ASSERT
#endif
