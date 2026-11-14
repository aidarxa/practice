/** @file
 * @brief Behavioral checks of the versioned module boundary, including exact values.
 */
#include "core/module_contract.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <type_traits>

int main() {
    static_assert(std::is_standard_layout<KernelInvocationV1>::value, "C layout");
    static_assert(std::is_trivial<KernelArgumentV1>::value, "POD arguments");
    KernelArgumentSchemaV1 schema{DB_ARG_PARAMETER, DB_TYPE_INT64, DB_ACCESS_READ, DB_ARG_NULLABLE};
    KernelModuleDescriptorV1 module{1, sizeof(KernelModuleDescriptorV1), DB_BACKEND_PCUDA, 0, &schema, 1};
    KernelArgumentV1 argument{schema, nullptr, 0, {}, 1, 0};
    argument.value.i64 = INT64_C(9007199254740993);
    int stream = 0;
    KernelInvocationV1 invocation{1, sizeof(KernelInvocationV1), DB_BACKEND_PCUDA, 0,
                                  &stream, &argument, 1, {{1, 1, 1}, {1, 1, 1}, 0}};
    KernelErrorV1 error{sizeof(KernelErrorV1), 0, 0, {}};
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    assert(argument.value.i64 == INT64_C(9007199254740993));
    argument.valid = 0;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    invocation.backend = module.backend = 123;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ABI_ERROR);
    invocation.backend = module.backend = DB_BACKEND_PCUDA;
    invocation.launch.grid[0] = UINT64_MAX;
    invocation.launch.grid[1] = 2;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_LAUNCH_ERROR);
    invocation.launch = {{1,1,1},{1,1,1},0};
    schema = {DB_ARG_BUFFER, DB_TYPE_INT32, DB_ACCESS_READ, 0};
    argument = {schema, nullptr, 0, {}, 1, 0};
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    int32_t data[2] = {};
    argument.data = data;
    argument.element_count = 2;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    argument.element_count = UINT64_MAX;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    argument.element_count = 2;
    argument.data = reinterpret_cast<uint8_t*>(data) + 1;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    schema.flags = DB_ARG_NULLABLE;
    argument = {schema, nullptr, 0, {}, 1, 0};
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    argument.reserved = 1;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    argument.reserved = 0;
    module.struct_size = 0;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ABI_ERROR);
    module.struct_size = sizeof(module);
    invocation.argument_count = 0;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_SCHEMA_ERROR);
    invocation.argument_count = 1;
    invocation.stream = nullptr;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_LAUNCH_ERROR);
    assert(db::kernelTypeBytes(DB_TYPE_TAGGED_NUMERIC) == 0);
    assert(db::kernelError(nullptr, 42, 0, "ignored") == 42);
    KernelErrorV1 wrong_size{};
    assert(db::kernelError(&wrong_size, 42, 0, "ignored") == 42);
    schema = {DB_ARG_PARAMETER, DB_TYPE_INT64, DB_ACCESS_READ, DB_ARG_NULLABLE};
    argument = {schema, nullptr, 0, {}, 1, 0};
    invocation.stream = &stream;
    argument.schema.type = DB_TYPE_FLOAT64;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_SCHEMA_ERROR);
    assert(error.argument_index == 0 && std::strlen(error.message) != 0);
    argument.schema = schema;
    invocation.abi_version = 2;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ABI_ERROR);
    invocation.abi_version = 1;
    invocation.backend = DB_BACKEND_LEGACY_SYCL;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ABI_ERROR);
    invocation.backend = DB_BACKEND_PCUDA;
    invocation.arguments = nullptr;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    invocation.arguments = &argument;
    invocation.launch.block[0] = 0;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_LAUNCH_ERROR);
    invocation.launch.block[0] = 1;
    schema.type = DB_TYPE_TAGGED_NUMERIC;
    argument.schema.type = DB_TYPE_FLOAT64;
    argument.valid = 1;
    argument.value.f64 = 1.25;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    schema.type = DB_TYPE_NULL_ONLY;
    argument.schema.type = DB_TYPE_NULL_ONLY;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_ARGUMENT_ERROR);
    argument.valid = 0;
    assert(db::validateInvocation(module, invocation, &error) == DB_KERNEL_OK);
    std::cout << "module ABI checks passed\n";
}
