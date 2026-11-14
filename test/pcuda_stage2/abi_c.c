/** @file
 * @brief Compile the public ABI as C11 and exercise a C function-pointer round trip.
 */
#include "core/module_abi.h"

static int32_t exact_integer(const KernelInvocationV1* call, KernelErrorV1* error) {
    (void)error;
    return call->arguments[0].value.i64 == INT64_C(9007199254740993) ? 0 : 1;
}
int main(void) {
    KernelArgumentV1 argument = {{DB_ARG_PARAMETER, DB_TYPE_INT64, DB_ACCESS_READ, DB_ARG_NULLABLE}, 0, 0, {0}, 1, 0};
    argument.value.i64 = INT64_C(9007199254740993);
    KernelInvocationV1 call = {1, sizeof(KernelInvocationV1), DB_BACKEND_PCUDA, 0, &argument, &argument, 1, {{1,1,1},{1,1,1},0}};
    DbKernelLaunchV1 launch = exact_integer;
    return launch(&call, 0);
}
