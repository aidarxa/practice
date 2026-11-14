/** @file @brief Compile the frozen v1 and native v2 layouts as independent C11 POD contracts. */
#include "core/module_abi_v2.h"
_Static_assert(DB_MODULE_ABI_VERSION == 1, "v1 must remain historical v1");
_Static_assert(DB_MODULE_ABI_V2_VERSION == 2, "native v2");
_Static_assert(sizeof(KernelArgumentV1) == 48, "v1 is not reinterpreted");
_Static_assert(_Alignof(KernelArgumentV2) == 8, "v2 alignment");
int main(void) {
    KernelArgumentV2 argument = {{DB_ARG_PARAMETER,DB_TYPE_TEXT,DB_ACCESS_READ,DB_ARG_NULLABLE},0,0,{{0,0}},0,0};
    return argument.value.text.bytes != 0 || argument.valid != 0;
}
