#pragma once
/** @file @brief Build-time PCUDA service kernels, independent of SQL/JIT headers. */
#include "core/pcuda_runtime.h"
namespace db {
/** @brief Compact legacy non-null sparse tuples on the device without host row materialization.
 * @pre source/target hold physical_rows * tuple_size words; count holds one word.
 * @details The legacy presence convention is any nonzero cell. This service does
 * not redefine SQL NULL/group semantics. Only enqueue; the caller waits before
 * reading count. All inputs, target and count remain alive on runtime's stream.
 */
void compactSparseRows(DeviceRuntime& runtime, const unsigned long long* source,
                       unsigned long long* target, unsigned long long* count,
                       size_t physical_rows, size_t tuple_size);
}
