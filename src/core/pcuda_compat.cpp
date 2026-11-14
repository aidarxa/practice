/** @file
 * @brief Check completion that the installed public PCUDA wrapper fails to check.
 * @details AdaptiveCpp 5989e1c discards inorder_queue::wait()'s result and its
 * pcudaGetLastError omits non-PCUDA errors. Keep version-sensitive headers here;
 * never include the SYCL frontend. Error storage in this version is process-wide:
 * a failure is conservatively reported, never attributed to an unrelated stream.
 */
#include "pcuda_compat.h"
#include <hipSYCL/runtime/pcuda/pcuda_runtime.hpp>
#include <hipSYCL/runtime/pcuda/pcuda_stream.hpp>
#include <hipSYCL/runtime/hardware.hpp>
#include <hipSYCL/runtime/inorder_queue.hpp>
#include <hipSYCL/runtime/async_errors.hpp>
#include <stdexcept>

namespace db::detail {
bool pcudaDeviceIsGpu(int backend, int platform, int device) {
    const auto* entry = hipsycl::rt::pcuda::pcuda_application::get().pcuda_rt()
        .get_topology().get_device(backend, platform, device);
    return entry && entry->dev->is_gpu();
}
std::string takePcudaruntimeErrors() {
    std::string message;
    hipsycl::rt::application::errors().pop_each_error([&](const hipsycl::rt::result& error) {
        if (!error.is_success()) message += error.what() + "\n";
    });
    return message;
}
void waitPcudastream(pcudaStream_t stream) {
    if (!stream) throw std::runtime_error("cannot wait on a null/default PCUDA stream");
    auto message = takePcudaruntimeErrors();
    const auto result = hipsycl::rt::pcuda::stream::get_queue(stream)->wait();
    if (!result.is_success()) message += result.what() + "\n";
    message += takePcudaruntimeErrors();
    if (!message.empty()) throw std::runtime_error("PCUDA stream completion: " + message);
}
}
