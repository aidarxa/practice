#pragma once
/** @file @brief Narrow installed-AdaptiveCpp compatibility seam, not a second backend. */
#include <AdaptiveCpp/pcuda/pcuda_runtime.hpp>
#include <string>
namespace db::detail {
bool pcudaDeviceIsGpu(int backend, int platform, int device);
std::string takePcudaruntimeErrors();
void waitPcudastream(pcudaStream_t stream);
}
