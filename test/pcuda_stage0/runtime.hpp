#pragma once

#include "pch.hpp"

#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace stage0 {
using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

inline void check(pcudaError_t error, const char* operation) {
    if (error != pcudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + pcudaGetErrorString(error));
}

// Never free memory or unload code after an unsuccessful teardown wait.
inline void cleanupCheck(pcudaError_t error, const char* operation) noexcept {
    if (error != pcudaSuccess) {
        std::cerr << operation << ": " << pcudaGetErrorString(error) << '\n';
        std::abort();
    }
}

inline std::size_t checkedBytes(std::size_t count, std::size_t width) {
    if (!width || count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("buffer size overflow");
    return count * width;
}

struct Stream {
    pcudaStream_t value = nullptr;
    Stream() { check(pcudaStreamCreate(&value), "create stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() {
        cleanupCheck(pcudaStreamSynchronize(value), "finish stream");
        cleanupCheck(pcudaStreamDestroy(value), "destroy stream");
    }
};

template <class T>
class Buffer {
public:
    Buffer(pcudaStream_t stream, std::size_t count, bool host = false)
        : stream_(stream), host_(host) {
        if (!count) throw std::runtime_error("zero allocation size");
        const auto bytes = checkedBytes(count, sizeof(T));
        check(host ? pcudaMallocHost(&data, bytes) : pcudaMalloc(&data, bytes), "allocate buffer");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        cleanupCheck(pcudaStreamSynchronize(stream_), "finish before free");
        cleanupCheck(host_ ? pcudaFreeHost(data) : pcudaFree(data), "free buffer");
    }
    T* data = nullptr;
private:
    pcudaStream_t stream_;
    bool host_;
};

template <class Launch>
class Module {
public:
    Module(const char* path, pcudaStream_t stream, const char* symbol) : stream_(stream) {
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) throw std::runtime_error(dlerror());
        dlerror();
        launch = reinterpret_cast<Launch>(dlsym(handle_, symbol));
        if (const char* error = dlerror()) {
            const std::string message(error);
            if (dlclose(handle_) != 0) std::abort();
            throw std::runtime_error(message);
        }
    }
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    ~Module() {
        cleanupCheck(pcudaStreamSynchronize(stream_), "finish before dlclose");
        if (dlclose(handle_) != 0) {
            std::cerr << dlerror() << '\n';
            std::abort();
        }
    }
    Launch launch = nullptr;
private:
    void* handle_ = nullptr;
    pcudaStream_t stream_;
};

inline pcudaDeviceProp selectDevice(const std::string& needle) {
    int backends = 0;
    check(pcudaGetBackendCount(&backends), "backend count");
    for (int b = 0; b < backends; ++b) {
        check(pcudaSetBackend(b), "set backend");
        int platforms = 0;
        check(pcudaGetPlatformCount(&platforms), "platform count");
        for (int p = 0; p < platforms; ++p) {
            check(pcudaSetPlatform(p), "set platform");
            int devices = 0;
            check(pcudaGetDeviceCount(&devices), "device count");
            for (int d = 0; d < devices; ++d) {
                pcudaDeviceProp prop{};
                check(pcudaGetDeviceProperties(&prop, d), "device properties");
                std::cerr << "device " << b << ':' << p << ':' << d << ' ' << prop.name << '\n';
                if (std::string(prop.name).find(needle) != std::string::npos) {
                    check(pcudaSetDeviceExt(b, p, d), "select device");
                    std::cerr << "selected " << prop.name << ", FP64=" << prop.pcudaHasFp64
                              << ", atomic64=" << prop.pcudaHasAtomic64 << '\n';
                    return prop;
                }
            }
        }
    }
    throw std::runtime_error("no device matching: " + needle);
}
} // namespace stage0
