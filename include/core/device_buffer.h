#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <cstddef>
#include <utility>

namespace db {

template <typename T>
class DynamicDeviceBuffer {
public:
    DynamicDeviceBuffer(sycl::queue q, size_t initial_capacity = 0)
        : q_(q), d_data_(nullptr), capacity_(0) {
        if (initial_capacity > 0) {
            allocate(initial_capacity);
        }
    }

    ~DynamicDeviceBuffer() {
        free();
    }

    // Запрет копирования
    DynamicDeviceBuffer(const DynamicDeviceBuffer&) = delete;
    DynamicDeviceBuffer& operator=(const DynamicDeviceBuffer&) = delete;

    // Конструктор перемещения
    DynamicDeviceBuffer(DynamicDeviceBuffer&& other) noexcept
        : q_(other.q_), d_data_(other.d_data_), capacity_(other.capacity_) {
        other.d_data_ = nullptr;
        other.capacity_ = 0;
    }

    // Оператор присваивания перемещением
    DynamicDeviceBuffer& operator=(DynamicDeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            // Note: queue q_ is a copyable handle, so we assign it
            q_ = other.q_;
            d_data_ = other.d_data_;
            capacity_ = other.capacity_;

            other.d_data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    void ensureCapacity(size_t requested_capacity) {
        if (requested_capacity > capacity_) {
            free();
            size_t new_capacity = static_cast<size_t>(requested_capacity * 1.5);
            allocate(new_capacity);
        }
    }

    void zero() {
        if (capacity_ > 0 && d_data_ != nullptr) {
            q_.memset(d_data_, 0, capacity_ * sizeof(T));
        }
    }

    void copyToHost(std::vector<T>& host_vec, size_t elements_to_copy) {
        if (elements_to_copy > capacity_) {
            throw std::runtime_error("copyToHost: elements_to_copy exceeds buffer capacity");
        }
        host_vec.resize(elements_to_copy);
        if (elements_to_copy > 0 && d_data_ != nullptr) {
            q_.copy(d_data_, host_vec.data(), elements_to_copy).wait();
        }
    }

    T* data() const { return d_data_; }
    size_t capacity() const { return capacity_; }

private:
    void allocate(size_t cap) {
        d_data_ = sycl::malloc_device<T>(cap, q_);
        capacity_ = cap;
    }

    void free() {
        if (d_data_) {
            sycl::free(d_data_, q_);
            d_data_ = nullptr;
            capacity_ = 0;
        }
    }

    sycl::queue q_;
    T* d_data_;
    size_t capacity_;
};

} // namespace db
