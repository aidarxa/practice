#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <limits>

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
            allocate(nextCapacity(requested_capacity));
        }
    }

    size_t nextCapacity(size_t requested_capacity) const {
        // Avoid 1.5x over-allocation for large analytical projections.  On
        // scale-factor 10, SELECT * can already require many GiB exactly; a
        // high-water multiplier turns a rejectable query into a driver-level
        // allocation failure.  Keep the small-query high-water mark only where
        // it is harmless.
        constexpr size_t kSmallBufferElements = 1ULL << 20;
        if (requested_capacity == 0) return 0;
        if (requested_capacity <= kSmallBufferElements) {
            if (requested_capacity > std::numeric_limits<size_t>::max() / 3 * 2) {
                throw std::overflow_error("DynamicDeviceBuffer capacity overflow");
            }
            size_t grown = requested_capacity + requested_capacity / 2;
            return grown > requested_capacity ? grown : requested_capacity;
        }
        return requested_capacity;
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
