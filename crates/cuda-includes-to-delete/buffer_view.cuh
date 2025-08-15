#pragma once

#include <cstdint>
#include <cassert>

template <typename T>
struct DeviceBufferConstView {
    T const* ptr;
    size_t size;

    __device__ __host__ __forceinline__ T const* begin() const {
        return ptr;
    }

    __device__ __host__ __forceinline__ T const* end() const {
        return ptr + len();
    }

    __device__ __host__ __forceinline__ T const& operator [](size_t idx) const {
        assert(idx < len());
        return ptr[idx];
    }

    __device__ __host__ __forceinline__ size_t len() const {
        return size / sizeof(T);
    }
};

struct DeviceRawBufferConstView {
    uintptr_t ptr;
    size_t size;

    template <typename T> __device__ __host__ DeviceBufferConstView<T> as_typed() const {
        assert(size % sizeof(T) == 0);
        return {
            reinterpret_cast<T const*>(ptr),
            size
        };
    }
};
