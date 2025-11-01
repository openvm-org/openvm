#pragma once

#include <cstddef>

template <typename T, size_t N> struct PtrArray {
    T *arr[N];

    __device__ __host__ PtrArray(T **raw_ptr) {
        for (size_t i = 0; i < N; i++) {
            arr[i] = raw_ptr[i];
        }
    }

    __device__ __host__ T *operator[](size_t i) { return arr[i]; }
    __device__ __host__ const T *operator[](size_t i) const { return arr[i]; }
};
