#pragma once

#include "trace_access.h"
#include <cuda/std/tuple>

template <size_t N> struct FpArray {
    uint32_t v[N];

    __device__ static FpArray from_row(RowSlice slice, size_t length = N) {
        FpArray result;
        for (int i = 0; i < length; i++) {
            result.v[i] = slice[i].asRaw();
        }
        for (int i = length; i < N; i++) {
            result.v[i] = 0;
        }
        return result;
    }
};

template <size_t N> __host__ __device__ bool operator<(const FpArray<N> &a, const FpArray<N> &b) {
    for (size_t i = 0; i < N; i++) {
        if (a.v[i] < b.v[i]) {
            return true;
        }
    }
    return false;
}

template <size_t N> __host__ __device__ bool operator==(const FpArray<N> &a, const FpArray<N> &b) {
    for (size_t i = 0; i < N; i++) {
        if (a.v[i] != b.v[i]) {
            return false;
        }
    }
    return true;
}

struct Fp16CompareOp {
    __host__ __device__ bool operator()(const FpArray<16> &a, const FpArray<16> &b) const {
        return a < b;
    }
};
