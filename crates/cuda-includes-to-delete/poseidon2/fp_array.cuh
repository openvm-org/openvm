#pragma once

#include "trace_access.h"

template <size_t N> struct FpArray {
    uint32_t v[N];

    __device__ RowSlice as_row() { return RowSlice((Fp *)&v[0], 1); }

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

    __device__ static FpArray from_raw_array(uint32_t const raw[N]) {
        FpArray result;
        for (int i = 0; i < N; i++) {
            result.v[i] = raw[i];
        }
        return result;
    }

    __device__ static FpArray from_u32_array(uint32_t const arr[N]) {
        FpArray result;
        for (int i = 0; i < N; i++) {
            result.v[i] = Fp(arr[i]).asRaw();
        }
        return result;
    }

    __device__ static FpArray from_u8_array(uint8_t const arr[N]) {
        FpArray result;
        for (int i = 0; i < N; i++) {
            result.v[i] = Fp(arr[i]).asRaw();
        }
        return result;
    }
};

template <size_t N> __host__ __device__ bool operator<(const FpArray<N> &a, const FpArray<N> &b) {
    for (size_t i = 0; i < N; i++) {
        if (a.v[i] != b.v[i]) {
            return a.v[i] < b.v[i];
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
