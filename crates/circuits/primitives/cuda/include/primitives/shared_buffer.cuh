#pragma once

#include "poseidon2.cuh"
#include "fp_array.cuh"
#include "trace_access.h"
#include <cassert>

template <typename T> struct SharedBuffer {
    T *data;
    uint32_t *idx;
    size_t capacity;

    __device__ SharedBuffer(T *data, uint32_t *idx, size_t capacity)
        : data(data), idx(idx), capacity(capacity) {}

    __device__ void push(T value) {
        uint32_t idx = atomicAdd(this->idx, 1);
        assert(idx < capacity && "SharedBuffer overflow");
        data[idx] = value;
    }
};

struct Poseidon2Buffer {
    SharedBuffer<FpArray<16>> state;

    __device__ Poseidon2Buffer(FpArray<16> *data, uint32_t *idx, size_t capacity)
        : state(data, idx, capacity) {}

    __device__ bool nonempty() const { return *state.idx > 0; }

    __device__ void receive(FpArray<16> value) { state.push(value); }

    __device__ void receive(RowSlice slice, size_t length) {
        FpArray<16> value = FpArray<16>::from_row(slice, length);
        state.push(value);
    }

    __device__ FpArray<8> compress_and_record(FpArray<8> &left, FpArray<8> &right) {
        FpArray<16> value;
        for (int i = 0; i < 8; i++) {
            value.v[i] = left.v[i];
            value.v[i + 8] = right.v[i];
        }
        state.push(value);

        poseidon2::poseidon2_mix((Fp *)&value.v[0]);

        FpArray<8> result;
        for (int i = 0; i < 8; i++) {
            result.v[i] = value.v[i];
        }
        return result;
    }

    __device__ FpArray<8> compress_and_record(RowSlice left, RowSlice right) {
        FpArray<8> left_array = FpArray<8>::from_row(left, 8);
        FpArray<8> right_array = FpArray<8>::from_row(right, 8);
        return compress_and_record(left_array, right_array);
    }

    __device__ FpArray<8> hash_and_record(FpArray<8> &left) {
        FpArray<8> zeros = FpArray<8>({0, 0, 0, 0, 0, 0, 0, 0});
        FpArray<8> result = compress_and_record(left, zeros);
        return result;
    }

    __device__ FpArray<8> hash_and_record(RowSlice left) {
        FpArray<8> zeros = FpArray<8>({0, 0, 0, 0, 0, 0, 0, 0});
        FpArray<8> result = compress_and_record(left, zeros.as_row());
        return result;
    }

    /// Compress 16 `Fp`s and record it, replacing the values with the hash.
    __device__ void compress_and_record_inplace(Fp *value_ptr) {
        FpArray<16> value;
        memcpy(value.v, value_ptr, sizeof(FpArray<16>));
        state.push(std::move(value));
        poseidon2::poseidon2_mix(value_ptr);
    }
};
