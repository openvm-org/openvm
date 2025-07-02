#pragma once

#include "poseidon2/columns.cuh"
#include "poseidon2/fp_array.cuh"
#include "poseidon2/tracegen.cuh"
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

    template <size_t SBOX_DEGREE, size_t SBOX_REGS, size_t HALF_FULL_ROUNDS, size_t PARTIAL_ROUNDS>
    __device__ FpArray<8> compress_and_record(FpArray<8> &left, FpArray<8> &right) {
        FpArray<16> value;
        for (int i = 0; i < 8; i++) {
            value.v[i] = left.v[i];
            value.v[i + 8] = right.v[i];
        }
        state.push(value);

        poseidon2::generate_trace_row_for_perm(
            poseidon2::Poseidon2Row<16, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::
                null(),
            value.as_row()
        );

        FpArray<8> result;
        for (int i = 0; i < 8; i++) {
            result.v[i] = value.v[i];
        }
        return result;
    }

    template <size_t SBOX_DEGREE, size_t SBOX_REGS, size_t HALF_FULL_ROUNDS, size_t PARTIAL_ROUNDS>
    __device__ FpArray<8> compress_and_record(RowSlice left, RowSlice right) {
        FpArray<8> left_array = FpArray<8>::from_row(left, 8);
        FpArray<8> right_array = FpArray<8>::from_row(right, 8);
        return compress_and_record<SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            left_array, right_array
        );
    }

    template <size_t SBOX_DEGREE, size_t SBOX_REGS, size_t HALF_FULL_ROUNDS, size_t PARTIAL_ROUNDS>
    __device__ FpArray<8> hash_and_record(FpArray<8> &left) {
        FpArray<8> zeros = FpArray<8>({0, 0, 0, 0, 0, 0, 0, 0});
        FpArray<8> result =
            compress_and_record<SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
                left, zeros
            );
        return result;
    }

    template <size_t SBOX_DEGREE, size_t SBOX_REGS, size_t HALF_FULL_ROUNDS, size_t PARTIAL_ROUNDS>
    __device__ FpArray<8> hash_and_record(RowSlice left) {
        FpArray<8> zeros = FpArray<8>({0, 0, 0, 0, 0, 0, 0, 0});
        FpArray<8> result =
            compress_and_record<SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
                left, zeros.as_row()
            );
        return result;
    }
};
