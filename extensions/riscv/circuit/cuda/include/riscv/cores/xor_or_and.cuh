#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_xor(const uint8_t *x, const uint8_t *y, uint8_t *out) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        out[i] = x[i] ^ y[i];
    }
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_or(const uint8_t *x, const uint8_t *y, uint8_t *out) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        out[i] = x[i] | y[i];
    }
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_and(const uint8_t *x, const uint8_t *y, uint8_t *out) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        out[i] = x[i] & y[i];
    }
}

template <size_t NUM_LIMBS> struct XorOrAndCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct XorOrAndCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T opcode_xor_flag;
    T opcode_or_flag;
    T opcode_and_flag;
};

template <size_t NUM_LIMBS> struct XorOrAndCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = XorOrAndCoreCols<T, NUM_LIMBS>;

    __device__ XorOrAndCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, XorOrAndCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];

        switch (record.local_opcode) {
        case 2:
            run_xor<NUM_LIMBS>(record.b, record.c, a);
            break;
        case 3:
            run_or<NUM_LIMBS>(record.b, record.c, a);
            break;
        case 4:
            run_and<NUM_LIMBS>(record.b, record.c, a);
            break;
        default:
#pragma unroll
            for (size_t i = 0; i < NUM_LIMBS; i++) {
                a[i] = 0;
            }
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

        COL_WRITE_VALUE(row, Cols, opcode_xor_flag, record.local_opcode == 2);
        COL_WRITE_VALUE(row, Cols, opcode_or_flag, record.local_opcode == 3);
        COL_WRITE_VALUE(row, Cols, opcode_and_flag, record.local_opcode == 4);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(record.b[i], record.c[i]);
        }
    }
};
