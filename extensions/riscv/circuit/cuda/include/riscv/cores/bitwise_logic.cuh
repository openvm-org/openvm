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

template <size_t NUM_LIMBS> struct BitwiseLogicCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct BitwiseLogicCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T opcode_xor_flag;
    T opcode_or_flag;
    T opcode_and_flag;
};

template <size_t NUM_LIMBS> struct BitwiseLogicCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = BitwiseLogicCoreCols<T, NUM_LIMBS>;

    __device__ BitwiseLogicCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint8_t const (&b)[NUM_LIMBS],
        uint8_t const (&c)[NUM_LIMBS],
        uint8_t local_opcode
    ) {
        uint8_t a[NUM_LIMBS];

        switch (local_opcode) {
        case 2:
            run_xor<NUM_LIMBS>(b, c, a);
            break;
        case 3:
            run_or<NUM_LIMBS>(b, c, a);
            break;
        case 4:
            run_and<NUM_LIMBS>(b, c, a);
            break;
        default:
#pragma unroll
            for (size_t i = 0; i < NUM_LIMBS; i++) {
                a[i] = 0;
            }
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, b);
        COL_WRITE_ARRAY(row, Cols, c, c);

        COL_WRITE_VALUE(row, Cols, opcode_xor_flag, local_opcode == 2);
        COL_WRITE_VALUE(row, Cols, opcode_or_flag, local_opcode == 3);
        COL_WRITE_VALUE(row, Cols, opcode_and_flag, local_opcode == 4);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(b[i], c[i]);
        }
    }

    __device__ void fill_trace_row(RowSlice row, BitwiseLogicCoreRecord<NUM_LIMBS> record) {
        fill_trace_row(row, record.b, record.c, record.local_opcode);
    }
};
