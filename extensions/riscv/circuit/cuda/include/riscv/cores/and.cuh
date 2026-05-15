#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_and(const uint8_t *x, const uint8_t *y, uint8_t *out) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        out[i] = x[i] & y[i];
    }
}

template <size_t NUM_LIMBS> struct AndCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct AndCoreCols {
    T is_valid;
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
};

template <size_t NUM_LIMBS> struct AndCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = AndCoreCols<T, NUM_LIMBS>;

    __device__ AndCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, AndCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];

        run_and<NUM_LIMBS>(record.b, record.c, a);

        COL_WRITE_VALUE(row, Cols, is_valid, true);
        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(record.b[i], record.c[i]);
        }
    }
};
