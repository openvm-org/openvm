#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS> struct MultiplicationCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
};

template <typename T, size_t NUM_LIMBS> struct MultiplicationCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T is_valid;
};

template <size_t NUM_LIMBS>
__forceinline__ __device__ void run_mul(
    const uint8_t *x,
    const uint8_t *y,
    uint8_t *out_a,
    uint32_t *carry
) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        uint32_t res = (i > 0) ? carry[i - 1] : 0;
#pragma unroll
        for (size_t j = 0; j <= i; j++) {
            res += static_cast<uint32_t>(x[j]) * static_cast<uint32_t>(y[i - j]);
        }
        carry[i] = res >> RV32_CELL_BITS;
        out_a[i] = static_cast<uint8_t>(res & ((1u << RV32_CELL_BITS) - 1));
    }
}

template <size_t NUM_LIMBS> struct MultiplicationCore {
    RangeTupleChecker<2> range_tuple_checker;

    template <typename T> using Cols = MultiplicationCoreCols<T, NUM_LIMBS>;

    __device__ MultiplicationCore(RangeTupleChecker<2> range_tuple_checker)
        : range_tuple_checker(range_tuple_checker) {}

    __device__ void fill_trace_row(RowSlice row, MultiplicationCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];
        uint32_t carry_buf[NUM_LIMBS];
        run_mul<NUM_LIMBS>(record.b, record.c, a, carry_buf);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            uint32_t vals[2] = {static_cast<uint32_t>(a[i]), carry_buf[i]};
            range_tuple_checker.add_count(vals);
        }

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_VALUE(row, Cols, is_valid, 1);
    }
};