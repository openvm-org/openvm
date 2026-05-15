#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_add(
    const uint8_t *x,
    const uint8_t *y,
    uint8_t *out,
    uint8_t *carry
) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        uint32_t res = (i > 0) ? carry[i - 1] : 0;
        res += static_cast<uint32_t>(x[i]) + static_cast<uint32_t>(y[i]);
        carry[i] = res >> RV64_CELL_BITS;
        out[i] = static_cast<uint8_t>(res & ((1u << RV64_CELL_BITS) - 1));
    }
}

template <size_t NUM_LIMBS> struct AddCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct AddCoreCols {
    T is_valid;
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
};

template <size_t NUM_LIMBS> struct AddCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = AddCoreCols<T, NUM_LIMBS>;

    __device__ AddCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, AddCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];
        uint8_t carry_buf[NUM_LIMBS];

        run_add<NUM_LIMBS>(record.b, record.c, a, carry_buf);

        COL_WRITE_VALUE(row, Cols, is_valid, true);
        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(a[i], a[i]);
        }
    }
};
