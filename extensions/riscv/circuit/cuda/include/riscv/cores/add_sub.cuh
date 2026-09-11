#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS, size_t LIMB_BITS>
__device__ __forceinline__ void run_add(
    const uint16_t *x,
    const uint16_t *y,
    uint16_t *out,
    uint32_t *carry
) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        uint32_t res = (i > 0) ? carry[i - 1] : 0;
        res += static_cast<uint32_t>(x[i]) + static_cast<uint32_t>(y[i]);
        carry[i] = res >> LIMB_BITS;
        out[i] = static_cast<uint16_t>(res & ((1u << LIMB_BITS) - 1));
    }
}

template <size_t NUM_LIMBS, size_t LIMB_BITS>
__device__ __forceinline__ void run_sub(
    const uint16_t *x,
    const uint16_t *y,
    uint16_t *out,
    uint32_t *carry
) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        uint32_t rhs = static_cast<uint32_t>(y[i]) + ((i > 0) ? carry[i - 1] : 0);
        if (static_cast<uint32_t>(x[i]) >= rhs) {
            out[i] = static_cast<uint16_t>(static_cast<uint32_t>(x[i]) - rhs);
            carry[i] = 0;
        } else {
            uint32_t wrap = (1u << LIMB_BITS) + static_cast<uint32_t>(x[i]) - rhs;
            out[i] = static_cast<uint16_t>(wrap);
            carry[i] = 1;
        }
    }
}

template <size_t NUM_LIMBS> struct AddSubCoreRecord {
    uint16_t b[NUM_LIMBS];
    uint16_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct AddSubCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T opcode_add_flag;
    T opcode_sub_flag;
};

template <size_t NUM_LIMBS, size_t LIMB_BITS, bool RANGE_CHECK_TOP_LIMB> struct AddSubCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = AddSubCoreCols<T, NUM_LIMBS>;

    __device__ AddSubCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint16_t const (&b)[NUM_LIMBS],
        uint16_t const (&c)[NUM_LIMBS],
        uint8_t local_opcode
    ) {
        uint16_t a[NUM_LIMBS];
        uint32_t carry_buf[NUM_LIMBS];

        if (local_opcode == 0) {
            run_add<NUM_LIMBS, LIMB_BITS>(b, c, a, carry_buf);
        } else {
            run_sub<NUM_LIMBS, LIMB_BITS>(b, c, a, carry_buf);
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, b);
        COL_WRITE_ARRAY(row, Cols, c, c);

        COL_WRITE_VALUE(row, Cols, opcode_add_flag, local_opcode == 0);
        COL_WRITE_VALUE(row, Cols, opcode_sub_flag, local_opcode == 1);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS - !RANGE_CHECK_TOP_LIMB; i++) {
            // The carry constraints only bound a[i] mod 2^LIMB_BITS. Every output limb not
            // constrained by the adapter must be canonicalized here.
            range_checker.add_count(a[i], LIMB_BITS);
        }
    }

    __device__ void fill_trace_row(RowSlice row, AddSubCoreRecord<NUM_LIMBS> record) {
        fill_trace_row(row, record.b, record.c, record.local_opcode);
    }
};
