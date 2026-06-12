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
        carry[i] = res >> RV64_BYTE_BITS;
        out[i] = static_cast<uint8_t>(res & ((1u << RV64_BYTE_BITS) - 1));
    }
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ void run_sub(
    const uint8_t *x,
    const uint8_t *y,
    uint8_t *out,
    uint8_t *carry
) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        uint32_t rhs = static_cast<uint32_t>(y[i]) + ((i > 0) ? carry[i - 1] : 0);
        if (static_cast<uint32_t>(x[i]) >= rhs) {
            out[i] = static_cast<uint8_t>(static_cast<uint32_t>(x[i]) - rhs);
            carry[i] = 0;
        } else {
            uint32_t wrap =
                (static_cast<uint32_t>(1u << RV64_BYTE_BITS) + static_cast<uint32_t>(x[i]) - rhs);
            out[i] = static_cast<uint8_t>(wrap);
            carry[i] = 1;
        }
    }
}

template <size_t NUM_LIMBS> struct AddSubCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct AddSubCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T opcode_add_flag;
    T opcode_sub_flag;
};

template <size_t NUM_LIMBS> struct AddSubCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = AddSubCoreCols<T, NUM_LIMBS>;

    __device__ AddSubCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, AddSubCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];
        uint8_t carry_buf[NUM_LIMBS];

        if (record.local_opcode == 0) {
            run_add<NUM_LIMBS>(record.b, record.c, a, carry_buf);
        } else {
            run_sub<NUM_LIMBS>(record.b, record.c, a, carry_buf);
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

        COL_WRITE_VALUE(row, Cols, opcode_add_flag, record.local_opcode == 0);
        COL_WRITE_VALUE(row, Cols, opcode_sub_flag, record.local_opcode == 1);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(a[i], a[i]);
            // Memory bus checks only packed u16 values; read bytes need separate bounds.
            bitwise_lookup.add_range(record.b[i], record.c[i]);
        }
    }
};
