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
        carry[i] = res >> RV32_CELL_BITS;
        out[i] = static_cast<uint8_t>(res & ((1u << RV32_CELL_BITS) - 1));
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
                (static_cast<uint32_t>(1u << RV32_CELL_BITS) + static_cast<uint32_t>(x[i]) - rhs);
            out[i] = static_cast<uint8_t>(wrap);
            carry[i] = 1;
        }
    }
}

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

template <size_t NUM_LIMBS> struct BaseAluCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct BaseAluCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T opcode_add_flag;
    T opcode_sub_flag;
    T opcode_xor_flag;
    T opcode_or_flag;
    T opcode_and_flag;
};

template <size_t NUM_LIMBS> struct BaseAluCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = BaseAluCoreCols<T, NUM_LIMBS>;

    __device__ BaseAluCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, BaseAluCoreRecord<NUM_LIMBS> record) {
        uint8_t a[NUM_LIMBS];
        uint8_t carry_buf[NUM_LIMBS];

        switch (record.local_opcode) {
        case 0:
            run_add<NUM_LIMBS>(record.b, record.c, a, carry_buf);
            break;
        case 1:
            run_sub<NUM_LIMBS>(record.b, record.c, a, carry_buf);
            break;
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
                carry_buf[i] = 0;
            }
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);

        COL_WRITE_VALUE(row, Cols, opcode_add_flag, record.local_opcode == 0);
        COL_WRITE_VALUE(row, Cols, opcode_sub_flag, record.local_opcode == 1);
        COL_WRITE_VALUE(row, Cols, opcode_xor_flag, record.local_opcode == 2);
        COL_WRITE_VALUE(row, Cols, opcode_or_flag, record.local_opcode == 3);
        COL_WRITE_VALUE(row, Cols, opcode_and_flag, record.local_opcode == 4);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            if (record.local_opcode == 0 || record.local_opcode == 1) {
                bitwise_lookup.add_xor(a[i], a[i]);
            } else {
                bitwise_lookup.add_xor(record.b[i], record.c[i]);
            }
        }
    }
};