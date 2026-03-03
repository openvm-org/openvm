#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

struct LessThanResult {
    bool cmp_result;
    size_t diff_idx;
    bool x_sign;
    bool y_sign;
};

template <size_t NUM_LIMBS>
__forceinline__ __device__ LessThanResult
run_less_than(bool is_slt, const uint8_t x[NUM_LIMBS], const uint8_t y[NUM_LIMBS]) {
    bool x_sign = ((x[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1)) == 1) && is_slt;
    bool y_sign = ((y[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1)) == 1) && is_slt;

#pragma unroll
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        if (x[i] != y[i]) {
            return {bool((x[i] < y[i]) ^ x_sign ^ y_sign), (size_t)i, x_sign, y_sign};
        }
    }
    return {false, NUM_LIMBS, x_sign, y_sign};
}

template <size_t NUM_LIMBS> struct LessThanCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct LessThanCoreCols {
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T cmp_result;

    T opcode_slt_flag;
    T opcode_sltu_flag;

    T b_msb_f;
    T c_msb_f;

    T diff_marker[NUM_LIMBS];
    T diff_val;
};

template <size_t NUM_LIMBS> struct LessThanCore {
    BitwiseOperationLookup bitwise_lookup;

    __device__ LessThanCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    template <typename T> using Cols = LessThanCoreCols<T, NUM_LIMBS>;

    __device__ void fill_trace_row(RowSlice row, LessThanCoreRecord<NUM_LIMBS> record) {
        constexpr uint8_t SLT = 0;

        bool is_slt = record.local_opcode == SLT;
        LessThanResult result = run_less_than<NUM_LIMBS>(is_slt, record.b, record.c);
        bool cmp_result = result.cmp_result;
        size_t diff_idx = result.diff_idx;
        bool b_sign = result.x_sign;
        bool c_sign = result.y_sign;

        uint8_t b_raw_msb = record.b[NUM_LIMBS - 1];
        uint8_t c_raw_msb = record.c[NUM_LIMBS - 1];

        uint32_t b_msb_f =
            b_sign ? (Fp::P - ((1u << RV32_CELL_BITS) - b_raw_msb)) : uint32_t(b_raw_msb);
        uint32_t c_msb_f =
            c_sign ? (Fp::P - ((1u << RV32_CELL_BITS) - c_raw_msb)) : uint32_t(c_raw_msb);

        uint8_t b_msb_range =
            b_sign ? uint8_t(b_raw_msb - (1u << (RV32_CELL_BITS - 1)))
                   : uint8_t(b_raw_msb + ((is_slt ? 1u : 0u) << (RV32_CELL_BITS - 1)));
        uint8_t c_msb_range =
            c_sign ? uint8_t(c_raw_msb - (1u << (RV32_CELL_BITS - 1)))
                   : uint8_t(c_raw_msb + ((is_slt ? 1u : 0u) << (RV32_CELL_BITS - 1)));

        uint32_t diff_val = 0;
        if (diff_idx == NUM_LIMBS) {
            diff_val = 0;
        } else if (diff_idx == (NUM_LIMBS - 1) && is_slt) {
            Fp fp_diff = cmp_result ? (Fp(c_msb_f) - Fp(b_msb_f)) : (Fp(b_msb_f) - Fp(c_msb_f));
            diff_val = fp_diff.asUInt32();
        } else if (cmp_result) {
            diff_val = uint32_t(record.c[diff_idx] - record.b[diff_idx]);
        } else {
            diff_val = uint32_t(record.b[diff_idx] - record.c[diff_idx]);
        }

        bitwise_lookup.add_range(b_msb_range, c_msb_range);

        uint8_t diff_marker[NUM_LIMBS] = {0};
        if (diff_idx != NUM_LIMBS) {
            bitwise_lookup.add_range(diff_val - 1, 0);
            diff_marker[diff_idx] = 1;
        }

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, diff_marker, diff_marker);

        COL_WRITE_VALUE(row, Cols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, Cols, b_msb_f, b_msb_f);
        COL_WRITE_VALUE(row, Cols, c_msb_f, c_msb_f);
        COL_WRITE_VALUE(row, Cols, diff_val, diff_val);
        COL_WRITE_VALUE(row, Cols, opcode_slt_flag, is_slt);
        COL_WRITE_VALUE(row, Cols, opcode_sltu_flag, !is_slt);
    }
};
