#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS> struct BranchLessThanCoreRecord;
template <typename T, size_t NUM_LIMBS> struct BranchLessThanCoreCols;
template <size_t NUM_LIMBS> struct BranchLessThanCore;
template <size_t NUM_LIMBS> struct BranchLessThanCoreRecord {
    uint8_t a[NUM_LIMBS];
    uint8_t b[NUM_LIMBS];
    uint32_t imm;
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct BranchLessThanCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];

    T cmp_result;
    T imm;

    T opcode_blt_flag;
    T opcode_bltu_flag;
    T opcode_bge_flag;
    T opcode_bgeu_flag;

    T a_msb_f;
    T b_msb_f;

    T cmp_lt;

    T diff_marker[NUM_LIMBS];
    T diff_val;
};

template <size_t NUM_LIMBS> struct BranchLessThanCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = BranchLessThanCoreCols<T, NUM_LIMBS>;

    __device__ BranchLessThanCore(BitwiseOperationLookup bitwise_lookup)
        : bitwise_lookup(bitwise_lookup) {}

    __device__ void fill_trace_row(RowSlice row, BranchLessThanCoreRecord<NUM_LIMBS> record) {
        static constexpr uint8_t BLT = 0;
        static constexpr uint8_t BLTU = 1;
        static constexpr uint8_t BGE = 2;
        static constexpr uint8_t BGEU = 3;

        int diff_idx = NUM_LIMBS;

        bool signed_op = (record.local_opcode == BLT) || (record.local_opcode == BGE);
        bool ge_op = (record.local_opcode == BGE) || (record.local_opcode == BGEU);

        for (int i = NUM_LIMBS - 1; i >= 0; i--) {
            if (record.a[i] != record.b[i]) {
                diff_idx = i;
                break;
            }
        }

        bool cmp_lt;
        if (diff_idx == NUM_LIMBS) {
            cmp_lt = false;
        } else if (signed_op && (diff_idx == NUM_LIMBS - 1)) {
            bool a_sign = record.a[NUM_LIMBS - 1] >= (1u << (RV32_CELL_BITS - 1));
            bool b_sign = record.b[NUM_LIMBS - 1] >= (1u << (RV32_CELL_BITS - 1));

            if (a_sign != b_sign) {
                cmp_lt = a_sign;
            } else {
                cmp_lt = record.a[diff_idx] < record.b[diff_idx];
            }
        } else {
            cmp_lt = record.a[diff_idx] < record.b[diff_idx];
        }

        bool cmp_result = ge_op ? !cmp_lt : cmp_lt;

        bool a_sign = signed_op && (record.a[NUM_LIMBS - 1] >= (1u << (RV32_CELL_BITS - 1)));
        bool b_sign = signed_op && (record.b[NUM_LIMBS - 1] >= (1u << (RV32_CELL_BITS - 1)));

        uint32_t a_msb_f = a_sign ? (Fp::P - ((1u << RV32_CELL_BITS) - record.a[NUM_LIMBS - 1]))
                                  : uint32_t(record.a[NUM_LIMBS - 1]);
        uint32_t b_msb_f = b_sign ? (Fp::P - ((1u << RV32_CELL_BITS) - record.b[NUM_LIMBS - 1]))
                                  : uint32_t(record.b[NUM_LIMBS - 1]);

        uint8_t a_msb_range =
            a_sign ? uint8_t(record.a[NUM_LIMBS - 1] - (1u << (RV32_CELL_BITS - 1)))
                   : uint8_t(
                         record.a[NUM_LIMBS - 1] + ((signed_op ? 1u : 0u) << (RV32_CELL_BITS - 1))
                     );
        uint8_t b_msb_range =
            b_sign ? uint8_t(record.b[NUM_LIMBS - 1] - (1u << (RV32_CELL_BITS - 1)))
                   : uint8_t(
                         record.b[NUM_LIMBS - 1] + ((signed_op ? 1u : 0u) << (RV32_CELL_BITS - 1))
                     );

        uint32_t diff_val = 0;
        if (diff_idx == NUM_LIMBS) {
            diff_val = 0;
        } else if (diff_idx == (NUM_LIMBS - 1) && signed_op) {
            diff_val =
                cmp_lt ? ((b_msb_f >= a_msb_f) ? (b_msb_f - a_msb_f) : (b_msb_f + Fp::P - a_msb_f))
                       : ((a_msb_f >= b_msb_f) ? (a_msb_f - b_msb_f) : (a_msb_f + Fp::P - b_msb_f));
        } else if (cmp_lt) {
            diff_val = uint32_t(record.b[diff_idx] - record.a[diff_idx]);
        } else {
            diff_val = uint32_t(record.a[diff_idx] - record.b[diff_idx]);
        }

        bitwise_lookup.add_range(a_msb_range, b_msb_range);

        uint8_t diff_marker[NUM_LIMBS] = {0};
        if (diff_idx != NUM_LIMBS) {
            bitwise_lookup.add_range(diff_val - 1, 0);
            diff_marker[diff_idx] = 1;
        }

        COL_WRITE_ARRAY(row, Cols, a, record.a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, diff_marker, diff_marker);

        COL_WRITE_VALUE(row, Cols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, Cols, imm, record.imm);
        COL_WRITE_VALUE(row, Cols, opcode_blt_flag, record.local_opcode == BLT);
        COL_WRITE_VALUE(row, Cols, opcode_bltu_flag, record.local_opcode == BLTU);
        COL_WRITE_VALUE(row, Cols, opcode_bge_flag, record.local_opcode == BGE);
        COL_WRITE_VALUE(row, Cols, opcode_bgeu_flag, record.local_opcode == BGEU);
        COL_WRITE_VALUE(row, Cols, a_msb_f, a_msb_f);
        COL_WRITE_VALUE(row, Cols, b_msb_f, b_msb_f);
        COL_WRITE_VALUE(row, Cols, cmp_lt, cmp_lt);
        COL_WRITE_VALUE(row, Cols, diff_val, diff_val);
    }
};