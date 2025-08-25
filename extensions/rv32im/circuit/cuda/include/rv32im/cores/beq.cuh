#pragma once

#include "primitives/trace_access.h"

template <size_t NUM_LIMBS> struct BranchEqualCoreRecord {
    uint8_t a[NUM_LIMBS];
    uint8_t b[NUM_LIMBS];
    uint32_t imm;
    uint8_t local_opcode;
};
template <typename T, size_t NUM_LIMBS> struct BranchEqualCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T cmp_result;
    T imm;
    T opcode_beq_flag;
    T opcode_bne_flag;
    T diff_inv_marker[NUM_LIMBS];
};

template <size_t NUM_LIMBS> struct BranchEqualCore {
    template <typename T> using Cols = BranchEqualCoreCols<T, NUM_LIMBS>;

    __device__ void fill_trace_row(RowSlice row, BranchEqualCoreRecord<NUM_LIMBS> rec) {
        size_t diff_idx = NUM_LIMBS;
        for (size_t i = 0; i < NUM_LIMBS; ++i) {
            if (rec.a[i] != rec.b[i]) {
                diff_idx = i;
                break;
            }
        }

        bool is_beq = (rec.local_opcode == 0);
        bool cmp_result;
        Fp diff_inv_val = Fp::zero();

        if (diff_idx == NUM_LIMBS) {
            cmp_result = is_beq;
            diff_idx = 0;
        } else {
            cmp_result = !is_beq;
            Fp diff = Fp(rec.a[diff_idx]) - Fp(rec.b[diff_idx]);
            diff_inv_val = inv(diff);
        }

        COL_WRITE_ARRAY(row, Cols, a, rec.a);
        COL_WRITE_ARRAY(row, Cols, b, rec.b);

        for (size_t i = 0; i < NUM_LIMBS; ++i) {
            COL_WRITE_VALUE(
                row, Cols, diff_inv_marker[i], (i == diff_idx) ? diff_inv_val : Fp::zero()
            );
        }

        COL_WRITE_VALUE(row, Cols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, Cols, imm, rec.imm);
        COL_WRITE_VALUE(row, Cols, opcode_beq_flag, is_beq);
        COL_WRITE_VALUE(row, Cols, opcode_bne_flag, !is_beq);
    }
};
