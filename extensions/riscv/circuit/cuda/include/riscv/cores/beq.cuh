#pragma once

#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

template <size_t NUM_LIMBS> struct BranchEqualCoreRecord {
    uint16_t a[NUM_LIMBS];
    uint16_t b[NUM_LIMBS];
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

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct BranchEqualCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = BranchEqualCoreCols<T, NUM_LIMBS>;

    __device__ BranchEqualCore(VariableRangeChecker rc) : range_checker(rc) {}

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

        // Mirror the AIR's per-limb LIMB_BITS-wide range-checks on a[i] and b[i]
        // (Pattern B u16 path).
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; ++i) {
            range_checker.add_count(rec.a[i], LIMB_BITS);
            range_checker.add_count(rec.b[i], LIMB_BITS);
        }
    }
};
