#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/cores/less_than.cuh"

using namespace riscv;

// ----------------------------------------------------------------------------
// Immediate-only fork of LessThanCore (SLTI/SLTIU): the `c` limbs are replaced by the ADDI-style
// two-column immediate encoding (imm_low11 + imm_sign); the sign-extended limbs are reconstructed
// here for the comparison, and imm_low11 is additionally range checked to 11 bits.
// ----------------------------------------------------------------------------

// Must match LessThanImmCoreRecord in src/less_than_imm/core.rs (repr(C)).
template <size_t NUM_LIMBS, size_t LIMB_BITS> struct LessThanImmCoreRecord {
    uint16_t b[NUM_LIMBS];
    uint16_t imm_low11;
    uint16_t imm_sign;
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS, size_t LIMB_BITS> struct LessThanImmCoreCols {
    T b[NUM_LIMBS];
    T imm_low11;
    T imm_sign;
    T cmp_result;

    T opcode_slt_flag;
    T opcode_sltu_flag;

    T b_msb_f;
    T c_msb_f;

    T diff_marker[NUM_LIMBS];
    T diff_val;
};

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct LessThanImmCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = LessThanImmCoreCols<T, NUM_LIMBS, LIMB_BITS>;

    __device__ LessThanImmCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void
    fill_trace_row(RowSlice row, LessThanImmCoreRecord<NUM_LIMBS, LIMB_BITS> record) {
        // LessThanImmOpcode: SLTI = 0, SLTIU = 1.
        bool is_slt = record.local_opcode == 0;

        // Sign-extended u16 limbs of the immediate (imm_to_u16_limbs in the Rust core).
        uint16_t c[NUM_LIMBS];
#pragma unroll
        for (size_t i = 1; i < NUM_LIMBS; i++) {
            c[i] = record.imm_sign * 0xFFFFu;
        }
        c[0] = record.imm_low11 + record.imm_sign * 0xF800u;

        LessThanResult result = run_less_than<NUM_LIMBS, LIMB_BITS>(is_slt, record.b, c);
        bool cmp_result = result.cmp_result;
        size_t diff_idx = result.diff_idx;
        bool b_sign = result.x_sign;
        bool c_sign = result.y_sign;

        uint32_t b_raw_msb = record.b[NUM_LIMBS - 1];
        uint32_t c_raw_msb = c[NUM_LIMBS - 1];

        uint32_t b_msb_f = b_sign ? (Fp::P - ((1u << LIMB_BITS) - b_raw_msb)) : b_raw_msb;
        uint32_t c_msb_f = c_sign ? (Fp::P - ((1u << LIMB_BITS) - c_raw_msb)) : c_raw_msb;

        // Shift signed MSBs into the same [0, 2^LIMB_BITS) range as unsigned MSBs.
        uint32_t b_msb_range =
            b_sign ? (b_raw_msb - (1u << (LIMB_BITS - 1)))
                   : (b_raw_msb + ((is_slt ? 1u : 0u) << (LIMB_BITS - 1)));
        uint32_t c_msb_range =
            c_sign ? (c_raw_msb - (1u << (LIMB_BITS - 1)))
                   : (c_raw_msb + ((is_slt ? 1u : 0u) << (LIMB_BITS - 1)));

        uint32_t diff_val = 0;
        if (diff_idx == NUM_LIMBS) {
            diff_val = 0;
        } else if (diff_idx == (NUM_LIMBS - 1)) {
            Fp fp_diff = cmp_result ? (Fp(c_msb_f) - Fp(b_msb_f)) : (Fp(b_msb_f) - Fp(c_msb_f));
            diff_val = fp_diff.asUInt32();
        } else if (cmp_result) {
            diff_val = uint32_t(uint16_t(uint32_t(c[diff_idx]) - uint32_t(record.b[diff_idx])));
        } else {
            diff_val = uint32_t(uint16_t(uint32_t(record.b[diff_idx]) - uint32_t(c[diff_idx])));
        }

        // Range check the low 11 bits of the immediate (unique decomposition of the operand).
        range_checker.add_count(record.imm_low11, 11);
        range_checker.add_count(b_msb_range, LIMB_BITS);
        range_checker.add_count(c_msb_range, LIMB_BITS);

        uint16_t diff_marker[NUM_LIMBS] = {0};
        if (diff_idx != NUM_LIMBS) {
            // Range-check diff_val - 1 to prove the first differing limb is non-zero.
            range_checker.add_count(diff_val - 1, LIMB_BITS);
            diff_marker[diff_idx] = 1;
        }

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, diff_marker, diff_marker);

        COL_WRITE_VALUE(row, Cols, imm_low11, record.imm_low11);
        COL_WRITE_VALUE(row, Cols, imm_sign, record.imm_sign);
        COL_WRITE_VALUE(row, Cols, cmp_result, cmp_result);
        COL_WRITE_VALUE(row, Cols, b_msb_f, b_msb_f);
        COL_WRITE_VALUE(row, Cols, c_msb_f, c_msb_f);
        COL_WRITE_VALUE(row, Cols, diff_val, diff_val);
        COL_WRITE_VALUE(row, Cols, opcode_slt_flag, is_slt);
        COL_WRITE_VALUE(row, Cols, opcode_sltu_flag, !is_slt);
    }
};
