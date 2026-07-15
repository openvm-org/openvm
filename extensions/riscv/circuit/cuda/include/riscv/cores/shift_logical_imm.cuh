#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/cores/shift_logical.cuh"

using namespace riscv;

// Core columns for logical shifts with an immediate shift amount.

// Must match ShiftLogicalImmCoreRecord in src/shift_logical_imm/core.rs (repr(C)).
template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftLogicalImmCoreRecord {
    uint16_t b[NUM_LIMBS];
    uint8_t shamt;
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftLogicalImmCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];

    T opcode_sll_flag;
    T opcode_srl_flag;

    T bit_multiplier_left;
    T carry_multiplier_left;

    T bit_shift_marker[LIMB_BITS];
    T limb_shift_marker[NUM_LIMBS];

    T bit_shift_carry[NUM_LIMBS];
    T bit_shift_aux[NUM_LIMBS];
};

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftLogicalImmCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = ShiftLogicalImmCoreCols<T, NUM_LIMBS, LIMB_BITS>;

    __device__ ShiftLogicalImmCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void
    fill_trace_row(RowSlice row, ShiftLogicalImmCoreRecord<NUM_LIMBS, LIMB_BITS> record) {
        // ShiftImmOpcode: SLLI = 0, SRLI = 1 (SRAI never reaches this chip).
        bool is_sll = record.local_opcode == 0;

        uint16_t shamt_limbs[NUM_LIMBS] = {0};
        shamt_limbs[0] = record.shamt;

        uint16_t a[NUM_LIMBS];
        size_t limb_shift = 0, bit_shift = 0;
        if (is_sll) {
            run_shift_left<NUM_LIMBS, LIMB_BITS>(record.b, shamt_limbs, a, limb_shift, bit_shift);
        } else {
            run_shift_right_logical<NUM_LIMBS, LIMB_BITS>(
                record.b, shamt_limbs, a, limb_shift, bit_shift
            );
        }

        // NOTE: no shamt-quotient range check here (unlike ShiftLogicalCore) — the immediate is
        // bound by the marker-sum constraints instead.

        size_t aux_bits = LIMB_BITS - bit_shift;
        uint16_t carry_arr[NUM_LIMBS];
        uint16_t aux_arr[NUM_LIMBS];
        for (size_t k = 0; k < NUM_LIMBS; k++) {
            uint32_t limb = record.b[k];
            uint32_t carry, aux;
            if (is_sll) {
                carry = limb >> aux_bits;
                aux = limb & ((1u << aux_bits) - 1u);
            } else {
                carry = limb & ((1u << bit_shift) - 1u);
                aux = limb >> bit_shift;
            }
            range_checker.add_count(carry, bit_shift);
            range_checker.add_count(aux, aux_bits);
            carry_arr[k] = (uint16_t)carry;
            aux_arr[k] = (uint16_t)aux;
        }
        COL_WRITE_ARRAY(row, Cols, bit_shift_carry, carry_arr);
        COL_WRITE_ARRAY(row, Cols, bit_shift_aux, aux_arr);

        uint16_t limb_marker[NUM_LIMBS] = {0};
        limb_marker[limb_shift] = 1u;
        COL_WRITE_ARRAY(row, Cols, limb_shift_marker, limb_marker);
        uint16_t bit_marker[LIMB_BITS] = {0};
        bit_marker[bit_shift] = 1u;
        COL_WRITE_ARRAY(row, Cols, bit_shift_marker, bit_marker);

        uint32_t bit_mult = 1u << bit_shift;
        uint32_t carry_mult = 1u << aux_bits;
        COL_WRITE_VALUE(row, Cols, carry_multiplier_left, is_sll ? carry_mult : 0u);
        COL_WRITE_VALUE(row, Cols, bit_multiplier_left, is_sll ? bit_mult : 0u);

        COL_WRITE_VALUE(row, Cols, opcode_srl_flag, is_sll ? 0u : 1u);
        COL_WRITE_VALUE(row, Cols, opcode_sll_flag, is_sll ? 1u : 0u);

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, a, a);
    }
};
