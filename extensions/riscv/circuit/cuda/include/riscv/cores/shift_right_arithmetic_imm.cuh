#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/cores/shift_right_arithmetic.cuh"

using namespace riscv;

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftRightArithmeticImmCoreRecord {
    uint16_t b[NUM_LIMBS];
    uint8_t shamt;
};

template <typename T, size_t NUM_LIMBS, size_t LIMB_BITS>
struct ShiftRightArithmeticImmCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];

    T b_sign;

    T bit_shift_marker[LIMB_BITS];
    T limb_shift_marker[NUM_LIMBS];
    T bit_shift_carry[NUM_LIMBS];
    T bit_shift_aux[NUM_LIMBS];
};

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftRightArithmeticImmCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = ShiftRightArithmeticImmCoreCols<T, NUM_LIMBS, LIMB_BITS>;

    __device__ ShiftRightArithmeticImmCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void
    fill_trace_row(RowSlice row, ShiftRightArithmeticImmCoreRecord<NUM_LIMBS, LIMB_BITS> record) {
        uint16_t shamt_limbs[NUM_LIMBS] = {0};
        shamt_limbs[0] = record.shamt;

        uint16_t a[NUM_LIMBS];
        size_t limb_shift = 0, bit_shift = 0;
        run_shift_right_arithmetic<NUM_LIMBS, LIMB_BITS>(
            record.b, shamt_limbs, a, limb_shift, bit_shift
        );

        size_t aux_bits = LIMB_BITS - bit_shift;
        uint16_t carry_arr[NUM_LIMBS];
        uint16_t aux_arr[NUM_LIMBS];
        for (size_t k = 0; k < NUM_LIMBS; k++) {
            uint32_t limb = record.b[k];
            uint32_t carry = limb & ((1u << bit_shift) - 1u);
            uint32_t aux = limb >> bit_shift;
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

        uint16_t b_sign = record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
        range_checker.add_count(
            (uint32_t)record.b[NUM_LIMBS - 1] - ((uint32_t)b_sign << (LIMB_BITS - 1)),
            LIMB_BITS - 1
        );

        COL_WRITE_VALUE(row, Cols, b_sign, b_sign);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, a, a);
    }
};
