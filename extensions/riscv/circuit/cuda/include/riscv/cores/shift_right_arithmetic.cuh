#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

// ----------------------------------------------------------------------------
// u16-limb arithmetic-right shift (SRA). Mirrors the SRL path of shift_logical.cuh: each b limb is
// split into a `carry` part (the low bits that cross into the previous limb) and an `aux` part (the
// bits that stay), then recombined additively so no constraint term exceeds 2^LIMB_BITS (which
// would alias the field modulus). The vacated high bits are filled with the sign bit of the top
// limb, which is range checked.
// ----------------------------------------------------------------------------

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftRightArithmeticCoreRecord {
    uint16_t b[NUM_LIMBS];
    uint16_t c[NUM_LIMBS];
};

template <typename T, size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftRightArithmeticCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];

    T b_sign;

    T bit_shift_marker[LIMB_BITS];
    T limb_shift_marker[NUM_LIMBS];

    T bit_shift_carry[NUM_LIMBS];
    T bit_shift_aux[NUM_LIMBS];
};

template <size_t NUM_LIMBS, size_t LIMB_BITS>
__forceinline__ __device__ void run_shift_right_arithmetic(
    const uint16_t x[NUM_LIMBS],
    const uint16_t y[NUM_LIMBS],
    uint16_t result[NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift
) {
    uint16_t msb = x[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
    uint16_t fill = (uint16_t)(((1u << LIMB_BITS) - 1u) * msb);
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        result[i] = fill;
    }
    size_t max_bits = NUM_LIMBS * LIMB_BITS;
    size_t shift = y[0] % max_bits;
    limb_shift = shift / LIMB_BITS;
    bit_shift = shift % LIMB_BITS;

    size_t limit = NUM_LIMBS - limb_shift;
    for (size_t i = 0; i < limit; i++) {
        uint32_t part1 = (uint32_t)x[i + limb_shift] >> bit_shift;
        uint32_t part2 = (i + limb_shift + 1 < NUM_LIMBS)
                             ? ((uint32_t)x[i + limb_shift + 1] << (LIMB_BITS - bit_shift))
                             : ((uint32_t)fill << (LIMB_BITS - bit_shift));
        result[i] = (part1 | part2) & ((1u << LIMB_BITS) - 1u);
    }
}

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct ShiftRightArithmeticCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = ShiftRightArithmeticCoreCols<T, NUM_LIMBS, LIMB_BITS>;

    __device__ ShiftRightArithmeticCore(VariableRangeChecker range) : range_checker(range) {}

    __device__ void
    fill_trace_row(RowSlice row, ShiftRightArithmeticCoreRecord<NUM_LIMBS, LIMB_BITS> record) {
        uint16_t a[NUM_LIMBS];
        size_t limb_shift = 0, bit_shift = 0;
        run_shift_right_arithmetic<NUM_LIMBS, LIMB_BITS>(
            record.b, record.c, a, limb_shift, bit_shift
        );

        size_t combined_bits = NUM_LIMBS * LIMB_BITS;
        size_t num_bits_log = 0;
        while ((1u << num_bits_log) < combined_bits) {
            ++num_bits_log;
        }
        range_checker.add_count(
            ((uint32_t)record.c[0] - (uint32_t)bit_shift -
             (uint32_t)(limb_shift * LIMB_BITS)) >>
                num_bits_log,
            LIMB_BITS - num_bits_log
        );

        // SRL-style carry/aux decomposition of each b limb: b[k] = carry[k] + aux[k] * 2^bit_shift.
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

        uint16_t b_sign_val = record.b[NUM_LIMBS - 1] >> (LIMB_BITS - 1);
        COL_WRITE_VALUE(row, Cols, b_sign, b_sign_val);
        // b_sign correctness: range check the low LIMB_BITS-1 bits of b[NUM_LIMBS - 1].
        range_checker.add_count(
            (uint32_t)record.b[NUM_LIMBS - 1] - ((uint32_t)b_sign_val << (LIMB_BITS - 1)),
            LIMB_BITS - 1
        );

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, a, a);
    }
};
