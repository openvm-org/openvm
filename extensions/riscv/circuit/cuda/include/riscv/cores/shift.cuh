#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS> struct ShiftCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct ShiftCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];

    T opcode_sll_flag;
    T opcode_srl_flag;
    T opcode_sra_flag;

    T bit_multiplier_left;
    T bit_multiplier_right;
    T b_sign;

    T bit_shift_marker[RV32_CELL_BITS];
    T limb_shift_marker[NUM_LIMBS];

    T bit_shift_carry[NUM_LIMBS];
};

template <size_t NUM_LIMBS>
__forceinline__ __device__ void get_shift(
    const uint8_t y[NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift
) {
    size_t max_bits = NUM_LIMBS * RV32_CELL_BITS;
    size_t shift = y[0] % max_bits;
    limb_shift = shift / RV32_CELL_BITS;
    bit_shift = shift % RV32_CELL_BITS;
}

template <size_t NUM_LIMBS>
__forceinline__ __device__ void run_shift_left(
    const uint8_t x[NUM_LIMBS],
    const uint8_t y[NUM_LIMBS],
    uint8_t result[NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift
) {
    get_shift<NUM_LIMBS>(y, limb_shift, bit_shift);

#pragma unroll
    for (size_t i = 0; i < limb_shift; i++) {
        result[i] = 0;
    }

#pragma unroll
    for (size_t i = limb_shift; i < NUM_LIMBS; i++) {
        if (i > limb_shift) {
            uint16_t high = (uint16_t)x[i - limb_shift] << bit_shift;
            uint16_t low = (uint16_t)x[i - limb_shift - 1] >> (RV32_CELL_BITS - bit_shift);
            result[i] = (high | low) % (1u << RV32_CELL_BITS);
        } else {
            uint16_t high = (uint16_t)x[i - limb_shift] << bit_shift;
            result[i] = high % (1u << RV32_CELL_BITS);
        }
    }
}

template <size_t NUM_LIMBS>
__forceinline__ __device__ void run_shift_right(
    const uint8_t x[NUM_LIMBS],
    const uint8_t y[NUM_LIMBS],
    uint8_t result[NUM_LIMBS],
    size_t &limb_shift,
    size_t &bit_shift,
    bool logical
) {
    uint8_t msb = x[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1);
    uint8_t fill = logical ? 0u : ((1u << RV32_CELL_BITS) - 1u) * msb;
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        result[i] = fill;
    }
    get_shift<NUM_LIMBS>(y, limb_shift, bit_shift);
    size_t limit = NUM_LIMBS - limb_shift;
#pragma unroll
    for (size_t i = 0; i < limit; i++) {
        uint16_t part1 = (uint16_t)(x[i + limb_shift] >> bit_shift);
        uint16_t part2_val = (i + limb_shift + 1 < NUM_LIMBS) ? x[i + limb_shift + 1] : fill;
        uint16_t part2 = (uint16_t)part2_val << (RV32_CELL_BITS - bit_shift);
        result[i] = (part1 | part2) % (1u << RV32_CELL_BITS);
    }
}

template <size_t NUM_LIMBS> struct ShiftCore {
    BitwiseOperationLookup bitwise_lookup;
    VariableRangeChecker range_checker;

    template <typename T> using Cols = ShiftCoreCols<T, NUM_LIMBS>;

    __device__ ShiftCore(BitwiseOperationLookup lookup, VariableRangeChecker range)
        : bitwise_lookup(lookup), range_checker(range) {}

    __device__ void fill_trace_row(RowSlice row, ShiftCoreRecord<NUM_LIMBS> record) {
        bool is_sll = record.local_opcode == 0;
        bool is_srl = record.local_opcode == 1;
        bool is_sra = record.local_opcode == 2;

        uint8_t a[NUM_LIMBS];
        size_t limb_shift = 0, bit_shift = 0;
        if (is_sll) {
            run_shift_left<NUM_LIMBS>(record.b, record.c, a, limb_shift, bit_shift);
        } else {
            run_shift_right<NUM_LIMBS>(record.b, record.c, a, limb_shift, bit_shift, is_srl);
        }

#pragma unroll
        for (size_t i = 0; i + 1 < NUM_LIMBS; i += 2) {
            bitwise_lookup.add_range(a[i], a[i + 1]);
        }

        size_t combined_bits = NUM_LIMBS * RV32_CELL_BITS;
        size_t num_bits_log = 0;
        while ((1u << num_bits_log) < combined_bits) {
            ++num_bits_log;
        }
        range_checker.add_count(
            ((uint32_t)record.c[0] - (uint32_t)bit_shift -
             (uint32_t)(limb_shift * RV32_CELL_BITS)) >>
                num_bits_log,
            RV32_CELL_BITS - num_bits_log
        );

        uint8_t carry_arr[NUM_LIMBS];
        if (bit_shift == 0) {
#pragma unroll
            for (size_t i = 0; i < NUM_LIMBS; i++) {
                range_checker.add_count(0u, 0u);
                carry_arr[i] = 0u;
            }
        } else {
#pragma unroll
            for (size_t i = 0; i < NUM_LIMBS; i++) {
                uint8_t carry = is_sll ? (record.b[i] >> (RV32_CELL_BITS - bit_shift))
                                       : (record.b[i] & ((1u << bit_shift) - 1u));
                range_checker.add_count((uint32_t)carry, bit_shift);
                carry_arr[i] = carry;
            }
        }

        COL_WRITE_ARRAY(row, Cols, bit_shift_carry, carry_arr);

        uint8_t limb_marker[NUM_LIMBS] = {0};
        limb_marker[limb_shift] = 1u;
        COL_WRITE_ARRAY(row, Cols, limb_shift_marker, limb_marker);
        uint8_t bit_marker[RV32_CELL_BITS] = {0};
        bit_marker[bit_shift] = 1u;
        COL_WRITE_ARRAY(row, Cols, bit_shift_marker, bit_marker);

        uint8_t b_sign_val = is_sra ? (record.b[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1)) : 0u;
        COL_WRITE_VALUE(row, Cols, b_sign, b_sign_val);
        if (is_sra) {
            bitwise_lookup.add_xor(record.b[NUM_LIMBS - 1], 1u << (RV32_CELL_BITS - 1));
        }
        COL_WRITE_VALUE(row, Cols, bit_multiplier_left, is_sll ? (1u << bit_shift) : 0u);
        COL_WRITE_VALUE(row, Cols, bit_multiplier_right, is_sll ? 0u : (1u << bit_shift));

        COL_WRITE_VALUE(row, Cols, opcode_sll_flag, is_sll ? 1u : 0u);
        COL_WRITE_VALUE(row, Cols, opcode_srl_flag, is_srl ? 1u : 0u);
        COL_WRITE_VALUE(row, Cols, opcode_sra_flag, is_sra ? 1u : 0u);

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, a, a);
    }
};