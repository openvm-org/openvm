#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS> struct AddICoreRecord {
    uint16_t rs1[NUM_LIMBS];
    uint16_t imm_low11;
    uint16_t imm_sign;
};

template <typename T, size_t NUM_LIMBS> struct AddICoreCols {
    T rd[NUM_LIMBS];
    T rs1[NUM_LIMBS];
    T imm_low11;
    T imm_sign;
    T is_valid;
};

template <size_t NUM_LIMBS, size_t LIMB_BITS, bool RANGE_MOST_SIGNIFICANT_LIMB = true>
struct AddICore {
    static_assert(NUM_LIMBS > 0 && LIMB_BITS >= 12 && LIMB_BITS <= sizeof(uint16_t) * 8);
    static constexpr uint32_t LIMB_BASE = 1u << LIMB_BITS;
    static constexpr uint32_t LIMB_MASK = LIMB_BASE - 1;
    static constexpr uint32_t IMM_SIGN_EXTENSION = LIMB_BASE - (1u << 11);

    VariableRangeChecker range_checker;

    template <typename T> using Cols = AddICoreCols<T, NUM_LIMBS>;

    __device__ AddICore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, AddICoreRecord<NUM_LIMBS> record) {
        uint16_t rd[NUM_LIMBS];

        // First limb: rs1[0] + imm_low11 + imm_sign * IMM_SIGN_EXTENSION
        uint32_t overflow = static_cast<uint32_t>(record.rs1[0]) +
                            static_cast<uint32_t>(record.imm_low11) +
                            static_cast<uint32_t>(record.imm_sign) * IMM_SIGN_EXTENSION;
        uint32_t carry = overflow >> LIMB_BITS;
        rd[0] = static_cast<uint16_t>(overflow & LIMB_MASK);

        // Remaining limbs: rs1[i] + sign_limb + carry
        uint32_t sign_limb = static_cast<uint32_t>(record.imm_sign) * LIMB_MASK;
#pragma unroll
        for (size_t i = 1; i < NUM_LIMBS; i++) {
            overflow = static_cast<uint32_t>(record.rs1[i]) + sign_limb + carry;
            carry = overflow >> LIMB_BITS;
            rd[i] = static_cast<uint16_t>(overflow & LIMB_MASK);
        }

        COL_WRITE_ARRAY(row, Cols, rs1, record.rs1);
        COL_WRITE_ARRAY(row, Cols, rd, rd);
        COL_WRITE_VALUE(row, Cols, imm_low11, record.imm_low11);
        COL_WRITE_VALUE(row, Cols, imm_sign, record.imm_sign);
        COL_WRITE_VALUE(row, Cols, is_valid, 1u);

        range_checker.add_count(record.imm_low11, 11);
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS - !RANGE_MOST_SIGNIFICANT_LIMB; i++) {
            range_checker.add_count(rd[i], LIMB_BITS);
        }
    }
};
