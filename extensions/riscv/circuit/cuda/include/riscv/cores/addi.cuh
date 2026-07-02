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

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct AddICore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = AddICoreCols<T, NUM_LIMBS>;

    __device__ AddICore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, AddICoreRecord<NUM_LIMBS> record) {
        uint16_t rd[NUM_LIMBS];

        // First limb: rs1[0] + imm_low11 + imm_sign * 0xF800
        uint32_t overflow = static_cast<uint32_t>(record.rs1[0])
            + static_cast<uint32_t>(record.imm_low11)
            + static_cast<uint32_t>(record.imm_sign) * 0xF800u;
        uint32_t carry = overflow >> LIMB_BITS;
        rd[0] = static_cast<uint16_t>(overflow & ((1u << LIMB_BITS) - 1));

        // Remaining limbs: rs1[i] + sign_u16 + carry
        uint32_t sign_u16 = static_cast<uint32_t>(record.imm_sign) * 0xFFFFu;
#pragma unroll
        for (size_t i = 1; i < NUM_LIMBS; i++) {
            overflow = static_cast<uint32_t>(record.rs1[i]) + sign_u16 + carry;
            carry = overflow >> LIMB_BITS;
            rd[i] = static_cast<uint16_t>(overflow & ((1u << LIMB_BITS) - 1));
        }

        COL_WRITE_ARRAY(row, Cols, rs1, record.rs1);
        COL_WRITE_ARRAY(row, Cols, rd, rd);
        COL_WRITE_VALUE(row, Cols, imm_low11, record.imm_low11);
        COL_WRITE_VALUE(row, Cols, imm_sign, record.imm_sign);
        COL_WRITE_VALUE(row, Cols, is_valid, 1u);

        range_checker.add_count(record.imm_low11, 11);
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            range_checker.add_count(rd[i], LIMB_BITS);
        }
    }
};
