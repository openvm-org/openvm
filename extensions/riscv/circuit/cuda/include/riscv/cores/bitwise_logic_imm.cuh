#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/cores/bitwise_logic.cuh"

using namespace riscv;

// Core columns for bitwise operations with a signed 12-bit immediate.

// Must match BitwiseLogicImmCoreRecord in src/bitwise_logic_imm/core.rs (repr(C, align(4))).
template <size_t NUM_LIMBS> struct alignas(4) BitwiseLogicImmCoreRecord {
    uint8_t b[NUM_LIMBS];
    uint8_t c_low[2];
    uint8_t imm_sign;
    uint8_t local_opcode;
};

template <typename T, size_t NUM_LIMBS> struct BitwiseLogicImmCoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c_low[2];
    T imm_sign;
    T opcode_xor_flag;
    T opcode_or_flag;
    T opcode_and_flag;
};

template <size_t NUM_LIMBS, size_t LIMB_BITS> struct BitwiseLogicImmCore {
    BitwiseOperationLookup bitwise_lookup;

    template <typename T> using Cols = BitwiseLogicImmCoreCols<T, NUM_LIMBS>;

    __device__ BitwiseLogicImmCore(BitwiseOperationLookup lookup) : bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, BitwiseLogicImmCoreRecord<NUM_LIMBS> record) {
        // Sign-extended byte limbs of the immediate.
        uint8_t sign_byte = record.imm_sign * uint8_t((1u << LIMB_BITS) - 1u);
        uint8_t c[NUM_LIMBS];
        c[0] = record.c_low[0];
        c[1] = record.c_low[1] + record.imm_sign * 0xf8;
#pragma unroll
        for (size_t i = 2; i < NUM_LIMBS; i++) {
            c[i] = sign_byte;
        }

        // BitwiseImmOpcode mirrors the XOR/OR/AND slots of BaseAluOpcode (XORI=2, ORI=3, ANDI=4).
        uint8_t a[NUM_LIMBS];
        switch (record.local_opcode) {
        case 2:
            run_xor<NUM_LIMBS>(record.b, c, a);
            break;
        case 3:
            run_or<NUM_LIMBS>(record.b, c, a);
            break;
        case 4:
            run_and<NUM_LIMBS>(record.b, c, a);
            break;
        default:
#pragma unroll
            for (size_t i = 0; i < NUM_LIMBS; i++) {
                a[i] = 0;
            }
        }

        // Adding 0xf8 forces c_low[1] into the 3-bit range.
        bitwise_lookup.add_range(record.c_low[0], record.c_low[1] + 0xf8);
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            bitwise_lookup.add_xor(record.b[i], c[i]);
        }

        COL_WRITE_VALUE(row, Cols, opcode_and_flag, record.local_opcode == 4);
        COL_WRITE_VALUE(row, Cols, opcode_or_flag, record.local_opcode == 3);
        COL_WRITE_VALUE(row, Cols, opcode_xor_flag, record.local_opcode == 2);
        COL_WRITE_VALUE(row, Cols, imm_sign, record.imm_sign);
        COL_WRITE_ARRAY(row, Cols, c_low, record.c_low);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, a, a);
    }
};
