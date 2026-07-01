#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

template <size_t NUM_LIMBS> struct AddICoreRecord {
    uint16_t b[NUM_LIMBS];
    uint16_t c[NUM_LIMBS];
};

template <typename T, size_t NUM_LIMBS> struct AddICoreCols {
    T a[NUM_LIMBS];
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T is_valid;
};

// AddI performs addition only — no sub variant, no opcode flag.
template <size_t NUM_LIMBS, size_t LIMB_BITS> struct AddICore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = AddICoreCols<T, NUM_LIMBS>;

    __device__ AddICore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, AddICoreRecord<NUM_LIMBS> record) {
        uint16_t a[NUM_LIMBS];
        uint32_t carry_buf[NUM_LIMBS];

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            uint32_t res = static_cast<uint32_t>(record.b[i]) + static_cast<uint32_t>(record.c[i]);
            if (i > 0) res += carry_buf[i - 1];
            carry_buf[i] = res >> LIMB_BITS;
            a[i] = static_cast<uint16_t>(res & ((1u << LIMB_BITS) - 1));
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_VALUE(row, Cols, is_valid, 1u);

#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            range_checker.add_count(a[i], LIMB_BITS);
        }
    }
};
