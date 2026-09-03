#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/jalr.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct JalrCoreCols {
    T imm;                                  // 2 bytes
    T rs1_data[PTR_U16_LIMBS];         // low 32 bits of rs1 as u16 cells
    T rd_high[PTR_U16_LIMBS];          // high u16 limb and bit-32 carry of rd
    T is_valid;                             // 1 byte
    T raw_target_bit0;                  // bit zero of the target before JALR masking
    T to_pc_idx_limbs[PTR_U16_LIMBS];  // target pc index after the low-bit split
    T imm_sign;                             // 1 byte
};

__device__ void run_jalr(
    uint32_t pc,
    uint32_t rs1,
    uint16_t imm,
    bool imm_sign,
    uint32_t &out_raw_target_pc,
    uint16_t rd_data[BLOCK_FE_WIDTH]
) {
    uint32_t offset = imm + (imm_sign ? (uint32_t(UINT16_MAX) << U16_BITS) : 0);
    int64_t signed_offset = (int64_t)(int32_t)offset;
    uint64_t raw_target_pc = uint64_t(rs1) + signed_offset;

    assert(raw_target_pc <= uint64_t(UINT32_MAX));
    uint32_t to_pc = uint32_t(raw_target_pc) & ~1u;
    // RISC-V clears bit 0 before checking instruction alignment.
    assert(to_pc % DEFAULT_PC_STEP == 0);
    out_raw_target_pc = uint32_t(raw_target_pc);
    uint64_t rd_val = uint64_t(pc) + DEFAULT_PC_STEP;
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        rd_data[i] = uint16_t(rd_val >> (i * U16_BITS));
    }
}

struct JalrCore {
    VariableRangeChecker rc;

    __device__ JalrCore(VariableRangeChecker rc) : rc(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t from_pc, uint32_t rs1_val, uint16_t imm, bool imm_sign
    ) {
        uint32_t raw_target_pc;
        uint16_t rd_data[BLOCK_FE_WIDTH];
        run_jalr(from_pc, rs1_val, imm, imm_sign, raw_target_pc, rd_data);

        // to_pc_idx_limbs decompose the target pc *index* (see the Rust filler).
        uint32_t to_pc_idx = (raw_target_pc & ~1u) >> PC_STEP_BITS;
        uint32_t to_pc_idx_limbs[2] = {
            to_pc_idx & ((1u << PC_IDX_LOW_BITS) - 1), to_pc_idx >> PC_IDX_LOW_BITS
        };
        rc.add_count(to_pc_idx_limbs[0], PC_IDX_LOW_BITS);
        rc.add_count(to_pc_idx_limbs[1], U16_BITS);

        uint32_t rd_low_u16_lo = rd_data[0];
        uint32_t rd_low_u16_hi = rd_data[1];

        // rd writes the byte return address 4 * (from_pc_idx + 1). The low limb is
        // DEFAULT_PC_STEP-aligned with a PC_IDX_LOW_BITS-bit quotient; the high limb is a u16.
        rc.add_count(rd_low_u16_lo >> PC_STEP_BITS, PC_IDX_LOW_BITS);
        rc.add_count(rd_low_u16_hi, U16_BITS);

        uint16_t rs1_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_limbs, rs1_val);

        COL_WRITE_VALUE(row, JalrCoreCols, imm_sign, imm_sign);
        COL_WRITE_ARRAY(row, JalrCoreCols, to_pc_idx_limbs, to_pc_idx_limbs);
        COL_WRITE_VALUE(
            row, JalrCoreCols, raw_target_bit0, (raw_target_pc & 1) == 1 ? 1 : 0
        );
        COL_WRITE_VALUE(row, JalrCoreCols, is_valid, 1);

        COL_WRITE_ARRAY(row, JalrCoreCols, rs1_data, rs1_limbs);
        uint32_t rd_limbs[PTR_U16_LIMBS] = {rd_low_u16_hi, rd_data[2]};
        COL_WRITE_ARRAY(row, JalrCoreCols, rd_high, rd_limbs);
        COL_WRITE_VALUE(row, JalrCoreCols, imm, imm);
    }
};

template <typename T> struct JalrCols {
    JalrAdapterCols<T> adapter;
    JalrCoreCols<T> core;
};

#include "../rvr/src/jalr.inc.cuh"
