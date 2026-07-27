#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/jalr.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64JalrCoreCols {
    T imm;                                  // 2 bytes
    T rs1_data[RV64_PTR_U16_LIMBS];         // low 32 bits of rs1 as u16 cells
    T rd_high[RV64_PTR_U16_LIMBS - 1];      // high u16 limb of low-32 rd
    T is_valid;                             // 1 byte
    T to_pc_least_sig_bit;                  // 1 byte
    T to_pc_limbs[RV64_PTR_U16_LIMBS];      // `to_pc * 2` after the low-bit split
    T imm_sign;                             // 1 byte
};

struct Rv64JalrCoreRecord {
    uint16_t imm;
    uint32_t from_pc;
    uint32_t rs1_val;
    uint8_t imm_sign; // 0 or 1
};

__device__ void run_jalr(
    uint32_t pc,
    uint32_t rs1,
    uint16_t imm,
    bool imm_sign,
    uint32_t &out_pc,
    uint16_t rd_data[BLOCK_FE_WIDTH]
) {
    uint32_t offset = imm + (imm_sign ? (uint32_t(UINT16_MAX) << U16_BITS) : 0);
    int64_t signed_offset = (int64_t)(int32_t)offset;
    uint64_t to_pc = uint64_t(rs1) + signed_offset;

    assert(to_pc < (uint64_t(1) << PC_BITS));
    out_pc = uint32_t(to_pc);
    uint32_t rd_val = pc + DEFAULT_PC_STEP;
    rd_data[0] = uint16_t(rd_val);
    rd_data[1] = uint16_t(rd_val >> U16_BITS);
#pragma unroll
    for (size_t i = RV64_PTR_U16_LIMBS; i < BLOCK_FE_WIDTH; i++) {
        rd_data[i] = 0;
    }
}

struct Rv64JalrCore {
    VariableRangeChecker rc;

    __device__ Rv64JalrCore(VariableRangeChecker rc) : rc(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t from_pc, uint32_t rs1_val, uint16_t imm, bool imm_sign
    ) {
        uint32_t to_pc;
        uint16_t rd_data[BLOCK_FE_WIDTH];
        run_jalr(from_pc, rs1_val, imm, imm_sign, to_pc, rd_data);

        uint16_t to_pc_u16[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(to_pc_u16, to_pc);
        uint32_t to_pc_limbs[2] = {uint32_t(to_pc_u16[0] >> 1), uint32_t(to_pc_u16[1])};
        // to_pc_limbs[0] is 15 bits because it is doubled to reconstruct
        // the aligned JALR target.
        rc.add_count(to_pc_limbs[0], U16_BITS - 1);
        rc.add_count(to_pc_limbs[1], PC_BITS - U16_BITS);

        uint32_t rd_low_u16_lo = rd_data[0];
        uint32_t rd_low_u16_hi = rd_data[1];

        // rd writes the low 32 bits of from_pc + DEFAULT_PC_STEP. The high
        // limb is narrowed to the remaining PC bits because from_pc is program-bus bounded.
        rc.add_count(rd_low_u16_lo, U16_BITS);
        rc.add_count(rd_low_u16_hi, PC_BITS - U16_BITS);

        uint16_t rs1_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_limbs, rs1_val);

        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm_sign, imm_sign);
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, to_pc_limbs, to_pc_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, to_pc_least_sig_bit, (to_pc & 1) == 1 ? 1 : 0);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, is_valid, 1);

        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rs1_data, rs1_limbs);
        uint32_t rd_limbs[RV64_PTR_U16_LIMBS - 1] = {rd_low_u16_hi};
        COL_WRITE_ARRAY(row, Rv64JalrCoreCols, rd_high, rd_limbs);
        COL_WRITE_VALUE(row, Rv64JalrCoreCols, imm, imm);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64JalrCoreRecord record) {
        fill_trace_row(row, record.from_pc, record.rs1_val, record.imm, record.imm_sign);
    }
};

template <typename T> struct Rv64JalrCols {
    Rv64JalrAdapterCols<T> adapter;
    Rv64JalrCoreCols<T> core;
};

struct Rv64JalrRecord {
    Rv64JalrAdapterRecord adapter;
    Rv64JalrCoreRecord core;
};

#include "../rvr/src/jalr.inc.cuh"
