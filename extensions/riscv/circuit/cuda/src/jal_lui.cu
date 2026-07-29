#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

constexpr uint32_t LUI_IMM_LOW_BITS = U16_BITS - RV_IS_TYPE_IMM_BITS;
constexpr uint32_t PC_HIGH_U16_SHIFT = 2 * U16_BITS - PC_BITS;

template <typename T> struct Rv64JalLuiCoreCols {
    T imm;                             // core_row.imm
    T rd_data[RV64_PTR_U16_LIMBS];     // low-32 bits of rd_data as u16 cells
    T imm_low_4;                       // low 4 bits of imm for LUI
    T is_jal;                          // core_row.is_jal
    T is_lui;                          // core_row.is_lui
    T is_sign_extend;                  // 1 if upper cells are 0xFFFF, 0 if 0x0000
};

struct Rv64JalLuiCore {
    VariableRangeChecker range_checker;

    __device__ Rv64JalLuiCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t imm, const uint16_t rd_data[BLOCK_FE_WIDTH], bool is_jal
    ) {
        uint32_t rd_lo = rd_data[0];
        uint32_t rd_hi = rd_data[1];

        bool is_sign_extend = (rd_hi >> (U16_BITS - 1)) & 1;
        uint32_t imm_low_4 = is_jal ? 0u : (imm & 0xfu);

        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        range_checker.add_count(
            2u * rd_hi - ((uint32_t)is_sign_extend << U16_BITS), U16_BITS
        );

        if (!is_jal) {
            range_checker.add_count(imm_low_4, LUI_IMM_LOW_BITS);
        } else {
            range_checker.add_count(rd_hi << PC_HIGH_U16_SHIFT, U16_BITS);
        }

        uint32_t rd_u16[2] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_sign_extend, is_sign_extend);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_lui, !is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_jal, is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm_low_4, imm_low_4);
        COL_WRITE_ARRAY(row, Rv64JalLuiCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm, imm);
    }
};

template <typename T> struct Rv64JalLuiCols {
    Rv64CondRdWriteAdapterCols<T> adapter;
    Rv64JalLuiCoreCols<T> core;
};

#include "../rvr/src/jal_lui.inc.cuh"
