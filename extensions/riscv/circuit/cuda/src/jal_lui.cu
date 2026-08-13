#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

constexpr uint32_t LUI_IMM_LOW_BITS = U16_BITS - RV_IS_TYPE_IMM_BITS;

template <typename T> struct JalLuiCoreCols {
    T imm;                             // core_row.imm
    T rd_data[PTR_U16_LIMBS];     // low-32 bits of rd_data as u16 cells
    T imm_low_4;                       // low 4 bits of imm for LUI
    T is_jal;                          // core_row.is_jal
    T is_lui;                          // core_row.is_lui
    T is_sign_extend;                  // 1 if upper cells are 0xFFFF, 0 if 0x0000
};

struct JalLuiCore {
    VariableRangeChecker range_checker;

    __device__ JalLuiCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(
        RowSlice row, uint32_t imm, const uint16_t rd_data[BLOCK_FE_WIDTH], bool is_jal
    ) {
        uint32_t rd_lo = rd_data[0];
        uint32_t rd_hi = rd_data[1];

        // JAL return addresses are zero-extended; only LUI sign-extends bit 31.
        bool is_sign_extend = is_jal ? false : ((rd_hi >> (U16_BITS - 1)) & 1);
        uint32_t imm_low_4 = is_jal ? 0u : (imm & 0xfu);

        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);

        if (!is_jal) {
            range_checker.add_count(
                2u * rd_hi - ((uint32_t)is_sign_extend << U16_BITS), U16_BITS
            );
            range_checker.add_count(imm_low_4, LUI_IMM_LOW_BITS);
        } else {
            // The return address is DEFAULT_PC_STEP-aligned; range-check its low-limb quotient.
            range_checker.add_count(rd_lo / DEFAULT_PC_STEP, U16_BITS - 2);
        }

        uint32_t rd_u16[2] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, JalLuiCoreCols, is_sign_extend, is_sign_extend);
        COL_WRITE_VALUE(row, JalLuiCoreCols, is_lui, !is_jal);
        COL_WRITE_VALUE(row, JalLuiCoreCols, is_jal, is_jal);
        COL_WRITE_VALUE(row, JalLuiCoreCols, imm_low_4, imm_low_4);
        COL_WRITE_ARRAY(row, JalLuiCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, JalLuiCoreCols, imm, imm);
    }
};

template <typename T> struct JalLuiCols {
    CondRdWriteAdapterCols<T> adapter;
    JalLuiCoreCols<T> core;
};

#include "../rvr/src/jal_lui.inc.cuh"
