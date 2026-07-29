#include <assert.h>

#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct Rv64AuipcCoreCols {
    T is_valid;
    T is_sign_extend;
    // The immediate is split around the byte shift in AUIPC's `imm << 8`.
    T imm_low_8;
    T imm_high_16;
    T pc_high;
    T rd_data[RV64_PTR_U16_LIMBS];
};

__device__ uint64_t run_auipc(uint32_t pc, uint32_t imm) {
    uint32_t offset = imm << RV64_BYTE_BITS;
    int64_t signed_offset = (int64_t)(int32_t)offset;
    return (uint64_t)pc + (uint64_t)signed_offset;
}

struct Rv64AuipcCore {
    VariableRangeChecker range_checker;

    __device__ Rv64AuipcCore(VariableRangeChecker range_checker) : range_checker(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, uint32_t from_pc, uint32_t imm) {
        uint32_t imm_low_8 = imm & ((1u << RV64_BYTE_BITS) - 1u);
        uint32_t imm_high_16 = (imm >> RV64_BYTE_BITS) & uint32_t(UINT16_MAX);
        uint16_t pc_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(pc_limbs, from_pc);
        uint64_t auipc = run_auipc(from_pc, imm);
        uint64_t auipc_hi = auipc >> 32;
        assert(auipc_hi == 0ull || auipc_hi == 0xffffffffull);
        uint32_t auipc_lo = (uint32_t)auipc;
        uint16_t rd_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_limbs, auipc_lo);
        uint32_t rd_lo = rd_limbs[0];
        uint32_t rd_hi = rd_limbs[1];
        uint32_t is_sign_ext = (auipc_hi != 0) ? 1u : 0u;
        uint32_t imm_sign = (imm_high_16 >> (U16_BITS - 1)) & 1u;

        range_checker.add_count(pc_limbs[0], U16_BITS);
        range_checker.add_count(pc_limbs[1], PC_BITS - U16_BITS);
        range_checker.add_count(imm_low_8, RV64_BYTE_BITS);
        range_checker.add_count(imm_high_16, U16_BITS);
        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        // Check that imm_sign matches the top bit of imm_high_16.
        range_checker.add_count(2u * imm_high_16 - (imm_sign << U16_BITS), U16_BITS);

        uint32_t rd_u16[RV64_PTR_U16_LIMBS] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_low_8, imm_low_8);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, imm_high_16, imm_high_16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, pc_high, pc_limbs[1]);
        COL_WRITE_ARRAY(row, Rv64AuipcCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_sign_extend, is_sign_ext);
        COL_WRITE_VALUE(row, Rv64AuipcCoreCols, is_valid, 1);
    }
};

template <typename T> struct Rv64AuipcCols {
    Rv64RdWriteAdapterCols<T> adapter;
    Rv64AuipcCoreCols<T> core;
};

#include "../rvr/src/auipc.inc.cuh"
