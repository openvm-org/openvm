#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // PC_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv-adapters/constants.cuh"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

constexpr uint32_t PC_HIGH_U16_SHIFT = 2 * RV64_U16_LIMB_BITS - PC_BITS;

template <typename T> struct Rv64JalLuiCoreCols {
    T imm;
    T rd_data[RV64_LOW32_U16_LIMBS];
    T imm_low_4;
    T is_jal;
    T is_lui;
    T is_sign_extend;
};

struct Rv64JalLuiCoreRecord {
    uint32_t imm;
    uint16_t rd_data[BLOCK_FE_WIDTH];
    bool is_jal;
};

struct Rv64JalLuiCore {
    VariableRangeChecker range_checker;

    __device__ Rv64JalLuiCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalLuiCoreRecord record) {
        uint32_t rd_lo = record.rd_data[0];
        uint32_t rd_hi = record.rd_data[1];

        bool is_sign_extend = (rd_hi >> RV64_U16_SIGN_BIT) & 1;
        uint32_t imm_low_4 = record.is_jal ? 0u : (record.imm & 0xfu);

        // Range-check the low-32 rd cells used by the JAL/LUI relations.
        range_checker.add_count(rd_lo, RV64_U16_LIMB_BITS);
        range_checker.add_count(rd_hi, RV64_U16_LIMB_BITS);
        // Tie is_sign_extend to bit 31, the top bit of rd_hi.
        range_checker.add_count(
            2u * rd_hi - ((uint32_t)is_sign_extend << RV64_U16_LIMB_BITS), RV64_U16_LIMB_BITS
        );

        if (!record.is_jal) {
            // LUI constrains the low immediate part used across the 12-bit shift.
            range_checker.add_count(imm_low_4, 4);
        } else {
            // JAL constrains the return address to fit within PC_BITS.
            range_checker.add_count(rd_hi << PC_HIGH_U16_SHIFT, RV64_U16_LIMB_BITS);
        }

        uint32_t rd_u16[2] = {rd_lo, rd_hi};
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_sign_extend, is_sign_extend);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_lui, !record.is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_jal, record.is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm_low_4, imm_low_4);
        COL_WRITE_ARRAY(row, Rv64JalLuiCoreCols, rd_data, rd_u16);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, imm, record.imm);
    }
};

template <typename T> struct Rv64JalLuiCols {
    Rv64CondRdWriteAdapterCols<T> adapter;
    Rv64JalLuiCoreCols<T> core;
};

struct Rv64JalLuiRecord {
    Rv64RdWriteAdapterRecord adapter;
    Rv64JalLuiCoreRecord core;
};

__global__ void jal_lui_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<Rv64JalLuiRecord> records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full = records[idx];

        Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64JalLuiCore core(VariableRangeChecker(rc_ptr, rc_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64JalLuiCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64JalLuiCols<uint8_t>));
    }
}

extern "C" int _jal_lui_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64JalLuiRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jal_lui_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
