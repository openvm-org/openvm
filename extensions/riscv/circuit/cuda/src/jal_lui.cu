#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // PC_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;
using namespace program;

// Pattern B u16: rd_data is 2 u16 limbs (low 32 bits of rd); imm_low_4 is a new witness used
// only for LUI's `imm = imm_low_4 + rd[1] * 16` constraint.
template <typename T> struct Rv64JalLuiCoreCols {
    T imm;
    T rd_data[2]; // low-32 bits of rd as 2 u16 limbs; upper limbs sign-extended at the bus
    T imm_low_4;  // imm & 0xf, range-checked to 4 bits (LUI only)
    T is_jal;
    T is_lui;
    T is_sign_extend; // 1 if upper limbs are 0xffff, 0 otherwise
};

struct Rv64JalLuiCoreRecord {
    uint32_t imm;
    uint16_t rd_data[BLOCK_FE_WIDTH]; // 4 u16 cells: low 32 bits in [0..1], sign-extension in [2..3]
    bool is_jal;
};

struct Rv64JalLuiCore {
    BitwiseOperationLookup bw;
    VariableRangeChecker range_checker;

    __device__ Rv64JalLuiCore(BitwiseOperationLookup bw, VariableRangeChecker rc)
        : bw(bw), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalLuiCoreRecord record) {
        uint32_t rd_lo = record.rd_data[0];
        uint32_t rd_hi = record.rd_data[1];

        bool is_sign_extend = (rd_hi >> 15) & 1;
        uint32_t imm_low_4 = record.is_jal ? 0u : (record.imm & 0xfu);

        // u16 range checks for rd_data.
        range_checker.add_count(rd_lo, 16);
        range_checker.add_count(rd_hi, 16);
        // Sign-extension consistency: 2 * rd_hi - is_sign_extend * 2^16 ∈ [0, 2^16).
        range_checker.add_count(2u * rd_hi - ((uint32_t)is_sign_extend << 16), 16);

        if (!record.is_jal) {
            // LUI: range-check imm_low_4 to 4 bits.
            range_checker.add_count(imm_low_4, 4);
        } else {
            // JAL: range-check rd_hi to PC_BITS - 16 bits via the shifted form.
            const uint32_t shift = 16 - (PC_BITS - 16);
            range_checker.add_count(rd_hi << shift, 16);
        }

        // bitwise lookup not used in the u16 path, but kept for API stability with the chip.
        (void)bw;

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
    uint32_t *bw_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full = records[idx];

        Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64JalLuiCore core(
            BitwiseOperationLookup(bw_ptr), VariableRangeChecker(rc_ptr, rc_bins)
        );
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
    uint32_t *d_bw,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jal_lui_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, d_bw, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
