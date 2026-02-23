#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/rdwrite.cuh"

using namespace riscv;

const uint32_t ADDITIONAL_BITS = 0b11000000;

template <typename T> struct Rv64JalLuiCoreCols {
    T imm;                              // core_row.imm
    T rd_data[RV32_REGISTER_NUM_LIMBS]; // core_row.rd_data
    T is_jal;                           // core_row.is_jal
    T is_lui;                           // core_row.is_lui
};

struct Rv64JalLuiCoreRecord {
    uint32_t imm;
    uint8_t rd_data[RV32_REGISTER_NUM_LIMBS];
    bool is_jal;
};

struct Rv64JalLuiCore {
    BitwiseOperationLookup bw;

    __device__ Rv64JalLuiCore(uint32_t *bw_ptr, uint32_t bw_bits) : bw(bw_ptr, bw_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalLuiCoreRecord record) {
#pragma unroll
        for (int i = 0; i < RV32_REGISTER_NUM_LIMBS; i += 2) {
            bw.add_range(record.rd_data[i], record.rd_data[i + 1]);
        }
        if (record.is_jal) {
            bw.add_xor(record.rd_data[RV32_REGISTER_NUM_LIMBS - 1], ADDITIONAL_BITS);
        }

        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_lui, !record.is_jal);
        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_jal, record.is_jal);
        COL_WRITE_ARRAY(row, Rv64JalLuiCoreCols, rd_data, record.rd_data);
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
    uint32_t bw_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full = records[idx];

        Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64JalLuiCore core(bw_ptr, bw_bits);
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
    uint32_t bw_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jal_lui_tracegen<<<grid, block>>>(
        d_trace, height, d_records, d_rc, rc_bins, d_bw, bw_bits, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
