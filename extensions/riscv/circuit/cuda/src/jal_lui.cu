#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"

using namespace riscv;

template <typename T> struct Rv64JalLuiCoreCols {
    T imm;                          // core_row.imm
    T rd_data[RV64_WORD_NUM_LIMBS]; // low-32 bits of rd_data; upper limbs are sign-extended
    T is_jal;                       // core_row.is_jal
    T is_lui;                       // core_row.is_lui
    T is_sign_extend;               // 1 if upper limbs are 0xFF, 0 if 0x00
};

struct Rv64JalLuiCoreRecord {
    uint32_t imm;
    uint8_t rd_data[RV64_REGISTER_NUM_LIMBS];
    bool is_jal;
};

struct Rv64JalLuiCore {
    BitwiseOperationLookup bw;

    __device__ Rv64JalLuiCore(uint32_t *bw_ptr, uint32_t bw_bits) : bw(bw_ptr, bw_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalLuiCoreRecord record) {
#pragma unroll
        for (int i = 0; i < RV64_WORD_NUM_LIMBS; i += 2) {
            bw.add_range(record.rd_data[i], record.rd_data[i + 1]);
        }
        bool is_sign_extend = (record.rd_data[3] >> (RV64_CELL_BITS - 1)) == 1;
        int32_t second_range_limb =
            static_cast<int32_t>(record.rd_data[3]) * 2 -
            (static_cast<int32_t>(is_sign_extend) << RV64_CELL_BITS);
        bw.add_range(
            static_cast<uint32_t>(record.rd_data[3]) *
                (4u * static_cast<uint32_t>(record.is_jal) +
                 static_cast<uint32_t>(!record.is_jal)),
            static_cast<uint32_t>(second_range_limb)
        );

        COL_WRITE_VALUE(row, Rv64JalLuiCoreCols, is_sign_extend, is_sign_extend);
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
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);

    jal_lui_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, d_bw, bw_bits, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
