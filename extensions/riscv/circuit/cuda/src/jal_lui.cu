#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/rdwrite.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"

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

        bool is_sign_extend = (rd_hi >> (U16_BITS - 1)) & 1;
        uint32_t imm_low_4 = record.is_jal ? 0u : (record.imm & 0xfu);

        range_checker.add_count(rd_lo, U16_BITS);
        range_checker.add_count(rd_hi, U16_BITS);
        range_checker.add_count(
            2u * rd_hi - ((uint32_t)is_sign_extend << U16_BITS), U16_BITS
        );

        if (!record.is_jal) {
            range_checker.add_count(imm_low_4, LUI_IMM_LOW_BITS);
        } else {
            range_checker.add_count(rd_hi << PC_HIGH_U16_SHIFT, U16_BITS);
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

    auto [grid, block] = kernel_launch_params(height, 512);

    jal_lui_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): tracegen from compact wire records + the per-exe operand
// table; materializes the same record structs in registers and calls the SAME
// fill methods as jal_lui_tracegen.
template <typename RecordView>
__global__ void jal_lui_tracegen_compact(
    Fp *trace,
    size_t height,
    RecordView records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        RvrWr1Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);

        Rv64JalLuiRecord full;
        full.adapter = rvr_decode_wr1_adapter(rec, entry);
        bool const is_jal = (entry.flags & RVR_OPERAND_FLAG_IS_JAL) != 0;
        full.core.imm = entry.c;
        rvr_jal_lui_rd_data(is_jal, rec.from_pc, entry.c, full.core.rd_data);
        full.core.is_jal = is_jal;
        Rv64CondRdWriteAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64JalLuiCore core(VariableRangeChecker(rc_ptr, rc_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64JalLuiCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(Rv64JalLuiCols<uint8_t>));
    }
}

extern "C" int _jal_lui_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrWr1Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(Rv64JalLuiCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    jal_lui_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_operand_table, pc_base, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _jal_lui_tracegen_g2, Rv64JalLuiCols, jal_lui_tracegen_compact, RvrWr1Compact, 256,
    operand_table, pc_base, range_checker, range_checker_num_bins, timestamp_max_bits
)
