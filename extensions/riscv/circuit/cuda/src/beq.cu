#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/beq.cuh"
#include "riscv/rvr_compact.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

using Rv64BranchEqualCore = BranchEqualCore<BLOCK_FE_WIDTH>;
template <typename T>
using Rv64BranchEqualCoreCols = BranchEqualCoreCols<T, BLOCK_FE_WIDTH>;
using Rv64BranchEqualCoreRecord = BranchEqualCoreRecord<BLOCK_FE_WIDTH>;

template <typename T> struct BranchEqualCols {
    Rv64BranchAdapterCols<T> adapter;
    Rv64BranchEqualCoreCols<T> core;
};

struct BranchEqualRecord {
    Rv64BranchAdapterRecord adapter;
    Rv64BranchEqualCoreRecord core;
};

__global__ void beq_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<BranchEqualRecord> records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full = records[idx];

        Rv64BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);

        Rv64BranchEqualCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqualCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(BranchEqualCols<uint8_t>));
    }
}

extern "C" int _beq_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BranchEqualRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchEqualCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);
    beq_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): tracegen from compact wire records + the per-exe operand
// table; materializes the same record structs in registers and calls the SAME
// fill methods as beq_tracegen.
__global__ void beq_tracegen_compact(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<RvrBranch2Compact> records,
    RvrOperandEntry const *operand_table,
    uint32_t pc_base,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        RvrBranch2Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);

        BranchEqualRecord full;
        full.adapter = rvr_decode_branch2_adapter(rec, entry);
#pragma unroll
        for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
            full.core.a[i] = rvr_u16_limb(rec.b, i);
            full.core.b[i] = rvr_u16_limb(rec.c, i);
        }
        full.core.imm = entry.c;
        full.core.local_opcode = entry.local_opcode;
        Rv64BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);
        Rv64BranchEqualCore core;
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchEqualCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(BranchEqualCols<uint8_t>));
    }
}

extern "C" int _beq_tracegen_compact(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<RvrBranch2Compact> d_records,
    RvrOperandEntry const *d_operand_table,
    uint32_t pc_base,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchEqualCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    beq_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_operand_table, pc_base, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
