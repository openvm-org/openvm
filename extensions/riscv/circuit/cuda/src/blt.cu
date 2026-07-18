#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/blt.cuh"
#include "riscv/rvr_compact.cuh"
#include "riscv/rvr_g2_trace.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

using Rv64BranchLessThanCoreRecord =
    BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64BranchLessThanCore = BranchLessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64BranchLessThanCoreCols =
    BranchLessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct BranchLessThanCols {
    Rv64BranchAdapterCols<T> adapter;
    Rv64BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv64BranchAdapterRecord adapter;
    Rv64BranchLessThanCoreRecord core;
};

__global__ void blt_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<BranchLessThanRecord> records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < records.len()) {
        auto const &full_record = records[idx];

        Rv64BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full_record.adapter);

        Rv64BranchLessThanCore core{VariableRangeChecker(rc_ptr, rc_bins)};
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full_record.core);
    } else {
        row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));
    }
}

extern "C" int _blt_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<BranchLessThanRecord> d_records,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(BranchLessThanCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height, 512);
    blt_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}

// M-GPUDEC (G2): tracegen from compact wire records + the per-exe operand
// table; materializes the same record structs in registers and calls the SAME
// fill methods as blt_tracegen.
template <typename RecordView>
__global__ void blt_tracegen_compact(
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
        RvrBranch2Compact const rec = records[idx];
        RvrOperandEntry const entry = rvr_operand_entry(operand_table, pc_base, rec.from_pc);

        BranchLessThanRecord full;
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
        Rv64BranchLessThanCore core{VariableRangeChecker(rc_ptr, rc_bins)};
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));
    }
}

extern "C" int _blt_tracegen_compact(
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
#ifdef OPENVM_RVR_CUDA_G2_ONLY
    return int(cudaErrorNotSupported);
#else
    assert(width == sizeof(BranchLessThanCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);
    blt_tracegen_compact<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_operand_table, pc_base, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
#endif
}

DEFINE_RVR_G2_TRACEGEN_LAUNCHER(
    _blt_tracegen_g2, BranchLessThanCols, blt_tracegen_compact, RvrBranch2Compact, 256,
    operand_table, pc_base, range_checker, range_checker_num_bins, timestamp_max_bits
)
