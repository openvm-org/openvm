#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/blt.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

// Pattern B (u16): each register read is BLOCK_FE_WIDTH=4 u16 cells with LIMB_BITS=16.
constexpr size_t RV64_BRANCH_NUM_LIMBS = BLOCK_FE_WIDTH;
constexpr size_t RV64_BRANCH_LIMB_BITS = 16;

using Rv64BranchLessThanCoreRecord =
    BranchLessThanCoreRecord<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>;
using Rv64BranchLessThanCore = BranchLessThanCore<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>;
template <typename T>
using Rv64BranchLessThanCoreCols =
    BranchLessThanCoreCols<T, RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>;

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

    auto [grid, block] = kernel_launch_params(height);
    blt_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
