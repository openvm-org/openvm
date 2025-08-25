#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h" // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "rv32im/cores/beq.cuh"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BranchEqualCore = BranchEqualCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32BranchEqualCoreCols = BranchEqualCoreCols<T, RV32_REGISTER_NUM_LIMBS>;
using Rv32BranchEqualCoreRecord = BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct BranchEqualCols {
    Rv32BranchAdapterCols<T> adapter;
    Rv32BranchEqualCoreCols<T> core;
};

struct BranchEqualRecord {
    Rv32BranchAdapterRecord adapter;
    Rv32BranchEqualCoreRecord core;
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

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);

        Rv32BranchEqualCore core;
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
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(BranchEqualCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    beq_tracegen<<<grid, block>>>(d_trace, height, d_records, d_rc, rc_bins, timestamp_max_bits);
    return cudaGetLastError();
}
