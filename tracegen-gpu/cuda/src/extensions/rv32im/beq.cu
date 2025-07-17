#include "adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "constants.h"         // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "cores/beq.cuh"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

static constexpr uint8_t BEQ = 0;

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
    uint8_t *records,
    size_t num_records,
    uint32_t *rc_ptr,
    uint32_t rc_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < num_records) {
        auto full = reinterpret_cast<BranchEqualRecord *>(records)[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins));
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
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_rc,
    uint32_t rc_bins
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BranchEqualRecord) >= record_len);
    assert(width == sizeof(BranchEqualCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    beq_tracegen<<<grid, block>>>(
        d_trace, height, d_records, record_len / sizeof(BranchEqualRecord), d_rc, rc_bins
    );
    return cudaGetLastError();
}
