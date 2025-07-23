#include "adapters/branch.cuh" // Rv32BranchAdapterCols, Rv32BranchAdapterRecord, Rv32BranchAdapter
#include "constants.h"         // RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS
#include "cores/branch_less_than.cuh"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32BranchLessThanCoreRecord = BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32BranchLessThanCore = BranchLessThanCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32BranchLessThanCoreCols = BranchLessThanCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct BranchLessThanCols {
    Rv32BranchAdapterCols<T> adapter;
    Rv32BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv32BranchAdapterRecord adapter;
    Rv32BranchLessThanCoreRecord core;
};

__global__ void blt_tracegen(
    Fp *trace,
    size_t height,
    uint8_t *records,
    size_t num_records,
    uint32_t *rc_ptr,
    uint32_t rc_bins,
    uint32_t *bw_ptr,
    uint32_t bw_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);

    if (idx < num_records) {
        auto full = reinterpret_cast<BranchLessThanRecord *>(records)[idx];

        Rv32BranchAdapter adapter(VariableRangeChecker(rc_ptr, rc_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, full.adapter);

        Rv32BranchLessThanCore core(BitwiseOperationLookup(bw_ptr, bw_bits));
        core.fill_trace_row(row.slice_from(COL_INDEX(BranchLessThanCols, core)), full.core);
    } else {
        row.fill_zero(0, sizeof(BranchLessThanCols<uint8_t>));
    }
}

extern "C" int _blt_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_rc,
    uint32_t rc_bins,
    uint32_t *d_bw,
    uint32_t bw_bits,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(BranchLessThanRecord) >= record_len);
    assert(width == sizeof(BranchLessThanCols<uint8_t>));

    auto [grid, block] = kernel_launch_params(height);
    blt_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        record_len / sizeof(BranchLessThanRecord),
        d_rc,
        rc_bins,
        d_bw,
        bw_bits,
        timestamp_max_bits
    );
    return cudaGetLastError();
}
