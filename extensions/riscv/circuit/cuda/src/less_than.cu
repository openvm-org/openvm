#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_u16.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// Pattern B u16 less_than: 4 u16 cells per side with LIMB_BITS = 16.
constexpr size_t RV64_LESS_THAN_NUM_LIMBS = BLOCK_FE_WIDTH;
constexpr size_t RV64_LESS_THAN_LIMB_BITS = 16;

using Rv64LessThanCoreRecord =
    LessThanCoreRecord<RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>;
using Rv64LessThanCore = LessThanCore<RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>;
template <typename T>
using Rv64LessThanCoreCols =
    LessThanCoreCols<T, RV64_LESS_THAN_NUM_LIMBS, RV64_LESS_THAN_LIMB_BITS>;

template <typename T> struct LessThanCols {
    Rv64BaseAluAdapterU16Cols<T> adapter;
    Rv64LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv64BaseAluAdapterU16Record adapter;
    Rv64LessThanCoreRecord core;
};

__global__ void rv64_less_than_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<LessThanRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64BaseAluAdapterU16(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64LessThanCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(LessThanCols<uint8_t>));
    }
}

extern "C" int _rv64_less_than_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanRecord> d_records,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv64_less_than_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
