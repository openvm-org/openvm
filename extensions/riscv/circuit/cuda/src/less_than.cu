#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64LessThanCoreRecord = LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64LessThanCore = LessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64LessThanCoreCols =
    LessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct LessThanCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
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

        auto adapter = Rv64BaseAluRegU16Adapter(
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
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_less_than_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
