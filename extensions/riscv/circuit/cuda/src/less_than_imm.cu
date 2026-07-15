#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/less_than_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLTI/SLTIU use u16 limbs and the single-read immediate adapter.
using Rv64LessThanImmCoreRecord = LessThanImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64LessThanImmCore = LessThanImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64LessThanImmCoreCols = LessThanImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct LessThanImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64LessThanImmCoreCols<T> core;
};

struct LessThanImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64LessThanImmCoreRecord core;
};

static_assert(sizeof(LessThanImmRecord) == 48);
static_assert(offsetof(LessThanImmRecord, core) == 32);

__global__ void less_than_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanImmRecord> records,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &record = records[idx];

        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_checker_ptr, range_checker_num_bins),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        auto core =
            Rv64LessThanImmCore(VariableRangeChecker(range_checker_ptr, range_checker_num_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(LessThanImmCols, core)), record.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _less_than_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<LessThanImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(LessThanImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    less_than_imm_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
