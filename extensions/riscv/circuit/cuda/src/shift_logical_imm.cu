#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/shift_logical_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SLLI/SRLI use u16 limbs (4 limbs of 16 bits) and the single-read ADDI adapter; the shift
// amount lives in the core record and the immediate operand is reconstructed from the core's
// marker columns.
using Rv64ShiftLogicalImmCore = ShiftLogicalImmCore<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftLogicalImmCoreRecord = ShiftLogicalImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftLogicalImmCoreCols = ShiftLogicalImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftLogicalImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64ShiftLogicalImmCoreCols<T> core;
};

struct ShiftLogicalImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64ShiftLogicalImmCoreRecord core;
};

static_assert(sizeof(ShiftLogicalImmRecord) == 44);
static_assert(offsetof(ShiftLogicalImmRecord, core) == 32);

__global__ void shift_logical_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftLogicalImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluImmU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftLogicalImmCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftLogicalImmCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _shift_logical_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftLogicalImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftLogicalImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    shift_logical_imm_tracegen<<<grid, block, 0, stream>>>(
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
