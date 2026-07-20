#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SLL/SRL use u16 limbs (4 limbs of 16 bits) and the u16 ALU adapter.
using Rv64ShiftLogicalCore = ShiftLogicalCore<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftLogicalCoreRecord = ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftLogicalCoreCols = ShiftLogicalCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftLogicalCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64ShiftLogicalCoreCols<T> core;
};

struct ShiftLogicalRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64ShiftLogicalCoreRecord core;
};

__global__ void rv64_shift_logical_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftLogicalRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter =
            Rv64BaseAluRegU16Adapter(
                VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits);
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftLogicalCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftLogicalCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_shift_logical_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftLogicalRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftLogicalCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_logical_tracegen<<<grid, block, 0, stream>>>(
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
