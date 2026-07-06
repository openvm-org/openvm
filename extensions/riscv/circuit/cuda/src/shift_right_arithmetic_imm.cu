#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_u16_imm.cuh"
#include "riscv/cores/shift_right_arithmetic.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SRAI uses u16 limbs (4 limbs of 16 bits) and the immediate-only u16 ALU adapter.
using Rv64ShiftRightArithmeticCoreRecord = ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftRightArithmeticCore = ShiftRightArithmeticCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftRightArithmeticCoreCols = ShiftRightArithmeticCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftRightArithmeticImmCols {
    Rv64BaseAluU16ImmAdapterCols<T> adapter;
    Rv64ShiftRightArithmeticCoreCols<T> core;
};

struct ShiftRightArithmeticImmRecord {
    Rv64BaseAluU16ImmAdapterRecord adapter;
    Rv64ShiftRightArithmeticCoreRecord core;
};

__global__ void rv64_shift_right_arithmetic_imm_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluU16ImmAdapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftRightArithmeticCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(
            row.slice_from(COL_INDEX(ShiftRightArithmeticImmCols, core)), rec.core
        );
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_shift_right_arithmetic_imm_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRightArithmeticImmRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightArithmeticImmCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_right_arithmetic_imm_tracegen<<<grid, block, 0, stream>>>(
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
