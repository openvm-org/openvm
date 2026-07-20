#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLLW/SRLW use the u16 shift-logical core (RV64_WORD_U16_LIMBS limbs of 16 bits) over the low
// 32-bit word and the u16 W adapter.
using Rv64ShiftWLogicalCore = ShiftLogicalCore<RV64_WORD_U16_LIMBS, U16_BITS>;
using Rv64ShiftWLogicalCoreRecord = ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWLogicalCoreCols = ShiftLogicalCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWLogicalCols {
    Rv64BaseAluWRegU16AdapterCols<T> adapter;
    Rv64ShiftWLogicalCoreCols<T> core;
};

struct ShiftWLogicalRecord {
    Rv64BaseAluWRegU16AdapterRecord adapter;
    Rv64ShiftWLogicalCoreRecord core;
};

__global__ void rv64_shift_w_logical_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<ShiftWLogicalRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWRegU16Adapter(
            VariableRangeChecker(range_ptr, range_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWLogicalCore(VariableRangeChecker(range_ptr, range_bins));
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWLogicalCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(ShiftWLogicalCols<uint8_t>));
    }
}

extern "C" int _rv64_shift_w_logical_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWLogicalRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftWLogicalCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_w_logical_tracegen<<<grid, block, 0, stream>>>(
        d_trace, height, d_records, d_range_checker, range_checker_num_bins, timestamp_max_bits
    );
    return CHECK_KERNEL();
}
