#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w.cuh"
#include "riscv/cores/shift.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64.
using Rv64ShiftWCoreRecord = ShiftCoreRecord<RV64_WORD_NUM_LIMBS>;
using Rv64ShiftWLogicalCore = ShiftLogicalCore<RV64_WORD_NUM_LIMBS>;
using Rv64ShiftWArithmeticRightCore = ShiftArithmeticRightCore<RV64_WORD_NUM_LIMBS>;
template <typename T>
using Rv64ShiftWLogicalCoreCols = ShiftLogicalCoreCols<T, RV64_WORD_NUM_LIMBS>;
template <typename T>
using Rv64ShiftWArithmeticRightCoreCols = ShiftArithmeticRightCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct ShiftWLogicalCols {
    Rv64BaseAluWAdapterCols<T> adapter;
    Rv64ShiftWLogicalCoreCols<T> core;
};

template <typename T> struct ShiftWArithmeticRightCols {
    Rv64BaseAluWAdapterCols<T> adapter;
    Rv64ShiftWArithmeticRightCoreCols<T> core;
};

struct ShiftWRecord {
    Rv64BaseAluWAdapterRecord adapter;
    Rv64ShiftWCoreRecord core;
};

__global__ void rv64_shift_w_logical_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<ShiftWRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWLogicalCore(
            BitwiseOperationLookup(lookup_ptr),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWLogicalCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(ShiftWLogicalCols<uint8_t>));
    }
}

__global__ void rv64_shift_w_arithmetic_right_tracegen(
    Fp *trace,
    size_t height,
    DeviceBufferConstView<ShiftWRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWArithmeticRightCore(
            BitwiseOperationLookup(lookup_ptr),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWArithmeticRightCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(ShiftWArithmeticRightCols<uint8_t>));
    }
}

extern "C" int _rv64_shift_w_logical_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftWLogicalCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_w_logical_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_shift_w_arithmetic_right_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftWArithmeticRightCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_w_arithmetic_right_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
