#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu.cuh"
#include "riscv/cores/shift.cuh"

using namespace riscv;
using namespace program;

// Concrete type aliases for 64-bit
using Rv64ShiftCoreRecord = ShiftCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64ShiftLeftCore = ShiftLeftCore<RV64_REGISTER_NUM_LIMBS>;
using Rv64ShiftRightCore = ShiftRightCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64ShiftLeftCoreCols = ShiftLeftCoreCols<T, RV64_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64ShiftRightCoreCols = ShiftRightCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct ShiftLeftCols {
    Rv64BaseAluAdapterCols<T> adapter;
    Rv64ShiftLeftCoreCols<T> core;
};

template <typename T> struct ShiftRightCols {
    Rv64BaseAluAdapterCols<T> adapter;
    Rv64ShiftRightCoreCols<T> core;
};

struct ShiftRecord {
    Rv64BaseAluAdapterRecord adapter;
    Rv64ShiftCoreRecord core;
};

__global__ void rv64_shift_left_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftLeftCore(
            BitwiseOperationLookup(lookup_ptr),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftLeftCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

__global__ void rv64_shift_right_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftRightCore(
            BitwiseOperationLookup(lookup_ptr),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftRightCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_shift_left_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftLeftCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_left_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}

extern "C" int _rv64_shift_right_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert(width == sizeof(ShiftRightCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    rv64_shift_right_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
