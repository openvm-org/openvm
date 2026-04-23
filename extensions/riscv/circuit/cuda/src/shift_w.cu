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
using Rv64ShiftWCore = ShiftCore<RV64_WORD_NUM_LIMBS>;
template <typename T> using Rv64ShiftWCoreCols = ShiftCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct ShiftWCols {
    Rv64BaseAluWAdapterCols<T> adapter;
    Rv64ShiftWCoreCols<T> core;
};

struct ShiftWRecord {
    Rv64BaseAluWAdapterRecord adapter;
    Rv64ShiftWCoreRecord core;
};

__global__ void shift_w_tracegen(
    Fp *trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWRecord> records,
    uint32_t *range_ptr,
    uint32_t range_bins,
    uint32_t *lookup_ptr,
    uint32_t lookup_bits,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(trace + idx, height);
    if (idx < records.len()) {
        auto const &rec = records[idx];
        auto adapter = Rv64BaseAluWAdapter(
            VariableRangeChecker(range_ptr, range_bins),
            BitwiseOperationLookup(lookup_ptr, lookup_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);
        auto core = Rv64ShiftWCore(
            BitwiseOperationLookup(lookup_ptr, lookup_bits),
            VariableRangeChecker(range_ptr, range_bins)
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(ShiftWCols, core)), rec.core);
    } else {
        row.fill_zero(0, width);
    }
}

extern "C" int _rv64_shift_w_tracegen(
    Fp *__restrict__ d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<ShiftWRecord> d_records,
    uint32_t *__restrict__ d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *__restrict__ d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(ShiftWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height, 512);

    shift_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        width,
        d_records,
        d_range_checker,
        range_checker_num_bins,
        d_bitwise_lookup,
        bitwise_num_bits,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
