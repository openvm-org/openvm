#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64.
using Rv64MulWCoreRecord = MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS>;
using Rv64MulWCore = MultiplicationCore<RV64_WORD_NUM_LIMBS>;
template <typename T> using Rv64MulWCoreCols = MultiplicationCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct Rv64MulWCols {
    Rv64MultWAdapterCols<T> adapter;
    Rv64MulWCoreCols<T> core;
};

struct Rv64MulWRecord {
    Rv64MultWAdapterRecord adapter;
    Rv64MulWCoreRecord core;
};

__global__ void mul_w_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64MulWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64MultWAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins),
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_num_bits),
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv64MulWCore core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MulWCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MulWCols<uint8_t>));
    }
}

extern "C" int _rv64_mul_w_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64MulWRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits,
    cudaStream_t stream
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv64MulWCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    mul_w_tracegen<<<grid, block, 0, stream>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}
