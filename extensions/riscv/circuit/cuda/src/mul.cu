#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32im/adapters/mul.cuh"
#include "rv32im/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv64MultiplicationCoreRecord = MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv64MultiplicationCore = MultiplicationCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64MultiplicationCoreCols = MultiplicationCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64MultiplicationCols {
    Rv64MultAdapterCols<T> adapter;
    Rv64MultiplicationCoreCols<T> core;
};

struct Rv64MultiplicationRecord {
    Rv64MultAdapterRecord adapter;
    Rv64MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];

        Rv64MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv64MultiplicationCore core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv64MultiplicationCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv64MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv64MultiplicationRecord> d_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv64MultiplicationCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    mul_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}