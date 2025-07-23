#include "adapters/mul.cuh"
#include "constants.h"
#include "cores/multiplication.cuh"
#include "histogram.cuh"
#include "launcher.cuh"
#include "trace_access.h"

using namespace riscv;

// Concrete type aliases for 32-bit
using Rv32MultiplicationCoreRecord = MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS>;
using Rv32MultiplicationCore = MultiplicationCore<RV32_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv32MultiplicationCoreCols = MultiplicationCoreCols<T, RV32_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv32MultiplicationCols {
    Rv32MultAdapterCols<T> adapter;
    Rv32MultiplicationCoreCols<T> core;
};

struct Rv32MultiplicationRecord {
    Rv32MultAdapterRecord adapter;
    Rv32MultiplicationCoreRecord core;
};

__global__ void mul_tracegen(
    Fp *d_trace,
    size_t height,
    uint8_t *d_records,
    size_t num_records,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    if (idx < num_records) {
        auto rec = reinterpret_cast<Rv32MultiplicationRecord *>(d_records)[idx];

        Rv32MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bins), 
            timestamp_max_bits
        );
        adapter.fill_trace_row(row, rec.adapter);

        RangeTupleChecker<2> range_tuple_checker(
            d_range_tuple_ptr, (uint32_t[2]){range_tuple_sizes.x, range_tuple_sizes.y}
        );
        Rv32MultiplicationCore core(range_tuple_checker);
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32MultiplicationCols, core)), rec.core);
    } else {
        row.fill_zero(0, sizeof(Rv32MultiplicationCols<uint8_t>));
    }
}

extern "C" int _mul_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    uint8_t *d_records,
    size_t record_len,
    uint32_t *d_range_checker_ptr,
    size_t range_checker_bins,
    uint32_t *d_range_tuple_ptr,
    uint2 range_tuple_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height * sizeof(Rv32MultiplicationRecord) >= record_len);
    assert(width == sizeof(Rv32MultiplicationCols<uint8_t>));
    size_t num_records = record_len / sizeof(Rv32MultiplicationRecord);
    auto [grid, block] = kernel_launch_params(height);

    mul_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        num_records,
        d_range_checker_ptr,
        range_checker_bins,
        d_range_tuple_ptr,
        range_tuple_sizes,
        timestamp_max_bits
    );
    return cudaGetLastError();
}