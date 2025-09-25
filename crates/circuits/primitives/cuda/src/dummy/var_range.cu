#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

template <typename T> struct DummyChipCols {
    T count;
    T value;
    T bits;
};

__global__ void var_range_dummy_tracegen(
    const uint32_t *data,
    Fp *trace,
    uint32_t *rc_count,
    size_t data_len,
    size_t rc_num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    VariableRangeChecker range_checker(rc_count, rc_num_bins);

    if (idx < data_len) {
        uint32_t value = data[idx];
        uint32_t bits = 32 - __clz(value);
        range_checker.add_count(value, bits);
        RowSlice row(trace + idx, data_len);
        COL_WRITE_VALUE(row, DummyChipCols, count, 1);
        COL_WRITE_VALUE(row, DummyChipCols, value, value);
        COL_WRITE_VALUE(row, DummyChipCols, bits, bits);
    }

    range_checker.merge(rc_count);
}

extern "C" int _var_range_dummy_tracegen(
    const uint32_t *d_data,
    Fp *d_trace,
    uint32_t *d_rc_count,
    size_t data_len,
    size_t rc_num_bins
) {
    auto [grid, block] = kernel_launch_params(data_len);
    var_range_dummy_tracegen<<<grid, block>>>(d_data, d_trace, d_rc_count, data_len, rc_num_bins);
    return CHECK_KERNEL();
}
