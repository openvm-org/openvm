#include <cstddef>
#include <cstdint>

#include "fp.h"
#include "launcher.cuh"
#include "primitives/trace_access.h"

template <typename T> struct DeferralCircuitCountCols {
    T is_valid;
    T row_idx;
    T mult;
};

__global__ void deferral_count_tracegen(
    Fp *trace,
    const size_t height,
    const uint32_t *count,
    const size_t num_def_circuits
) {
    const uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= height) {
        return;
    }
    RowSlice row(trace + row_idx, height);

    if (row_idx >= num_def_circuits) {
        row.fill_zero(0, sizeof(DeferralCircuitCountCols<uint8_t>));
        return;
    }

    COL_WRITE_VALUE(row, DeferralCircuitCountCols, is_valid, Fp::one());
    COL_WRITE_VALUE(row, DeferralCircuitCountCols, row_idx, row_idx);
    COL_WRITE_VALUE(row, DeferralCircuitCountCols, mult, count[row_idx]);
}

extern "C" int _deferral_count_tracegen(
    Fp *d_trace,
    const size_t height,
    const uint32_t *d_count,
    const size_t num_def_circuits
) {
    auto [grid, block] = kernel_launch_params(height);
    deferral_count_tracegen<<<grid, block>>>(d_trace, height, d_count, num_def_circuits);
    return CHECK_KERNEL();
}
