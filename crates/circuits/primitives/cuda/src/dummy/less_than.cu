#include "launcher.cuh"
#include "primitives/histogram.cuh"
#include "primitives/less_than.cuh"
#include "primitives/trace_access.h"
#include <climits>

__global__ void cukernel_assert_less_than_tracegen(
    Fp *trace,
    uint32_t trace_height,
    uint2 *pairs,
    uint32_t max_bits,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height)
        return;

    uint2 xy = pairs[row_idx];
    uint32_t x = xy.x;
    uint32_t y = xy.y;

    trace[row_idx] = Fp(x);
    trace[trace_height + row_idx] = Fp(y);
    trace[2 * trace_height + row_idx] = Fp::one();

    {
        VariableRangeChecker range_checker(rc_count, rc_num_bins);
        Fp *decomp = trace + 3 * trace_height + row_idx;

        AssertLessThan::generate_subrow(
            range_checker, max_bits, x, y, aux_len, RowSlice(decomp, trace_height)
        );
    }
}

__global__ void cukernel_less_than_tracegen(
    Fp *trace,
    size_t trace_height,
    uint2 *pairs,
    uint32_t max_bits,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height)
        return;

    uint2 xy = pairs[row_idx];
    uint32_t x = xy.x;
    uint32_t y = xy.y;

    trace[row_idx] = Fp(x);
    trace[trace_height + row_idx] = Fp(y);

    {
        VariableRangeChecker range_checker(rc_count, rc_num_bins);
        Fp *out = trace + 2 * trace_height + row_idx;
        Fp *decomp = trace + 3 * trace_height + row_idx;

        IsLessThan::generate_subrow(
            range_checker, max_bits, x, y, aux_len, RowSlice(decomp, trace_height), out
        );
    }
}

__global__ void cukernel_less_than_array_tracegen(
    Fp *trace,
    uint32_t trace_height,
    uint32_t *pairs,
    uint32_t max_bits,
    uint32_t array_len,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= trace_height)
        return;

    Fp *x_ptr = trace + row_idx;
    Fp *y_ptr = trace + trace_height * array_len + row_idx;
    RowSlice x(x_ptr, trace_height);
    RowSlice y(y_ptr, trace_height);

    for (uint32_t i = 0; i < array_len; i++) {
        x[i] = Fp(pairs[row_idx * 2 * array_len + i]);
        y[i] = Fp(pairs[row_idx * 2 * array_len + i + array_len]);
    }

    {
        VariableRangeChecker range_checker(rc_count, rc_num_bins);
        Fp *out = trace + (2 * array_len) * trace_height + row_idx;
        Fp *diff_marker = trace + (2 * array_len + 1) * trace_height + row_idx;
        Fp *diff_inv = trace + (3 * array_len + 1) * trace_height + row_idx;
        Fp *lt_decomp = trace + (3 * array_len + 2) * trace_height + row_idx;

        IsLessThanArray::generate_subrow(
            range_checker,
            max_bits,
            x,
            y,
            array_len,
            aux_len,
            RowSlice(diff_marker, trace_height),
            diff_inv,
            RowSlice(lt_decomp, trace_height),
            out
        );
    }
}

extern "C" int _assert_less_than_tracegen(
    Fp *trace,
    size_t trace_height,
    uint32_t *pairs,
    uint32_t max_bits,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    auto [grid, block] = kernel_launch_params(trace_height);
    cukernel_assert_less_than_tracegen<<<grid, block>>>(
        trace, trace_height, (uint2 *)pairs, max_bits, aux_len, rc_count, rc_num_bins
    );
    return CHECK_KERNEL();
}

extern "C" int _less_than_tracegen(
    Fp *trace,
    size_t trace_height,
    uint32_t *pairs,
    uint32_t max_bits,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    auto [grid, block] = kernel_launch_params(trace_height);
    cukernel_less_than_tracegen<<<grid, block>>>(
        trace, trace_height, (uint2 *)pairs, max_bits, aux_len, rc_count, rc_num_bins
    );
    return CHECK_KERNEL();
}

extern "C" int _less_than_array_tracegen(
    Fp *trace,
    size_t trace_height,
    uint32_t *pairs,
    uint32_t max_bits,
    uint32_t array_len,
    uint32_t aux_len,
    uint32_t *rc_count,
    uint32_t rc_num_bins
) {
    auto [grid, block] = kernel_launch_params(trace_height);
    cukernel_less_than_array_tracegen<<<grid, block>>>(
        trace, trace_height, pairs, max_bits, array_len, aux_len, rc_count, rc_num_bins
    );
    return CHECK_KERNEL();
}