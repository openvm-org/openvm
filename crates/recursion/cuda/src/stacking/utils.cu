#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "poly_common.cuh"
#include "stacking_blob.cuh"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdint.h>
#include <vector_types.h>

__device__ __forceinline__ uint2
binary_search(const uint32_t *__restrict__ arr, uint32_t idx, uint32_t size) {
    uint32_t lo = 0, hi = size;
    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (arr[mid] <= idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return {arr[lo], lo};
}

__global__ void stacked_slice_data_kernel(
    StackedSliceData *__restrict__ out,                      // [num_slices]
    const uint32_t *__restrict__ slice_offsets,              // [num_airs + num_commits - 1]
    const StackedTraceData *__restrict__ stacked_trace_data, // [num_airs + num_commits - 1]
    uint32_t num_airs,
    uint32_t num_commits,
    uint32_t num_slices,
    uint32_t n_stack,
    uint32_t l_skip
) {
    uint32_t global_slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_slice_idx >= num_slices) {
        return;
    }

    // slice_offsets contains starting global slice idx for each trace
    auto [global_slice_offset, record_idx] =
        binary_search(slice_offsets, global_slice_idx, num_airs + num_commits - 1);
    auto [commit_idx, start_col_idx, start_row_idx, log_height, width, need_rot] =
        stacked_trace_data[record_idx];

    uint32_t slice_height = 1 << std::max(log_height, l_skip);
    uint32_t trace_col_idx = global_slice_idx - global_slice_offset;
    uint32_t unbounded_row_idx = (start_row_idx + (trace_col_idx * slice_height));

    uint32_t log_stacked_height = l_skip + n_stack;
    uint32_t stacked_height = 1 << log_stacked_height;

    uint32_t row_idx = unbounded_row_idx & (stacked_height - 1);
    uint32_t col_idx = start_col_idx + (unbounded_row_idx >> log_stacked_height);
    int32_t n = (int32_t)log_height - (int32_t)l_skip;
    bool is_last_for_claim =
        (row_idx + slice_height == stacked_height) ||
        (commit_idx == 0 && record_idx + 1 == num_airs && trace_col_idx + 1 == width) ||
        (commit_idx != 0 && trace_col_idx + 1 == width);

    out[global_slice_idx] = {commit_idx, col_idx, row_idx, n, is_last_for_claim, need_rot};
}

__global__ void compute_coefficients_kernel(
    FpExt *__restrict__ coeff_terms,                 // [num_slices]
    uint64_t *__restrict__ coeff_term_keys,          // [num_slices]
    PolyPrecomputation *__restrict__ precomps,       // [num_slices]
    const StackedSliceData *__restrict__ slice_data, // [num_slices]
    const FpExt *__restrict__ u,
    const FpExt *__restrict__ r,
    const FpExt *__restrict__ lambda_pows,
    uint32_t num_commits,
    uint32_t num_slices,
    uint32_t n_stack,
    uint32_t l_skip
) {
    uint32_t global_slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_slice_idx >= num_slices) {
        return;
    }

    auto [commit_idx, col_idx, row_idx, n, is_last_for_claim, need_rot] =
        slice_data[global_slice_idx];
    uint32_t n_lift = n > 0 ? (uint32_t)n : 0;

    uint32_t num_bits = n_stack - n_lift;
    uint32_t b = row_idx >> (l_skip + n_lift);
    FpExt eq_bits = eval_eq_bits_ext(u + n_lift + 1, b, num_bits);
    FpExt ind = eval_in_uni_ext(u[0], n, l_skip);

    if (n < 0) {
        l_skip = (uint32_t)((int32_t)l_skip + n);
    }
    FpExt r_neg = n < 0 ? pow(r[0], 1 << -n) : FpExt(Fp::zero());
    const FpExt *r_n = n < 0 ? &r_neg : r;

    FpExt eq_prism = eval_eq_prism_ext(u, r_n, n_lift + 1, l_skip) * ind;
    FpExt rot_kernel_prism = eval_rot_kernel_prism_ext(u, r_n, n_lift + 1, l_skip) * ind;

    FpExt coeff_term = eq_bits * (lambda_pows[global_slice_idx << 1] * eq_prism);
    if (need_rot) {
        coeff_term =
            coeff_term + (eq_bits * (lambda_pows[(global_slice_idx << 1) + 1] * rot_kernel_prism));
    }
    coeff_terms[global_slice_idx] = coeff_term;
    coeff_term_keys[global_slice_idx] = ((uint64_t)commit_idx << 32) | (uint64_t)col_idx;
    precomps[global_slice_idx] = {eq_prism, rot_kernel_prism, eq_bits};
}

// ============================================================================
// LAUNCHERS
// ============================================================================

struct FpExtAdd {
    __device__ __forceinline__ FpExt operator()(const FpExt &a, const FpExt &b) const {
        return a + b;
    }
};

extern "C" int _stacked_slice_data(
    StackedSliceData *d_out,
    const uint32_t *d_slice_offsets,
    const StackedTraceData *d_stacked_trace_data,
    uint32_t num_airs,
    uint32_t num_commits,
    uint32_t num_slices,
    uint32_t n_stack,
    uint32_t l_skip
) {
    auto [grid, block] = kernel_launch_params(num_slices, 256);
    stacked_slice_data_kernel<<<grid, block>>>(
        d_out,
        d_slice_offsets,
        d_stacked_trace_data,
        num_airs,
        num_commits,
        num_slices,
        n_stack,
        l_skip
    );
    return CHECK_KERNEL();
}

extern "C" int _compute_coefficients_temp_bytes(
    FpExt *d_coeff_terms,
    uint64_t *d_coeff_term_keys,
    FpExt *d_coeffs,
    uint64_t *d_coeff_keys,
    uint32_t num_slices,
    size_t *d_num_coeffs,
    size_t *h_temp_bytes_out
) {
    size_t reduce_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(
        nullptr,
        reduce_storage_bytes,
        d_coeff_term_keys,
        d_coeff_keys,
        d_coeff_terms,
        d_coeffs,
        d_num_coeffs,
        FpExtAdd{},
        num_slices,
        cudaStreamPerThread
    );
    *h_temp_bytes_out = reduce_storage_bytes;
    return CHECK_KERNEL();
}

extern "C" int _compute_coefficients(
    FpExt *d_coeff_terms,
    uint64_t *d_coeff_term_keys,
    FpExt *d_coeffs,
    uint64_t *d_coeff_keys,
    PolyPrecomputation *d_precomps,
    const StackedSliceData *d_slice_data,
    const FpExt *d_u,
    const FpExt *d_r,
    const FpExt *d_lambda_pows,
    uint32_t num_commits,
    uint32_t num_slices,
    uint32_t n_stack,
    uint32_t l_skip,
    void *d_temp_buffer,
    size_t temp_bytes,
    size_t *d_num_coeffs
) {
    auto [grid, block] = kernel_launch_params(num_slices, 256);
    compute_coefficients_kernel<<<grid, block>>>(
        d_coeff_terms,
        d_coeff_term_keys,
        d_precomps,
        d_slice_data,
        d_u,
        d_r,
        d_lambda_pows,
        num_commits,
        num_slices,
        n_stack,
        l_skip
    );

    int ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    cub::DeviceReduce::ReduceByKey(
        d_temp_buffer,
        temp_bytes,
        d_coeff_term_keys,
        d_coeff_keys,
        d_coeff_terms,
        d_coeffs,
        d_num_coeffs,
        FpExtAdd{},
        num_slices,
        cudaStreamPerThread
    );
    return CHECK_KERNEL();
}
