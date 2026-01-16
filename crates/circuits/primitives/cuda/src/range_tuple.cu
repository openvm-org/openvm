#include "fp.h"
#include "launcher.cuh"
#include <stdio.h>

__global__ void range_tuple_checker_tracegen(
    const uint32_t *count,
    const uint32_t *cpu_count,
    Fp *trace,
    const uint32_t *sizes,
    uint32_t num_dims,
    size_t num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        uint32_t tmp_idx = idx;
        for (int i = num_dims - 1; i >= 0; i--) {
            trace[idx + num_bins * i] = tmp_idx % sizes[i];
            tmp_idx /= sizes[i];
        }
        
        int32_t first_nonzero = 0;
        for (uint32_t i = 0; i < num_dims; i++) {
            // tuple[i] = trace[idx + num_bins * i]
            if (trace[idx + num_bins * i]) first_nonzero = i;
        }
        for (int32_t i = 0; i < first_nonzero; i++) {
            // is_first[i] = trace[idx + num_bins * (num_dims + i)]
            trace[idx + num_bins * (num_dims + i)] = 0;
        }
        for (int32_t i = first_nonzero; i < num_dims - 1; i++) {
            // is_first[i] = trace[idx + num_bins * (num_dims + i)]
            trace[idx + num_bins * (num_dims + i)] = 1;
        }
        uint32_t mult = count[idx] + (cpu_count ? cpu_count[idx] : 0);
        trace[idx + num_bins * (num_dims * 2 - 1)] = Fp(mult);
    }
}

extern "C" int _range_tuple_checker_tracegen(
    const uint32_t *d_count,
    const uint32_t *d_cpu_count,
    Fp *d_trace,
    const uint32_t *d_sizes,
    uint32_t num_dims,
    size_t num_bins
) {
    auto [grid, block] = kernel_launch_params(num_bins);
    range_tuple_checker_tracegen<<<grid, block>>>(
        d_count, d_cpu_count, d_trace, d_sizes, num_dims, num_bins
    );
    return CHECK_KERNEL();
}
