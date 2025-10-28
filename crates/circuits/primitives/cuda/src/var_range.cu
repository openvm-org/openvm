#include "fp.h"
#include "launcher.cuh"

__global__ void range_checker_tracegen(
    const uint32_t *count,
    const uint32_t *cpu_count,
    Fp *trace,
    size_t num_bins
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        trace[idx] = Fp(count[idx] + (cpu_count ? cpu_count[idx] : 0));
    }
}

extern "C" int _range_checker_tracegen(
    const uint32_t *d_count,
    const uint32_t *d_cpu_count,
    Fp *d_trace,
    size_t num_bins
) {
    auto [grid, block] = kernel_launch_params(num_bins);
    range_checker_tracegen<<<grid, block>>>(d_count, d_cpu_count, d_trace, num_bins);
    return CHECK_KERNEL();
}
