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
        uint32_t n = idx + 1;
        uint32_t max_bits = 31 - __clz(n);
        uint32_t two_to_max_bits = 1U << max_bits;
        uint32_t value = n - two_to_max_bits;
        uint32_t mult_val = count[idx] + (cpu_count ? cpu_count[idx] : 0);

        bool is_selector_zero = (value + 1 == two_to_max_bits);
        Fp selector = Fp(value) + Fp::one() - Fp(two_to_max_bits);
        Fp selector_inverse = is_selector_zero ? Fp::zero() : inv(selector);
        Fp is_not_wrap = is_selector_zero ? Fp::zero() : Fp::one();

        trace[idx] = Fp(value);                                                    
        trace[idx + num_bins] = Fp(max_bits);
        trace[idx + 2 * num_bins] = Fp(two_to_max_bits);
        trace[idx + 3 * num_bins] = selector_inverse;
        trace[idx + 4 * num_bins] = is_not_wrap;
        trace[idx + 5 * num_bins] = Fp(mult_val);
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
