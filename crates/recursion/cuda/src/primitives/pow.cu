#include "fp.h"
#include "launcher.cuh"

#include <cassert>
#include <cstddef>
#include <cstdint>

const uint32_t N = 32;

__global__ void pow_checker_tracegen(
    const uint32_t *pow_count,
    const uint32_t *range_count,
    const uint32_t *cpu_pow_count,
    const uint32_t *cpu_range_count,
    Fp *trace
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        uint32_t pow_mult = pow_count[idx] + (cpu_pow_count ? cpu_pow_count[idx] : 0);
        uint32_t range_mult = range_count[idx] + (cpu_range_count ? cpu_range_count[idx] : 0);
        trace[idx] = Fp(idx);
        trace[idx + N] = idx != 31 ? Fp(1 << idx) : Fp(1 << 30) * Fp(2);
        trace[idx + 2 * N] = Fp(pow_mult);
        trace[idx + 3 * N] = Fp(range_mult);
    }
}

extern "C" int _pow_checker_tracegen(
    const uint32_t *d_pow_count,
    const uint32_t *d_range_count,
    const uint32_t *d_cpu_pow_count,
    const uint32_t *d_cpu_range_count,
    Fp *d_trace,
    size_t n
) {
    assert(n == N);
    auto [grid, block] = kernel_launch_params(N);
    pow_checker_tracegen<<<grid, block>>>(
        d_pow_count, d_range_count, d_cpu_pow_count, d_cpu_range_count, d_trace
    );
    return CHECK_KERNEL();
}
