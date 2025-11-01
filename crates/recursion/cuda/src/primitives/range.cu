#include "fp.h"
#include "launcher.cuh"

#include <cstddef>
#include <cstdint>

__global__ void range_checker_tracegen(const uint32_t *count, Fp *trace, size_t height) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height) {
        trace[idx] = Fp(idx);
        trace[idx + height] = Fp(count[idx]);
    }
}

extern "C" int _range_checker_recursion_tracegen(
    const uint32_t *d_count,
    Fp *d_trace,
    size_t num_bits
) {
    size_t height = 1 << num_bits;
    auto [grid, block] = kernel_launch_params(height);
    range_checker_tracegen<<<grid, block>>>(d_count, d_trace, height);
    return CHECK_KERNEL();
}
