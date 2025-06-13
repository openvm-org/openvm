#include "fp.h"
#include "launcher.cuh"

__global__ void bitwise_op_lookup_tracegen(const uint32_t *count, Fp *trace, uint32_t num_rows) {
    uint32_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < num_rows) {
        trace[row_idx] = Fp(count[row_idx]);
        trace[row_idx + num_rows] = Fp(count[row_idx + num_rows]);
    }
}

extern "C" int _bitwise_op_lookup_tracegen(
    const uint32_t *d_count,
    Fp *d_trace,
    uint32_t num_bits
) {
    uint32_t num_rows = 1 << (2 * num_bits);
    auto [grid, block] = kernel_launch_params(num_rows);
    bitwise_op_lookup_tracegen<<<grid, block>>>(d_count, d_trace, num_rows);
    return cudaGetLastError();
}
