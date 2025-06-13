#include "histogram.cuh"
#include "launcher.cuh"

// Requests a range check for every even index pair, and an XOR for every odd
// index pair.
__global__ void send_bitwise_operation_lookups(
    uint32_t *d_count,
    uint32_t num_bits,
    uint2 *pairs,
    size_t num_pairs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) {
        return;
    }

    BitwiseOperationLookup bitwise(d_count, num_bits);
    uint2 xy = pairs[idx];
    uint32_t x = xy.x;
    uint32_t y = xy.y;

    if (idx % 2 == 0) {
        bitwise.add_range(x, y);
    } else {
        bitwise.add_xor(x, y);
    }
}

extern "C" int _send_bitwise_operation_lookups(
    uint32_t *d_count,
    uint32_t num_bits,
    uint32_t *pairs,
    size_t num_pairs
) {
    auto [grid, block] = kernel_launch_params(num_pairs);
    send_bitwise_operation_lookups<<<grid, block>>>(d_count, num_bits, (uint2 *)pairs, num_pairs);
    return cudaGetLastError();
}
