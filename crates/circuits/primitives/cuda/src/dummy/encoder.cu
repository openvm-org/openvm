#include "launcher.cuh"
#include "primitives/encoder.cuh"
#include <cassert>

__global__ void cukernel_encoder_tracegen(
    Fp *trace,
    uint32_t num_flags,
    uint32_t max_degree,
    bool reserve_invalid,
    uint32_t k
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_flags)
        return;

    Encoder encoder(num_flags, max_degree, reserve_invalid, k);
    encoder.write_flag_pt(RowSlice(trace + idx, num_flags), idx);
}

// Given the Encoder parameters, generates a size-(num_flags x k) trace
// with row idx being the k-dimentional point corresponding to idx. Note
// that for this dummy trace we do not impose that the number of rows
// is an exponent of 2.
extern "C" int _encoder_tracegen(
    Fp *trace,
    uint32_t num_flags,
    uint32_t max_degree,
    bool reserve_invalid,
    uint32_t expected_k
) {
    auto [grid, block] = kernel_launch_params(num_flags);
    uint32_t k = compute_k(num_flags, max_degree, reserve_invalid);
    assert(k == expected_k);

    cukernel_encoder_tracegen<<<grid, block>>>(trace, num_flags, max_degree, reserve_invalid, k);

    return CHECK_KERNEL();
}
