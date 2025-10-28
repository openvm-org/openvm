#include "launcher.cuh"
#include "primitives/is_zero.cuh"

__global__ void cukernel_iszero_tracegen(Fp *output, Fp *inputs, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    Fp *out = output + n + tid;
    Fp *inv = output + tid;
    IsZero::generate_subrow(inputs[tid], inv, out);
}

extern "C" int _iszero_tracegen(Fp *output, Fp *inputs, uint32_t n) {
    auto [grid, block] = kernel_launch_params(n);
    cukernel_iszero_tracegen<<<grid, block>>>(output, inputs, n);
    return CHECK_KERNEL();
}