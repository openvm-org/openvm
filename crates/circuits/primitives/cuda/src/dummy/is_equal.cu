#include "launcher.cuh"
#include "primitives/is_equal.cuh"
#include "primitives/trace_access.h"

__global__ void cukernel_isequal_tracegen(Fp *output, Fp *inputs_x, Fp *inputs_y, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    Fp *out = output + n + tid;
    Fp *inv = output + tid;
    IsEqual::generate_subrow(inputs_x[tid], inputs_y[tid], inv, out);
}

// inputs_x and inputs_y, could be part of output (trace) not necessary due to testing
__global__ void cukernel_isequal_array_tracegen(
    Fp *output,
    Fp *inputs_x,
    Fp *inputs_y,
    uint32_t array_len,
    uint32_t n
) {
    // shape of inputs_x and inputs_y is (array_len, n), to satisfy the interface
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    // Create row slices for x and y inputs
    Fp *x_ptr = inputs_x + tid;
    Fp *y_ptr = inputs_y + tid;
    RowSlice x(x_ptr, n);
    RowSlice y(y_ptr, n);

    // Set up output and diff_inv_marker pointers
    RowSlice diff_inv_marker(output + tid, n);
    Fp *out = output + tid + array_len * n;

    // Generate trace using array version
    IsEqualArray::generate_subrow(array_len, x, y, diff_inv_marker, out);
}

extern "C" int _isequal_tracegen(Fp *output, Fp *inputs_x, Fp *inputs_y, uint32_t n) {
    auto [grid, block] = kernel_launch_params(n);
    cukernel_isequal_tracegen<<<grid, block>>>(output, inputs_x, inputs_y, n);
    return CHECK_KERNEL();
}

extern "C" int _isequal_array_tracegen(
    Fp *output,
    Fp *inputs_x,
    Fp *inputs_y,
    uint32_t array_len,
    uint32_t n
) {
    auto [grid, block] = kernel_launch_params(n);
    cukernel_isequal_array_tracegen<<<grid, block>>>(output, inputs_x, inputs_y, array_len, n);
    return CHECK_KERNEL();
}