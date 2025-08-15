#pragma once

#include <algorithm>
#include <cuda_runtime.h>

static const size_t MAX_THREADS = 1024;
static const size_t WARP_SIZE = 32;

inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

inline std::pair<dim3, dim3> kernel_launch_params(
    size_t count,
    size_t threads_per_block = MAX_THREADS
) {
    size_t block = std::min(count, threads_per_block);
    size_t grid = div_ceil(count, block);
    return std::make_pair(dim3(grid, 1, 1), dim3(block, 1, 1));
}

inline std::pair<dim3, dim3> kernel_launch_2d_params(size_t x, size_t y) {
    dim3 block = dim3(std::min(x, WARP_SIZE), std::min(y, WARP_SIZE));
    dim3 grid = dim3(div_ceil(x, block.x), div_ceil(y, block.y));
    return std::make_pair(grid, block);
}

#define CUDA_OK(expr) do {                                  \
    cudaError_t err = expr;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    }                                                       \
} while(0)

