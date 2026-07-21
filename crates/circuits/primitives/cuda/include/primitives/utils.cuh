#pragma once

#include "primitives/constants.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>

// Kernel to fill size-n buffer with a certain value
template <typename T> __global__ void fill_buffer(T *buffer, T value, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        buffer[tid] = value;
    }
}

// Convert 2 bytes to a u16 in little endian order
// **SAFETY**: b has to be at least 2 bytes long
__device__ __forceinline__ uint16_t u16_from_bytes_le(const uint8_t *b) {
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

// Convert 4 bytes to a u32 in little endian order
// **SAFETY**: b has to be at least 4 bytes long
__device__ __forceinline__ uint32_t u32_from_bytes_le(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

template <typename T>
__device__ __forceinline__ void ptr_to_u16_limbs(T (&out)[2], uint32_t value) {
    out[0] = T(uint16_t(value));
    out[1] = T(uint16_t(value >> 16));
}

template <typename T, size_t NUM_LIMBS>
__device__ __forceinline__ void bytes_to_u16_limbs(T (&out)[NUM_LIMBS], const uint8_t *bytes) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        out[i] = T(u16_from_bytes_le(bytes + 2 * i));
    }
}

__device__ __host__ __forceinline__ uint32_t ptr_bound_from_high_u16(
    uint16_t high_u16,
    uint32_t ptr_max_bits
) {
    return uint32_t(high_u16) << (riscv::RV64_PTR_BITS - ptr_max_bits);
}

// Convert 4 bytes to a u32 in big endian order
// **SAFETY**: b has to be at least 4 bytes long
__device__ __forceinline__ uint32_t u32_from_bytes_be(const uint8_t *b) {
    return (uint32_t)b[3] | ((uint32_t)b[2] << 8) | ((uint32_t)b[1] << 16) | ((uint32_t)b[0] << 24);
}

// Convert 8 bytes to a u64 in little endian order
// **SAFETY**: b has to be at least 8 bytes long
__device__ __forceinline__ uint64_t u64_from_bytes_le(const uint8_t *b) {
    return (uint64_t)b[0] | ((uint64_t)b[1] << 8) | ((uint64_t)b[2] << 16) | ((uint64_t)b[3] << 24) |
           ((uint64_t)b[4] << 32) | ((uint64_t)b[5] << 40) | ((uint64_t)b[6] << 48) |
           ((uint64_t)b[7] << 56);
}

template <typename T> __device__ __host__ __forceinline__ T d_div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T> __device__ __host__ __forceinline__ T next_multiple_of(T a, T b) {
    return d_div_ceil(a, b) * b;
}

// Launch params for grid-stride kernels: `threads_per_block`-wide blocks with
// the grid capped at `max_blocks`, so oversize inputs iterate in the kernel's
// stride loop instead of growing the grid with the input. `count` must be
// nonzero. Unlike `kernel_launch_params` (launcher.cuh), which launches one
// thread per element with an unbounded grid.
inline std::pair<dim3, dim3> grid_stride_launch_params(
    size_t count,
    size_t threads_per_block,
    size_t max_blocks
) {
    size_t blocks = std::min(d_div_ceil(count, threads_per_block), max_blocks);
    return std::make_pair(dim3(blocks, 1, 1), dim3(threads_per_block, 1, 1));
}

// UB-free 64-bit rotate left. (-(n)) & 63 equals (64 - n) % 64, so when
// n == 0 the right shift is by 0 instead of by 64 (which would be UB).
__device__ __host__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << (n & 63)) | (x >> ((-n) & 63));
}

__device__ __host__ __forceinline__ uint32_t rotr(uint32_t value, int n) {
    return (value >> n) | (value << (32 - n));
}
