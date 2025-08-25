#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Kernel to fill size-n buffer with a certain value
template <typename T> __global__ void fill_buffer(T *buffer, T value, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        buffer[tid] = value;
    }
}

// Convert 4 bytes to a u32 in little endian order
// **SAFETY**: b has to be at least 4 bytes long
__device__ __forceinline__ uint32_t u32_from_bytes_le(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// Convert 4 bytes to a u32 in big endian order
// **SAFETY**: b has to be at least 4 bytes long
__device__ __forceinline__ uint32_t u32_from_bytes_be(const uint8_t *b) {
    return (uint32_t)b[3] | ((uint32_t)b[2] << 8) | ((uint32_t)b[1] << 16) | ((uint32_t)b[0] << 24);
}

template <typename T> __device__ __host__ __forceinline__ T d_div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T> __device__ __host__ __forceinline__ T next_multiple_of(T a, T b) {
    return d_div_ceil(a, b) * b;
}

#define SHL64(x, s) ((uint64_t)((x) << ((s) & 63)))
#define SHR64(x, s) ((uint64_t)((x) >> (64 - ((s) & 63))))
#define ROTL64(x, s) (SHL64((x), (s)) | SHR64((x), (s)))

__device__ __host__ __forceinline__ uint32_t rotr(uint32_t value, int n) {
    return (value >> n) | (value << (32 - n));
}