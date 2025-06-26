#pragma once

// Kernel to fill size-n buffer with a certain value
template <typename T> __global__ void fill_buffer(T *buffer, T value, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        buffer[tid] = value;
    }
}
