#include "fp.h"

__global__ void cukernel_fibair_tracegen(Fp *output, uint32_t a, uint32_t b, uint32_t n) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    output[0] = Fp(a);
    output[n] = Fp(b);

    for (uint32_t i = 1; i < n; i++) {
        Fp prev_a = output[i - 1];
        Fp prev_b = output[n + i - 1];
        output[i] = prev_b;
        output[n + i] = prev_a + prev_b;
    }
}

extern "C" int _fibair_tracegen(Fp *output, uint32_t a, uint32_t b, uint32_t n) {
    dim3 grid(1);
    dim3 block(1);
    cukernel_fibair_tracegen<<<grid, block>>>(output, a, b, n);
    return cudaGetLastError();
}
