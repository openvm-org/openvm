#pragma once

#include "bigint_ops.cuh"
#include "overflow_ops.cuh"
#include <cstdio>

static __global__ void init_prepared_prime_buffers_kernel(
    BigUintGpu *prime,
    OverflowInt *prime_overflow,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    BigUintGpu prime_value(prime_limbs, prime_limb_count, 8);
    *prime = prime_value;
    *prime_overflow = OverflowInt(prime_value, prime_value.num_limbs);
}
