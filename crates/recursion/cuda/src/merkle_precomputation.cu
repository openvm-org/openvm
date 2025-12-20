#include "fp.h"
#include "launcher.cuh"
#include "poseidon2.cuh"
#include "types.h"

#include <cstddef>
#include <driver_types.h>

struct VectorDescriptor {
    size_t data_offset;   // offset into d_data (in units of Fp)
    size_t len;           // number of Fp elements in this vector
    size_t output_offset; // offset into d_outputs (in units of WIDTH)
};

__global__ void cukernel_hash_vectors(
    const Fp *d_data,
    const VectorDescriptor *d_descriptors,
    size_t num_vectors,
    Fp *d_pre_states,
    Fp *d_post_states
) {
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) {
        return;
    }

    const VectorDescriptor desc = d_descriptors[vec_idx];
    const Fp *data = d_data + desc.data_offset;
    Fp *pre_out = d_pre_states + desc.output_offset * WIDTH;
    Fp *post_out = d_post_states + desc.output_offset * WIDTH;

    Fp state[WIDTH];
#pragma unroll
    for (size_t i = 0; i < WIDTH; ++i) {
        state[i] = Fp::zero();
    }

    size_t chunk_idx = 0;
    size_t pos = 0;

    for (size_t j = 0; j < desc.len; ++j) {
        state[pos] = data[j];
        pos++;
        if (pos == CHUNK) {
#pragma unroll
            for (size_t i = 0; i < WIDTH; ++i) {
                pre_out[chunk_idx * WIDTH + i] = state[i];
            }
            poseidon2::poseidon2_mix(state);
#pragma unroll
            for (size_t i = 0; i < WIDTH; ++i) {
                post_out[chunk_idx * WIDTH + i] = state[i];
            }
            chunk_idx++;
            pos = 0;
        }
    }

    // Handle partial final chunk
    if (pos != 0) {
#pragma unroll
        for (size_t i = 0; i < WIDTH; ++i) {
            pre_out[chunk_idx * WIDTH + i] = state[i];
        }
        poseidon2::poseidon2_mix(state);
#pragma unroll
        for (size_t i = 0; i < WIDTH; ++i) {
            post_out[chunk_idx * WIDTH + i] = state[i];
        }
    }
}

extern "C" int _merkle_precomputation_hash_vectors(
    const Fp *d_data,
    const VectorDescriptor *d_descriptors,
    size_t num_vectors,
    Fp *d_pre_states,
    Fp *d_post_states
) {
    if (num_vectors == 0) {
        return cudaSuccess;
    }
    auto [grid, block] = kernel_launch_params(num_vectors);
    cukernel_hash_vectors<<<grid, block, 0, cudaStreamPerThread>>>(
        d_data, d_descriptors, num_vectors, d_pre_states, d_post_states
    );
    return CHECK_KERNEL();
}
