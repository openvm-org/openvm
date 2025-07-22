#include "fp.h"
#include "poseidon2.cuh"
#include "launcher.cuh"

struct alignas(32) digest_t {
    Fp cells[CELLS_OUT];
};

#define COPY_DIGEST(dst, src) memcpy(dst, src, sizeof(digest_t))

template<int ADDR_SPACE_IDX>
__global__ void merkle_tree_init(
    uint8_t * __restrict__ data,
    digest_t * __restrict__ out
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;

    Fp cells[CELLS] = {0};
    // TODO: revisit when we sort out address space handling
    if constexpr (ADDR_SPACE_IDX < 4) {
        cells[0] = Fp(data[gid]);
    } else {
        cells[0] = reinterpret_cast<Fp*>(data)[gid];
    }
    
    poseidon2::poseidon2_mix(cells);

    COPY_DIGEST(&out[gid], cells);
}

/// TODO: multiple compress should be here
__global__ void merkle_tree_compress(
    digest_t * __restrict__ in,
    digest_t * __restrict__ out
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;

    Fp cells[CELLS];
    COPY_DIGEST(cells, &in[2 * gid]);
    COPY_DIGEST(cells + CELLS_OUT, &in[2 * gid + 1]);

    poseidon2::poseidon2_mix(cells);

    COPY_DIGEST(&out[gid], cells);
}

/// This kernel restores the path from subtree node to the root.
__global__ void merkle_tree_restore_path(
    digest_t* __restrict__ in_out,
    digest_t* __restrict__ zero_hash,
    const size_t remaining_size
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }

    Fp cells[CELLS];
    COPY_DIGEST(cells, &in_out[remaining_size]);
    
    for (auto i = 0; i < remaining_size; i++) {
        COPY_DIGEST(cells + CELLS_OUT, &zero_hash[i]);
        poseidon2::poseidon2_mix(cells);
        COPY_DIGEST(&in_out[remaining_size - i - 1], cells);
    }
}

__global__ void calculate_zero_hash(
    digest_t* zero_hash,
    const size_t size
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }

    Fp cells[CELLS] = {0};
    poseidon2::poseidon2_mix(cells);
    COPY_DIGEST(zero_hash, cells);

    for (auto i = 0; i < size; i++) {
        COPY_DIGEST(cells + CELLS_OUT, &zero_hash[i]);
        poseidon2::poseidon2_mix(cells);
        COPY_DIGEST(&zero_hash[i + 1], cells);
    }
}

__global__ void merkle_tree_root(
    uintptr_t*  __restrict__ in_roots, // aka digest_t**
    digest_t*  __restrict__ out,
    const size_t num_roots
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }
    digest_t** in = reinterpret_cast<digest_t**>(in_roots);

    Fp cells[CELLS];
    auto steps = num_roots >> 1;
    for (auto i = 0; i < steps; i++) {
        COPY_DIGEST(cells, in[2 * i]);
        COPY_DIGEST(cells + CELLS_OUT, in[2 * i + 1]);
        poseidon2::poseidon2_mix(cells);
        COPY_DIGEST(&out[steps + i - 1], cells);
    }
    for (auto out_idx = steps - 1; out_idx --> 0;) {
        COPY_DIGEST(cells, &out[ 2 * out_idx + 1]);
        COPY_DIGEST(cells + CELLS_OUT, &out[2 * out_idx + 2]);
        poseidon2::poseidon2_mix(cells);
        COPY_DIGEST(&out[out_idx], cells);
    }
}

#undef COPY_DIGEST

extern "C" int _build_merkle_subtree(
    uint8_t* data,
    const size_t size,
    digest_t* buffer,
    const size_t tree_offset,
    const uint addr_space_idx,
    cudaStream_t stream
) {
    digest_t* tree = buffer + tree_offset;
    assert(size & (size - 1) == 0);
    {
        auto [grid, block] = kernel_launch_params(size);
        switch (addr_space_idx) {   // TODO: revisit when we sort out address space handling
            case 1:
                merkle_tree_init<1><<<grid, block, 0, stream>>>(data, tree + (size - 1));
                break;
            case 2:
                merkle_tree_init<2><<<grid, block, 0, stream>>>(data, tree + (size - 1));
                break;
            case 3:
                merkle_tree_init<3><<<grid, block, 0, stream>>>(data, tree + (size - 1));
                break;
            case 4:
                merkle_tree_init<4><<<grid, block, 0, stream>>>(data, tree + (size - 1));
                break;
            default:
                return -1;
        }
    }
    for (auto i = size / 2; i > 0; i /= 2) {
        auto [grid, block] = kernel_launch_params(i);
        merkle_tree_compress<<<grid, block, 0, stream>>>(tree + (2 * i - 1), tree + (i - 1));
    }
    return cudaGetLastError();
}

extern "C" int _restore_merkle_subtree_path(
    digest_t* in_out,
    digest_t* zero_hash,
    const size_t remaining_size,
    const size_t full_size,
    cudaStream_t stream
) {
    merkle_tree_restore_path<<<1, 1, 0, stream>>>(
        in_out,
        zero_hash + full_size - remaining_size,
        remaining_size
    );
    return cudaGetLastError();
}

extern "C" int _finalize_merkle_tree(
    uintptr_t* in,
    digest_t* out,
    const size_t num_roots,
    cudaStream_t stream
) {
    assert(num_roots & (num_roots - 1) == 0);
    merkle_tree_root<<<1, 1, 0, stream>>>(in, out, num_roots);
    return cudaGetLastError();
}

extern "C" int _calculate_zero_hash(
    digest_t* zero_hash,
    const size_t size
) {
    calculate_zero_hash<<<1, 1>>>(zero_hash, size);
    return cudaGetLastError();
}