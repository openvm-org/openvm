#include "launcher.cuh"
#include "poseidon2.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"

using poseidon2::poseidon2_mix;

struct alignas(32) digest_t {
    Fp cells[CELLS_OUT];
};

#define COPY_DIGEST(dst, src) memcpy(dst, src, sizeof(digest_t))

// `ADDR_SPACE_IDX` is the address space minus `ADDR_SPACE_OFFSET` (which is 1)
template <int ADDR_SPACE_IDX>
__global__ void merkle_tree_init(uint8_t *__restrict__ data, digest_t *__restrict__ out) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;

    Fp cells[CELLS] = {0};
    // TODO: revisit when we sort out address space handling
#pragma unroll
    for (size_t i = 0; i < CELLS_OUT; ++i) {
        if constexpr (ADDR_SPACE_IDX < 3) {
            cells[i] = Fp(data[CELLS_OUT * gid + i]);
        } else {
            cells[i] = reinterpret_cast<Fp *>(data)[CELLS_OUT * gid + i];
        }
    }

    poseidon2_mix(cells);

    COPY_DIGEST(&out[gid], cells);
}

__global__ void merkle_tree_compress(
    digest_t *__restrict__ in,
    digest_t *__restrict__ out,
    size_t num_compressions
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_compressions) {
        return;
    }

    Fp cells[CELLS];
    COPY_DIGEST(cells, &in[2 * gid]);
    COPY_DIGEST(cells + CELLS_OUT, &in[2 * gid + 1]);

    poseidon2_mix(cells);

    COPY_DIGEST(&out[gid], cells);
}

__global__ void merkle_tree_restore_path(
    digest_t *__restrict__ in_out,
    digest_t *__restrict__ zero_hash,
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
        poseidon2_mix(cells);
        COPY_DIGEST(&in_out[remaining_size - i - 1], cells);
    }
}

__global__ void calculate_zero_hash(digest_t *zero_hash, const size_t size) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }

    Fp cells[CELLS] = {0};
    poseidon2_mix(cells);
    COPY_DIGEST(zero_hash, cells);

    for (auto i = 0; i < size; i++) {
        COPY_DIGEST(cells + CELLS_OUT, &zero_hash[i]);
        poseidon2_mix(cells);
        COPY_DIGEST(&zero_hash[i + 1], cells);
    }
}

__global__ void merkle_tree_root(
    uintptr_t *__restrict__ in_roots, // aka digest_t**
    digest_t *__restrict__ out,
    const size_t num_roots
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }
    digest_t **in = reinterpret_cast<digest_t **>(in_roots);
    for (auto i = 0; i < num_roots; ++i) {
        COPY_DIGEST(&out[num_roots - 1 + i], in[i]);
    }

    Fp cells[CELLS];
    for (auto out_idx = num_roots - 1; out_idx-- > 0;) {
        COPY_DIGEST(cells, &out[2 * out_idx + 1]);
        COPY_DIGEST(cells + CELLS_OUT, &out[2 * out_idx + 2]);
        poseidon2_mix(cells);
        COPY_DIGEST(&out[out_idx], cells);
    }
}

// ================== Merkle tree update routine ==================

__global__ void initial_subtrees_advance(
    uintptr_t *d_subtrees,
    size_t const *actual_subtree_heights,
    size_t const num_subtrees,
    size_t const subtree_height
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_subtrees) {
        return;
    }
    digest_t **subtrees = reinterpret_cast<digest_t **>(d_subtrees);
    auto const h = actual_subtree_heights[gid];
    subtrees[gid] += (1 << (h + 1)) - 1 + (subtree_height - h);
}

__global__ void adjust_subtrees_before_layer_update(
    uintptr_t *d_subtrees,
    size_t const *actual_subtree_heights,
    size_t const num_subtrees,
    size_t const h
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_subtrees) {
        return;
    }
    digest_t **subtrees = reinterpret_cast<digest_t **>(d_subtrees);
    subtrees[gid] -=
        h <= actual_subtree_heights[gid] ? 1 << (actual_subtree_heights[gid] - h + 1) : 1;
}

template <typename T> struct MerkleCols {
    T expand_direction;
    T height_section;
    T parent_height;
    T is_root;
    T parent_as_label;
    T parent_address_label;
    T parent_hash[CELLS_OUT];
    T left_child_hash[CELLS_OUT];
    T right_child_hash[CELLS_OUT];
    T left_direction_different;
    T right_direction_different;
};

struct LabeledDigest {
    uint32_t address_space_idx;
    uint32_t label;
    uint32_t timestamp; // unused
    uint32_t digest_raw[CELLS_OUT];
};

__global__ void prepare_for_updating(
    uint32_t *child_buf,
    LabeledDigest *leaves,
    uint32_t const num_leaves
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_leaves) {
        return;
    }
    child_buf[gid] = gid;
    Fp cells[CELLS] = {0};
    COPY_DIGEST(cells, leaves[gid].digest_raw);
    poseidon2_mix(cells);
    COPY_DIGEST(leaves[gid].digest_raw, cells);
    leaves[gid].address_space_idx -= 1;
    leaves[gid].label /= CELLS_OUT;
}

uint32_t const MISSING_CHILD = UINT_MAX;

__global__ void mark_parents(
    uint32_t const *current_layer_ptrs,
    uint32_t *child_ptrs,
    LabeledDigest const *layer,
    uint32_t const num_children,
    uint32_t const h,
    uint32_t *out_num_parents
) {
    uint32_t num_parents = 1;
    for (uint32_t i = 1; i < num_children; ++i) {
        auto const ptr1 = current_layer_ptrs[i - 1];
        auto const ptr2 = current_layer_ptrs[i];
        if (layer[ptr1].address_space_idx != layer[ptr2].address_space_idx ||
            (layer[ptr1].label >> h) != (layer[ptr2].label >> h)) {
            ++num_parents;
        }
    }
    for (uint32_t i = 0, j = 0; i < num_parents; ++i) {
        child_ptrs[2 * i] = child_ptrs[2 * i + 1] = MISSING_CHILD;
        auto const address_space_idx = layer[current_layer_ptrs[j]].address_space_idx;
        auto const label = layer[current_layer_ptrs[j]].label >> h;
        while (j < num_children) {
            auto const ptr = current_layer_ptrs[j];
            if (layer[ptr].address_space_idx != address_space_idx ||
                (layer[ptr].label >> h) != label) {
                break;
            }
            child_ptrs[2 * i + (layer[ptr].label >> (h - 1)) % 2] = ptr;
            ++j;
        }
    }
    *out_num_parents = num_parents;
}

__device__ void fill_merkle_trace_row(
    RowSlice row,
    bool new_values,
    uint32_t as_label,
    uint32_t parent_label,
    uint32_t parent_height,
    Fp *digests,
    bool left_new,
    bool right_new,
    Poseidon2Buffer &poseidon2
) {
    COL_WRITE_VALUE(row, MerkleCols, expand_direction, new_values ? Fp::neg_one() : Fp::one());
    COL_WRITE_VALUE(row, MerkleCols, height_section, false);
    COL_WRITE_VALUE(row, MerkleCols, parent_height, parent_height);
    COL_WRITE_VALUE(row, MerkleCols, is_root, false);
    COL_WRITE_VALUE(row, MerkleCols, parent_as_label, as_label);
    COL_WRITE_VALUE(row, MerkleCols, parent_address_label, parent_label);
    COL_WRITE_ARRAY(row, MerkleCols, left_child_hash, digests);
    COL_WRITE_ARRAY(row, MerkleCols, right_child_hash, digests + CELLS_OUT);
    poseidon2.compress_and_record_inplace(digests);
    COL_WRITE_ARRAY(row, MerkleCols, parent_hash, digests);
    COL_WRITE_VALUE(row, MerkleCols, left_direction_different, left_new != new_values);
    COL_WRITE_VALUE(row, MerkleCols, right_direction_different, right_new != new_values);
}

__device__ digest_t const *layer_value_on_height(
    digest_t const *subtree_layer,
    digest_t const *zero_hash,
    uint32_t const height,
    uint32_t const layer_actual_height,
    size_t const label
) {
    auto const layer_size =
        1 << (height <= layer_actual_height ? (layer_actual_height - height) : 0);
    return label < layer_size ? subtree_layer + label : zero_hash + height;
}

__global__ void update_merkle_layer(
    uint32_t layer_height,
    digest_t const *zero_hash,
    size_t const *actual_subtree_heights,
    LabeledDigest *layer,
    uint32_t const *child_ptrs,
    uint32_t *parent_ptrs,
    size_t const num_parents,
    uintptr_t *d_subtree_layers,
    Fp *const merkle_trace,
    size_t const trace_height,
    Fp *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_parents) {
        return;
    }
    Fp cells[CELLS];
    digest_t **subtree_layers = reinterpret_cast<digest_t **>(d_subtree_layers);

    uint32_t const parent_ptr = parent_ptrs[idx] =
        ((child_ptrs[2 * idx] == MISSING_CHILD) ? child_ptrs[2 * idx + 1] : child_ptrs[2 * idx]);
    uint32_t const address_space_idx = layer[parent_ptr].address_space_idx;
    uint32_t const parent_label = layer[parent_ptr].label >> layer_height;
    auto const subtree_layer = subtree_layers[address_space_idx];
    Poseidon2Buffer poseidon2(
        reinterpret_cast<FpArray<16> *>(poseidon2_buffer), poseidon2_buffer_idx, poseidon2_capacity
    );
    auto const old_left_digest = layer_value_on_height(
        subtree_layer,
        zero_hash,
        layer_height - 1,
        actual_subtree_heights[address_space_idx],
        2 * parent_label
    );
    auto const old_right_digest = layer_value_on_height(
        subtree_layer,
        zero_hash,
        layer_height - 1,
        actual_subtree_heights[address_space_idx],
        2 * parent_label + 1
    );
    { // old values trace row
        COPY_DIGEST(cells, old_left_digest);
        COPY_DIGEST(cells + CELLS_OUT, old_right_digest);
        RowSlice row(merkle_trace + 2 * idx, trace_height);
        fill_merkle_trace_row(
            row,
            false,
            address_space_idx,
            parent_label,
            layer_height,
            cells,
            false,
            false,
            poseidon2
        );
    }

    { // new values trace row + actual update
        bool left_new = false;
        if (auto const child_ptr = child_ptrs[2 * idx]; child_ptr != MISSING_CHILD) {
            COPY_DIGEST(&subtree_layer[2 * parent_label], layer[child_ptr].digest_raw);
            left_new = true;
        }
        COPY_DIGEST(cells, old_left_digest);
        bool right_new = false;
        if (auto const child_ptr = child_ptrs[2 * idx + 1]; child_ptr != MISSING_CHILD) {
            COPY_DIGEST(&subtree_layer[2 * parent_label + 1], layer[child_ptr].digest_raw);
            right_new = true;
        }
        COPY_DIGEST(cells + CELLS_OUT, old_right_digest);
        RowSlice row(merkle_trace + 2 * idx + 1, trace_height);
        fill_merkle_trace_row(
            row,
            true,
            address_space_idx,
            parent_label,
            layer_height,
            cells,
            left_new,
            right_new,
            poseidon2
        );
        COPY_DIGEST(layer[parent_ptr].digest_raw, cells);
    }
}

__device__ uint32_t drop_highest_bit(uint32_t x) { return x & ~(1 << (31 - __clz(x))); }

__global__ void update_to_root(
    uint32_t *layer_ids,
    LabeledDigest *layer,
    size_t layer_size,
    size_t const num_roots,
    uintptr_t *d_subtrees,
    digest_t *out,
    Fp *const merkle_trace,
    uint32_t merkle_trace_offset,
    size_t const trace_height,
    size_t const root_height,
    Fp *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }
    digest_t **subtrees = reinterpret_cast<digest_t **>(d_subtrees);
    for (size_t i = 0; i < layer_size; ++i) {
        auto const idx = layer_ids[i];
        auto const address_space_idx = layer[idx].address_space_idx;
        layer[idx].label = num_roots - 1 + address_space_idx;
        if (subtrees[address_space_idx]) {
            COPY_DIGEST(subtrees[address_space_idx], layer[idx].digest_raw);
        }
    }

    Fp cells[CELLS];
    Poseidon2Buffer poseidon2(
        reinterpret_cast<FpArray<16> *>(poseidon2_buffer), poseidon2_buffer_idx, poseidon2_capacity
    );
    for (auto out_idx = num_roots - 1; out_idx-- > 0;) {
        size_t const h = root_height - (31 - __clz((uint32_t)out_idx + 1));
        uint32_t children_ids[2] = {MISSING_CHILD, MISSING_CHILD};
        for (size_t i = 0; i < layer_size; ++i) {
            if (auto local_idx = layer[layer_ids[i]].label - 2 * out_idx;
                local_idx == 1 || local_idx == 2) {
                children_ids[local_idx - 1] = i;
            }
        }
        if (children_ids[0] == MISSING_CHILD && children_ids[1] == MISSING_CHILD) {
            continue;
        }
        merkle_trace_offset -= 2;
        {
            RowSlice row(merkle_trace + merkle_trace_offset, trace_height);
            COPY_DIGEST(cells, &out[2 * out_idx + 1]);
            COPY_DIGEST(cells + CELLS_OUT, &out[2 * out_idx + 2]);
            fill_merkle_trace_row(
                row, false, drop_highest_bit(out_idx + 1), 0, h, cells, false, false, poseidon2
            );
            COL_WRITE_VALUE(row, MerkleCols, height_section, true);
        }
        for (auto i : {0, 1}) {
            if (children_ids[i] != MISSING_CHILD) {
                COPY_DIGEST(
                    &out[2 * out_idx + 1 + i], layer[layer_ids[children_ids[i]]].digest_raw
                );
            }
            COPY_DIGEST(cells + CELLS_OUT * i, &out[2 * out_idx + 1 + i]);
        }

        size_t const surely_surviving_child = std::min(children_ids[0], children_ids[1]);
        if (children_ids[0] != MISSING_CHILD && children_ids[1] != MISSING_CHILD) {
            size_t const max_idx = children_ids[children_ids[0] == surely_surviving_child];
            layer_ids[max_idx] = layer_ids[--layer_size];
        }
        layer[layer_ids[surely_surviving_child]].label = out_idx;
        {
            RowSlice row(merkle_trace + merkle_trace_offset + 1, trace_height);
            fill_merkle_trace_row(
                row,
                true,
                drop_highest_bit(out_idx + 1),
                0,
                h,
                cells,
                children_ids[0] != MISSING_CHILD,
                children_ids[1] != MISSING_CHILD,
                poseidon2
            );
            COPY_DIGEST(layer[layer_ids[surely_surviving_child]].digest_raw, cells);
            COL_WRITE_VALUE(row, MerkleCols, height_section, true);
        }
    }
    COPY_DIGEST(out, layer[layer_ids[0]].digest_raw);
    for (auto i : {0, 1}) {
        RowSlice row(merkle_trace + i, trace_height);
        COL_WRITE_VALUE(row, MerkleCols, is_root, true);
    }
    assert(layer_size == 1);
}

// ================== Merkle tree update routine end ==================

__global__ void get_subtree_root(
    digest_t *const *subtrees,
    size_t const address_space_idx,
    Fp *out
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid != 0) {
        return;
    }
    COPY_DIGEST(out, subtrees[address_space_idx]);
}

#undef COPY_DIGEST

// `addr_space_idx` is the address space _shifted_ by ADDR_SPACE_OFFSET = 1
extern "C" int _build_merkle_subtree(
    uint8_t *data,
    const size_t size,
    digest_t *buffer,
    const size_t tree_offset,
    const uint addr_space_idx,
    cudaStream_t stream
) {
    digest_t *tree = buffer + tree_offset;
    assert((size & (size - 1)) == 0);
    {
        auto [grid, block] = kernel_launch_params(size);
        switch (addr_space_idx) { // TODO: revisit when we sort out address space handling
        case 0:
            merkle_tree_init<0><<<grid, block, 0, stream>>>(data, tree + (size - 1));
            break;
        case 1:
            merkle_tree_init<1><<<grid, block, 0, stream>>>(data, tree + (size - 1));
            break;
        case 2:
            merkle_tree_init<2><<<grid, block, 0, stream>>>(data, tree + (size - 1));
            break;
        case 3:
            merkle_tree_init<3><<<grid, block, 0, stream>>>(data, tree + (size - 1));
            break;
        default:
            return -1;
        }
    }
    for (auto i = size / 2; i > 0; i /= 2) {
        auto [grid, block] = kernel_launch_params(i);
        merkle_tree_compress<<<grid, block, 0, stream>>>(tree + (2 * i - 1), tree + (i - 1), i);
    }
    return cudaGetLastError();
}

extern "C" int _restore_merkle_subtree_path(
    digest_t *in_out,
    digest_t *zero_hash,
    const size_t remaining_size,
    const size_t full_size,
    cudaStream_t stream
) {
    merkle_tree_restore_path<<<1, 1, 0, stream>>>(
        in_out, zero_hash + full_size - remaining_size, remaining_size
    );
    return cudaGetLastError();
}

extern "C" int _finalize_merkle_tree(
    uintptr_t *in,
    digest_t *out,
    const size_t num_roots,
    cudaStream_t stream
) {
    assert((num_roots & (num_roots - 1)) == 0);
    merkle_tree_root<<<1, 1, 0, stream>>>(in, out, num_roots);
    return cudaGetLastError();
}

extern "C" int _calculate_zero_hash(digest_t *zero_hash, const size_t size) {
    calculate_zero_hash<<<1, 1>>>(zero_hash, size);
    return cudaGetLastError();
}

/// Updates the digests in `subtrees`, replacing them with the new ones,
/// while also producing the trace.
/// Here, `layer` is obtained from the touched memory.
/// We go layer by layer, from the leaves to the root,
/// without reordering the layer and only updating its digests,
/// and maintaining the indices of its values that are still relevant.
/// On each layer, we first compute the number of their parents serially
/// (it is possible to do it in parallel), and then update the values in `subtrees`
/// and write to the trace.
///
/// After we reach the address space subtree roots, we call a single `update_to_root` function
/// to do the remaining work there.
extern "C" int _update_merkle_tree(
    size_t const num_leaves,
    LabeledDigest *layer,
    size_t subtree_height,
    uint32_t *child_buf,
    uint32_t *tmp_buf,
    Fp *const merkle_trace,
    size_t const unpadded_trace_height,
    size_t const num_subtrees,
    uintptr_t *subtrees,
    digest_t *top_roots,
    digest_t const *zero_hash,
    size_t const *actual_subtree_heights,
    Fp *d_poseidon2_raw_buffer,
    uint32_t *d_poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    assert(num_leaves > 0);
    uint32_t num_children = num_leaves;
    size_t const trace_height = [](uint32_t x) {
        return x ? (1u << (32 - __builtin_clz(x - 1))) : 0;
    }(unpadded_trace_height);

    {
        auto [grid, block] = kernel_launch_params(num_leaves, 256);
        prepare_for_updating<<<grid, block>>>(child_buf, layer, num_children);
    }
    {
        auto [grid, block] = kernel_launch_params(num_subtrees);
        initial_subtrees_advance<<<grid, block>>>(
            subtrees, actual_subtree_heights, num_subtrees, subtree_height
        );
    }

    uint32_t *d_num_parents;
    uint32_t merkle_trace_offset = unpadded_trace_height;
    cudaMallocAsync(&d_num_parents, sizeof(uint32_t), cudaStreamPerThread);
    for (uint32_t h = 1; h <= subtree_height; ++h) {
        uint32_t num_parents;
        mark_parents<<<1, 1>>>(child_buf, tmp_buf, layer, num_children, h, d_num_parents);
        cudaMemcpyAsync(
            &num_parents,
            d_num_parents,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread
        );
        cudaStreamSynchronize(cudaStreamPerThread);
        {
            auto [grid, block] = kernel_launch_params(num_subtrees);
            adjust_subtrees_before_layer_update<<<grid, block>>>(
                subtrees, actual_subtree_heights, num_subtrees, h
            );
        }
        merkle_trace_offset -= 2 * num_parents;
        auto [grid, block] = kernel_launch_params(num_parents, 256);
        update_merkle_layer<<<grid, block>>>(
            h,
            zero_hash,
            actual_subtree_heights,
            layer,
            tmp_buf,
            child_buf,
            num_parents,
            subtrees,
            merkle_trace + merkle_trace_offset,
            trace_height,
            d_poseidon2_raw_buffer,
            d_poseidon2_buffer_idx,
            poseidon2_capacity
        );
        num_children = num_parents;
    }
    cudaFreeAsync(d_num_parents, cudaStreamPerThread);
    update_to_root<<<1, 1>>>(
        child_buf,
        layer,
        num_children,
        num_subtrees,
        subtrees,
        top_roots,
        merkle_trace,
        merkle_trace_offset,
        trace_height,
        subtree_height + __builtin_ctz(num_subtrees),
        d_poseidon2_raw_buffer,
        d_poseidon2_buffer_idx,
        poseidon2_capacity
    );

    return cudaGetLastError();
}
