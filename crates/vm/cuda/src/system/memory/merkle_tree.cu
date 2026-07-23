#include "launcher.cuh"
#include "poseidon2.cuh"
#include "primitives/shared_buffer.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/params.cuh"

#include <cub/cub.cuh>

using poseidon2::poseidon2_mix;

struct alignas(32) digest_t {
    Fp cells[DIGEST_WIDTH];
};

#define COPY_DIGEST(dst, src) memcpy(dst, src, sizeof(digest_t))

enum MemoryMerkleSubTreeLayout : uint8_t {
    FULL = 0,
    OMIT_BOTTOM_LEVELS = 1,
};

__device__ __forceinline__ void hash_raw_memory_leaf(
    uint8_t const *__restrict__ data,
    uint32_t const addr_space_idx,
    size_t const leaf_label,
    digest_t *out
) {
    Fp cells[CELLS] = {0};
#pragma unroll
    for (size_t i = 0; i < DIGEST_WIDTH; ++i) {
        if (addr_space_idx + 1 == DEFERRAL_AS) {
            cells[i] = reinterpret_cast<Fp const *>(data)[DIGEST_WIDTH * leaf_label + i];
        } else {
            auto byte_off = U16_CELL_SIZE * (DIGEST_WIDTH * leaf_label + i);
            cells[i] = Fp(u16_from_bytes_le(data + byte_off));
        }
    }

    poseidon2_mix(cells);
    COPY_DIGEST(out, cells);
}

// `ADDR_SPACE_IDX` is the address space minus `ADDR_SPACE_OFFSET` (which is 1)
//
// DEFERRAL_AS stores Fp cells directly.
// Non-deferral address spaces store u16 cells in little-endian byte order.
template <int ADDR_SPACE_IDX>
__global__ void merkle_tree_init(uint8_t *__restrict__ data, digest_t *__restrict__ out) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;

    digest_t digest;
    hash_raw_memory_leaf(data, ADDR_SPACE_IDX, gid, &digest);

    COPY_DIGEST(&out[gid], &digest);
}

__device__ void recompute_omitted_node(
    uint8_t const *__restrict__ data,
    uint32_t const addr_space_idx,
    uint32_t const node_height,
    // label is the index of the node within its level `node_height` (label 0 = leftmost). 
    // the node roots a subtree of `2^node_height` leaves, 
    // namely leaf labels `[label << node_height .. (label + 1) << node_height)`.
    // i.e. those leaves get hashed to this node.
    size_t const label,
    digest_t *out
) {
    // layer is a fixed-size, thread-local scratch buffer of 2^OMITTED_BOTTOM_LEVELS 
    // digests, used to rebuild an omitted subtree bottom-up.
    digest_t layer[1 << OMITTED_BOTTOM_LEVELS];
    // num_leaves denote the number of leaves of the subtree rooted at this node 
    size_t const num_leaves = 1 << node_height;
    // recall again that node_height is counted starting from the bottom
    // that is, node_height = 0 for the leafs
    size_t const first_leaf = label << node_height;

    for (size_t i = 0; i < num_leaves; ++i) {
        hash_raw_memory_leaf(data, addr_space_idx, first_leaf + i, &layer[i]);
    }

    // cells is a 2-to-1 compression buffer
    Fp cells[CELLS];
    for (size_t width = num_leaves / 2; width > 0; width /= 2) {
        for (size_t i = 0; i < width; ++i) {
            COPY_DIGEST(cells, &layer[2 * i]);
            COPY_DIGEST(cells + CELLS_OUT, &layer[2 * i + 1]);
            poseidon2_mix(cells);
            COPY_DIGEST(&layer[i], cells);
        }
    }
    COPY_DIGEST(out, &layer[0]);
}

__global__ void merkle_tree_init_omitted(
    uint8_t *__restrict__ data,
    digest_t *__restrict__ out,
    uint32_t const addr_space_idx
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    recompute_omitted_node(data, addr_space_idx, OMITTED_BOTTOM_LEVELS, gid, &out[gid]);
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

template <typename T> struct MerkleCols {
    T expand_direction;
    T height_section;
    T parent_height;
    T parent_height_inv;
    T is_root;
    T parent_as_label;
    T parent_address_label;
    T parent_hash[CELLS_OUT];
    T left_child_hash[CELLS_OUT];
    T right_child_hash[CELLS_OUT];
    T left_direction_different;
    T right_direction_different;
    T left_adj_ref;
    T right_adj_ref;
};

struct LabeledDigest {
    uint32_t address_space_idx;
    uint32_t label;
    /// "This node's value changed": arrives precomputed for leaves in the record's third
    /// word (the inventory record-conversion kernel compares final against initial
    /// values; see `MemoryMerkleRecord`) and is OR-propagated upward by
    /// `update_merkle_layer`. Dirtiness decides whether a node emits a final-direction
    /// trace row.
    uint32_t is_dirty;
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

__global__ void set_parent_id_adjacent_differences(
    uint32_t const *current_layer_ptrs,
    uint32_t *parent_ids,
    LabeledDigest const *layer,
    uint32_t const num_children,
    uint32_t const h
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_children) {
        return;
    }
    if (gid == 0) {
        parent_ids[gid] = 0;
    } else {
        auto const ptr1 = current_layer_ptrs[gid - 1];
        auto const ptr2 = current_layer_ptrs[gid];
        parent_ids[gid] = layer[ptr1].address_space_idx != layer[ptr2].address_space_idx
            || (layer[ptr1].label >> h) != (layer[ptr2].label >> h);
    }
}

uint32_t const MISSING_CHILD = UINT_MAX;

__global__ void group_by_parent(
    uint32_t const *current_layer_ptrs,
    uint32_t const *parent_ids,
    uint32_t *child_ptrs,
    LabeledDigest const *layer,
    uint32_t const num_children,
    uint32_t const h
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_children) {
        return;
    }

    uint32_t const ptr = current_layer_ptrs[gid];
    uint32_t const my_place = parent_ids[gid] * 2 + (layer[ptr].label >> (h - 1)) % 2;
    uint32_t const siblings_place = my_place ^ 1;

    child_ptrs[my_place] = ptr;
    // Mark the sibling as absent if we are the only child
    {
        uint32_t const sibling_id = gid + (siblings_place - my_place);
        if (sibling_id >= num_children || parent_ids[sibling_id] != parent_ids[gid]) {
            child_ptrs[siblings_place] = MISSING_CHILD;
        }
    }
}

/// Reference-count adjustment for a child of an initial row
__device__ inline Fp child_adj_ref(bool emits_final, bool present, bool dirty) {
    if (emits_final && present && !dirty) {
        return Fp::one();
    }
    if (!emits_final && !present) {
        return Fp::neg_one();
    }
    return Fp::zero();
}

/// Fills one merkle trace row and records its compression (leaving the parent digest in
/// `digests[0..CELLS_OUT]`).
///
/// For final rows (`new_values`), `direction_different` marks children *not expanded
/// finally* (untouched or touched-but-clean), whose hashes are borrowed from the initial
/// tree.
__device__ void fill_merkle_trace_row(
    RowSlice row,
    bool new_values,
    uint32_t as_label,
    uint32_t parent_label,
    uint32_t parent_height,
    Fp *digests,
    bool left_present,
    bool right_present,
    bool left_dirty,
    bool right_dirty,
    bool emits_final,
    Poseidon2Buffer &poseidon2
) {
    COL_WRITE_VALUE(row, MerkleCols, expand_direction, new_values ? Fp::neg_one() : Fp::one());
    COL_WRITE_VALUE(row, MerkleCols, height_section, false);
    COL_WRITE_VALUE(row, MerkleCols, parent_height, parent_height);
    COL_WRITE_VALUE(row, MerkleCols, parent_height_inv, inv(Fp(parent_height)));
    COL_WRITE_VALUE(row, MerkleCols, is_root, false);
    COL_WRITE_VALUE(row, MerkleCols, parent_as_label, as_label);
    COL_WRITE_VALUE(row, MerkleCols, parent_address_label, parent_label);
    COL_WRITE_ARRAY(row, MerkleCols, left_child_hash, digests);
    COL_WRITE_ARRAY(row, MerkleCols, right_child_hash, digests + CELLS_OUT);
    poseidon2.compress_and_record_inplace(digests);
    COL_WRITE_ARRAY(row, MerkleCols, parent_hash, digests);
    COL_WRITE_VALUE(row, MerkleCols, left_direction_different, new_values && !left_dirty);
    COL_WRITE_VALUE(row, MerkleCols, right_direction_different, new_values && !right_dirty);
    COL_WRITE_VALUE(
        row,
        MerkleCols,
        left_adj_ref,
        new_values ? Fp::zero() : child_adj_ref(emits_final, left_present, left_dirty)
    );
    COL_WRITE_VALUE(
        row,
        MerkleCols,
        right_adj_ref,
        new_values ? Fp::zero() : child_adj_ref(emits_final, right_present, right_dirty)
    );
}

// A "virtual node" is a node of the conceptual full subtree, addressed by
// (node_height, label). It is "virtual" because depending on where it falls it
// can map to one of four different physical representations:
//   1. Non-existent: `label` lies beyond the touched layer (higher memory
//      addresses that were never written). It is not stored at all; its value
//      is the precomputed constant `zero_hash[node_height]`.
//   2. Omitted bottom level: under the OMIT_BOTTOM_LEVELS layout, nodes with
//      `node_height < OMITTED_BOTTOM_LEVELS` are not stored either; they are
//      recomputed on demand from `initial_data`.
//   3. On the vertical path (`node_height > actual_height`): a stored node
//      above the touched subtree, indexed linearly by height.
//   4. Inside the actual touched subtree (`node_height <= actual_height`): a
//      stored node at the heap-style index for its (height, label).
// virtual_node_exists / stored_node_index / load_virtual_node implement this
// mapping; see stored_node_index for the exact stored indices of cases 3 and 4.
__device__ __forceinline__ bool virtual_node_exists(
    uint32_t const node_height,
    // actual height denotes the height of the subtree excluding the vertical path length
    // that is, the height of the subtree excluding the higher memory addresses
    // which are never touched
    uint32_t const actual_height, 
    size_t const label
) {
    auto const layer_size = 1 << (node_height <= actual_height ? (actual_height - node_height) : 0);
    return label < layer_size;
}

__device__ __forceinline__ size_t stored_node_index(
    uint32_t const subtree_height,
    uint32_t const actual_height,
    uint32_t const node_height,
    size_t const label
) {
    if (node_height > actual_height) {
        return subtree_height - node_height;
    }
    auto const path_len = subtree_height - actual_height;
    return path_len + ((1 << (actual_height - node_height)) - 1) + label;
}

__device__ void load_virtual_node(
    digest_t const *subtree,
    digest_t const *zero_hash,
    uint8_t const layout,
    uint8_t const *initial_data,
    uint32_t const address_space_idx,
    uint32_t const subtree_height,
    uint32_t const actual_height,
    uint32_t const node_height,
    size_t const label,
    digest_t *out
) {
    if (!virtual_node_exists(node_height, actual_height, label)) {
        COPY_DIGEST(out, &zero_hash[node_height]);
        return;
    }

    if (layout == OMIT_BOTTOM_LEVELS && node_height < OMITTED_BOTTOM_LEVELS) {
        recompute_omitted_node(initial_data, address_space_idx, node_height, label, out);
        return;
    }

    auto const idx = stored_node_index(subtree_height, actual_height, node_height, label);
    COPY_DIGEST(out, &subtree[idx]);
}

__device__ void store_virtual_node(
    digest_t *subtree,
    uint8_t const layout,
    uint32_t const subtree_height,
    uint32_t const actual_height,
    uint32_t const node_height,
    size_t const label,
    digest_t const *value
) {
    if (layout == OMIT_BOTTOM_LEVELS && node_height < OMITTED_BOTTOM_LEVELS) {
        return;
    }
    auto const idx = stored_node_index(subtree_height, actual_height, node_height, label);
    COPY_DIGEST(&subtree[idx], value);
}

/// For each parent group of the current layer, computes whether the parent is dirty
/// (some present child is dirty; leaf dirtiness arrives precomputed in the records).
/// Launched over `num_children` threads so the count can stay on device: entries beyond
/// the parent count (read from the tail of the inclusive scan) are zeroed, letting the
/// subsequent prefix sum run over the host-known `num_children` length. The prefix
/// places each dirty parent's final row and its total gives the layer's final-row count.
__global__ void mark_parent_dirty(
    uint32_t const *child_ptrs,
    LabeledDigest const *layer,
    uint32_t *parent_dirty,
    size_t const num_children,
    uint32_t const *num_parents_minus_one
) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= num_children) {
        return;
    }
    uint32_t dirty = 0;
    if (gid <= *num_parents_minus_one) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t const child_ptr = child_ptrs[2 * gid + i];
            if (child_ptr != MISSING_CHILD) {
                dirty |= layer[child_ptr].is_dirty;
            }
        }
    }
    parent_dirty[gid] = dirty;
}

__global__ void update_merkle_layer(
    uint32_t layer_height,
    digest_t const *zero_hash,
    size_t const *actual_subtree_heights,
    uint8_t const *subtree_layouts,
    uintptr_t const *initial_data_ptrs,
    uint32_t const subtree_height,
    LabeledDigest *layer,
    uint32_t const *child_ptrs,
    uint32_t *parent_ptrs,
    size_t const num_parents,
    uintptr_t *d_subtrees,
    Fp *const merkle_trace,
    size_t const trace_height,
    // Inclusive prefix sum of per-parent dirtiness for this layer; a dirty parent's
    // final row goes at `num_parents + prefix[idx] - 1` within the layer's region.
    uint32_t const *parent_dirty_prefix,
    // Set only for the topmost layer when it holds the true root (single-subtree
    // configs): the root's final row must exist even if it is clean, since the AIR pins
    // the first two rows to the root public values.
    bool const force_final,
    Fp *poseidon2_buffer,
    uint32_t *poseidon2_buffer_idx,
    size_t poseidon2_capacity
) {
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_parents) {
        return;
    }
    Fp cells[CELLS];
    digest_t **subtrees = reinterpret_cast<digest_t **>(d_subtrees);

    uint32_t const parent_ptr = parent_ptrs[idx] =
        ((child_ptrs[2 * idx] == MISSING_CHILD) ? child_ptrs[2 * idx + 1] : child_ptrs[2 * idx]);
    uint32_t const address_space_idx = layer[parent_ptr].address_space_idx;
    uint32_t const parent_label = layer[parent_ptr].label >> layer_height;
    auto const subtree = subtrees[address_space_idx];
    uint32_t const actual_height = actual_subtree_heights[address_space_idx];
    uint8_t const layout = subtree_layouts[address_space_idx];
    auto const initial_data = reinterpret_cast<uint8_t const *>(initial_data_ptrs[address_space_idx]);
    Poseidon2Buffer poseidon2(
        reinterpret_cast<FpArray<16> *>(poseidon2_buffer), poseidon2_buffer_idx, poseidon2_capacity
    );
    digest_t old_left_digest;
    load_virtual_node(
        subtree,
        zero_hash,
        layout,
        initial_data,
        address_space_idx,
        subtree_height,
        actual_height,
        layer_height - 1,
        2 * parent_label,
        &old_left_digest
    );
    digest_t old_right_digest;
    load_virtual_node(
        subtree,
        zero_hash,
        layout,
        initial_data,
        address_space_idx,
        subtree_height,
        actual_height,
        layer_height - 1,
        2 * parent_label + 1,
        &old_right_digest
    );
    bool const left_present = child_ptrs[2 * idx] != MISSING_CHILD;
    bool const right_present = child_ptrs[2 * idx + 1] != MISSING_CHILD;
    bool const left_dirty = left_present && layer[child_ptrs[2 * idx]].is_dirty;
    bool const right_dirty = right_present && layer[child_ptrs[2 * idx + 1]].is_dirty;
    bool const node_dirty = left_dirty || right_dirty;
    bool const emits_final = node_dirty || force_final;

    digest_t old_parent;
    { // initial (old values) trace row -- one per touched node
        COPY_DIGEST(cells, &old_left_digest);
        COPY_DIGEST(cells + CELLS_OUT, &old_right_digest);
        RowSlice row(merkle_trace + idx, trace_height);
        fill_merkle_trace_row(
            row,
            false,
            address_space_idx,
            parent_label,
            layer_height,
            cells,
            left_present,
            right_present,
            left_dirty,
            right_dirty,
            emits_final,
            poseidon2
        );
        // fill_merkle_trace_row leaves the compressed parent digest in `cells`.
        COPY_DIGEST(&old_parent, cells);
    }

    { // subtree update + optional final (new values) trace row
        if (left_present) {
            COPY_DIGEST(cells, layer[child_ptrs[2 * idx]].digest_raw);
            store_virtual_node(
                subtree,
                layout,
                subtree_height,
                actual_height,
                layer_height - 1,
                2 * parent_label,
                reinterpret_cast<digest_t const *>(layer[child_ptrs[2 * idx]].digest_raw)
            );
        } else {
            COPY_DIGEST(cells, &old_left_digest);
        }
        if (right_present) {
            COPY_DIGEST(cells + CELLS_OUT, layer[child_ptrs[2 * idx + 1]].digest_raw);
            store_virtual_node(
                subtree,
                layout,
                subtree_height,
                actual_height,
                layer_height - 1,
                2 * parent_label + 1,
                reinterpret_cast<digest_t const *>(layer[child_ptrs[2 * idx + 1]].digest_raw)
            );
        } else {
            COPY_DIGEST(cells + CELLS_OUT, &old_right_digest);
        }
        if (emits_final) {
            // Dirty finals are packed after the layer's initial rows in prefix order; a
            // clean forced root (rank 0 in a single-node layer) lands there trivially.
            uint32_t const rank = parent_dirty_prefix[idx] - (node_dirty ? 1 : 0);
            RowSlice row(merkle_trace + num_parents + rank, trace_height);
            fill_merkle_trace_row(
                row,
                true,
                address_space_idx,
                parent_label,
                layer_height,
                cells,
                left_present,
                right_present,
                left_dirty,
                right_dirty,
                emits_final,
                poseidon2
            );
            COPY_DIGEST(layer[parent_ptr].digest_raw, cells);
        } else {
            // Clean node: the new digest equals the stored one; no final row, no final
            // compression.
            COPY_DIGEST(layer[parent_ptr].digest_raw, &old_parent);
        }
        layer[parent_ptr].is_dirty = node_dirty;
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
        bool const left_present = children_ids[0] != MISSING_CHILD;
        bool const right_present = children_ids[1] != MISSING_CHILD;
        bool const left_dirty =
            left_present && layer[layer_ids[children_ids[0]]].is_dirty;
        bool const right_dirty =
            right_present && layer[layer_ids[children_ids[1]]].is_dirty;
        bool const node_dirty = left_dirty || right_dirty;
        // The true root's final row always exists: the AIR pins the first two rows to
        // the initial/final root public values.
        bool const emits_final = node_dirty || out_idx == 0;

        merkle_trace_offset -= emits_final ? 2 : 1;
        digest_t old_parent;
        {
            RowSlice row(merkle_trace + merkle_trace_offset, trace_height);
            COPY_DIGEST(cells, &out[2 * out_idx + 1]);
            COPY_DIGEST(cells + CELLS_OUT, &out[2 * out_idx + 2]);
            fill_merkle_trace_row(
                row,
                false,
                drop_highest_bit(out_idx + 1),
                0,
                h,
                cells,
                left_present,
                right_present,
                left_dirty,
                right_dirty,
                emits_final,
                poseidon2
            );
            COL_WRITE_VALUE(row, MerkleCols, height_section, true);
            COPY_DIGEST(&old_parent, cells);
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
        if (emits_final) {
            RowSlice row(merkle_trace + merkle_trace_offset + 1, trace_height);
            fill_merkle_trace_row(
                row,
                true,
                drop_highest_bit(out_idx + 1),
                0,
                h,
                cells,
                left_present,
                right_present,
                left_dirty,
                right_dirty,
                emits_final,
                poseidon2
            );
            COPY_DIGEST(layer[layer_ids[surely_surviving_child]].digest_raw, cells);
            COL_WRITE_VALUE(row, MerkleCols, height_section, true);
        } else {
            // Clean node: new digest equals the stored one; no final row or compression.
            COPY_DIGEST(layer[layer_ids[surely_surviving_child]].digest_raw, &old_parent);
        }
        layer[layer_ids[surely_surviving_child]].is_dirty = node_dirty;
    }
    COPY_DIGEST(out, layer[layer_ids[0]].digest_raw);
    // The host computed the trace height exactly, so the downward walk must land the
    // root pair precisely at rows 0..1.
    assert(merkle_trace_offset == 0);
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
    const uint8_t layout,
    cudaStream_t stream
) {
    digest_t *tree = buffer + tree_offset;
    assert((size & (size - 1)) == 0);
    {
        auto [grid, block] = kernel_launch_params(size);
        if (layout == OMIT_BOTTOM_LEVELS) {
            merkle_tree_init_omitted<<<grid, block, 0, stream>>>(
                data, tree + (size - 1), addr_space_idx
            );
        } else {
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
    }
    for (auto i = size / 2; i > 0; i /= 2) {
        auto [grid, block] = kernel_launch_params(i);
        merkle_tree_compress<<<grid, block, 0, stream>>>(tree + (2 * i - 1), tree + (i - 1), i);
    }
    return CHECK_KERNEL();
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
    return CHECK_KERNEL();
}

extern "C" int _finalize_merkle_tree(
    uintptr_t *in,
    digest_t *out,
    const size_t num_roots,
    cudaStream_t stream
) {
    assert((num_roots & (num_roots - 1)) == 0);
    merkle_tree_root<<<1, 1, 0, stream>>>(in, out, num_roots);
    return CHECK_KERNEL();
}

extern "C" int _calculate_zero_hash(digest_t *zero_hash, const size_t size, cudaStream_t stream) {
    calculate_zero_hash<<<1, 1, 0, stream>>>(zero_hash, size);
    return CHECK_KERNEL();
}

extern "C" int _get_prefix_scan_temp_bytes(uint32_t *d_arr, size_t n, size_t *h_temp_n, cudaStream_t stream) {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_arr, d_arr, n, stream);
    *h_temp_n = temp_bytes;
    return CHECK_KERNEL();
}

/// Updates the digests in `subtrees`, replacing them with the new ones,
/// while also producing the trace.
/// Here, `layer` is obtained from the touched memory.
/// We go layer by layer, from the leaves to the root,
/// without reordering the layer and only updating its digests,
/// and maintaining the indices of its values that are still relevant.
///
/// After we reach the address space subtree roots, we call a single `update_to_root` function
/// to do the remaining work there.
extern "C" int _update_merkle_tree(
    size_t const num_leaves,
    LabeledDigest *layer,
    size_t subtree_height,
    uint32_t *child_buf,
    uint32_t *tmp_buf,
    uint32_t *dirty_buf,
    uint8_t *d_temp_storage,
    size_t need_tmp_storage_bytes,
    Fp *const merkle_trace,
    size_t const unpadded_trace_height,
    size_t const num_subtrees,
    uintptr_t *subtrees,
    digest_t *top_roots,
    digest_t const *zero_hash,
    size_t const *actual_subtree_heights,
    uint8_t const *subtree_layouts,
    uintptr_t const *initial_data_ptrs,
    Fp *d_poseidon2_raw_buffer,
    uint32_t *d_poseidon2_buffer_idx,
    size_t poseidon2_capacity,
    cudaStream_t stream
) {
    assert(num_leaves > 0);
    // poseidon2_capacity arrives from Rust in units of Fp elements; convert to record count.
    assert(poseidon2_capacity % 16 == 0 && "poseidon2_capacity must be a multiple of 16");
    size_t poseidon2_record_capacity = poseidon2_capacity / 16;
    uint32_t num_children = num_leaves;
    size_t const trace_height = [](uint32_t x) {
        return x ? (1u << (32 - __builtin_clz(x - 1))) : 0;
    }(unpadded_trace_height);

    {
        auto [grid, block] = kernel_launch_params(num_leaves, 256);
        prepare_for_updating<<<grid, block, 0, stream>>>(child_buf, layer, num_children);
        if (int err = CHECK_KERNEL(); err) {
            return err;
        }
    }

    uint32_t merkle_trace_offset = unpadded_trace_height;
    for (uint32_t h = 1; h <= subtree_height; ++h) {
        uint32_t* parent_ids = tmp_buf + 2 * num_children;
        // First, find for each child whether it has a different parent from the previous one
        {
            auto [grid, block] = kernel_launch_params(num_children);
            set_parent_id_adjacent_differences<<<grid, block, 0, stream>>>(child_buf, parent_ids, layer, num_children, h);
            if (int err = CHECK_KERNEL(); err) {
                return err;
            }
        }
        // Then, convert it to the partial sum
        {
            // Now, perform the inclusive sum in-place
            cub::DeviceScan::InclusiveSum(
                d_temp_storage, need_tmp_storage_bytes,
                parent_ids, parent_ids, num_children,
                stream
            );
            if (int err = CHECK_KERNEL(); err) {
                return err;
            }
        }
        // Finally, reorder the children
        {
            auto [grid, block] = kernel_launch_params(num_children);
            group_by_parent<<<grid, block, 0, stream>>>(
                child_buf,
                parent_ids,
                tmp_buf,
                layer,
                num_children,
                h
            );
            if (int err = CHECK_KERNEL(); err) {
                return err;
            }
        }
        bool const force_final = (num_subtrees == 1) && (h == subtree_height);
        // TODO: might need to think of better layout, or maybe different way of storing
        // Count this layer's dirty parents (each contributes a final row) and prefix-sum
        // so every dirty parent knows its final-row slot. Launched over `num_children`
        // with the parent count read on device, so a single sync fetches both counts.
        {
            auto [grid, block] = kernel_launch_params(num_children);
            mark_parent_dirty<<<grid, block, 0, stream>>>(
                tmp_buf, layer, dirty_buf, num_children, parent_ids + (num_children - 1)
            );
            if (int err = CHECK_KERNEL(); err) {
                return err;
            }
        }
        cub::DeviceScan::InclusiveSum(
            d_temp_storage, need_tmp_storage_bytes, dirty_buf, dirty_buf, num_children, stream
        );
        if (int err = CHECK_KERNEL(); err) {
            return err;
        }
        uint32_t num_parents = 0;
        uint32_t num_dirty = 0;
        cudaMemcpyAsync(
            &num_parents,
            parent_ids + (num_children - 1),
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            &num_dirty,
            dirty_buf + (num_children - 1),
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream
        );
        if (int err = CHECK_KERNEL(); err) {
            return err;
        }
        cudaStreamSynchronize(stream);
        ++num_parents;
        // A forced-final layer holds the single true root, which emits a final row even
        // when clean.
        uint32_t const layer_rows = num_parents + (force_final ? 1 : num_dirty);
        // The trace height is computed exactly on the host from the leaf records; a
        // mismatch here means that invariant broke, and writing would corrupt memory.
        if (layer_rows > merkle_trace_offset) {
            return -1;
        }
        merkle_trace_offset -= layer_rows;
        auto [grid, block] = kernel_launch_params(num_parents, 256);
        update_merkle_layer<<<grid, block, 0, stream>>>(
            h,
            zero_hash,
            actual_subtree_heights,
            subtree_layouts,
            initial_data_ptrs,
            subtree_height,
            layer,
            tmp_buf,
            child_buf,
            num_parents,
            subtrees,
            merkle_trace + merkle_trace_offset,
            trace_height,
            dirty_buf,
            force_final,
            d_poseidon2_raw_buffer,
            d_poseidon2_buffer_idx,
            poseidon2_record_capacity
        );
        num_children = num_parents;
    }
    update_to_root<<<1, 1, 0, stream>>>(
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
        poseidon2_record_capacity
    );

    return CHECK_KERNEL();
}
