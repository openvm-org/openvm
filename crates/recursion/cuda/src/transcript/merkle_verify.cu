#include "fp.h"
#include "launcher.cuh"
#include "poseidon2-air/params.cuh"
#include "poseidon2.cuh"
#include "primitives/trace_access.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <driver_types.h>

using PoseidonParams = Poseidon2ParamsS1;
static constexpr size_t POSEIDON2_WIDTH = 16;

template <typename T> struct MerkleVerifyCols {
    T proof_idx;
    T is_proof_start;
    T merkle_proof_idx;
    T is_valid;
    T is_first_merkle;
    T is_last_merkle;

    T is_combining_leaves;
    T leaf_sub_idx;

    T idx;
    T idx_parity;
    T total_depth;
    T height;

    T left[DIGEST_SIZE];
    T right[DIGEST_SIZE];

    T recv_left;
    T recv_right;

    T commit_major;
    T commit_minor;

    T output[POSEIDON2_WIDTH];
};

struct CombinationIndices {
    size_t source_layer;
    size_t left_source_index;
    size_t right_source_index;
    size_t result_layer;
    size_t result_index;
};

__device__ __forceinline__ Fp bool_to_fp(bool value) { return value ? Fp::one() : Fp::zero(); }

__device__ bool compute_combination_indices(size_t k, size_t idx, CombinationIndices &out) {
    if (k == 0) {
        return false;
    }
    size_t total_ops = (size_t(1) << k) - 1;
    if (idx >= total_ops) {
        return false;
    }
    size_t current = idx;
    size_t layer = 0;
    while (layer < k) {
        size_t exponent = k - (layer + 1);
        size_t ops_in_layer = size_t(1) << exponent;
        if (current < ops_in_layer) {
            size_t index_within_layer = current;
            out.source_layer = layer;
            out.left_source_index = index_within_layer * 2;
            out.right_source_index = index_within_layer * 2 + 1;
            out.result_layer = layer + 1;
            out.result_index = index_within_layer;
            return true;
        }
        current -= ops_in_layer;
        layer += 1;
    }
    return false;
}

__device__ __forceinline__ void copy_digest(Fp *dst, const Fp *src) {
#pragma unroll
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        dst[i] = src[i];
    }
}

template <size_t SBOX_REGS>
__global__ void cukernel_merkle_verify_tracegen(
    Fp *d_trace,
    size_t trace_height,
    size_t trace_width,
    const MerkleVerifyRecord *records,
    size_t num_records,
    const Fp *leaf_hashes,
    const Fp *siblings,
    size_t num_leaves,
    size_t k,
    Fp *poseidon_inputs,
    size_t num_valid_rows,
    const size_t *proof_row_starts,
    size_t num_proofs,
    Fp *leaf_scratch
) {
    size_t rec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rec_idx >= num_records) {
        return;
    }
    const MerkleVerifyRecord record = records[rec_idx];

    size_t leaf_stride = num_leaves * DIGEST_SIZE;
    Fp *leaf_layer = leaf_scratch + rec_idx * leaf_stride;
    const Fp *rec_leaf_hashes = leaf_hashes + record.leaf_hash_offset;
    for (size_t leaf = 0; leaf < num_leaves; ++leaf) {
        copy_digest(leaf_layer + leaf * DIGEST_SIZE, rec_leaf_hashes + leaf * DIGEST_SIZE);
    }

    Fp current_hash[DIGEST_SIZE];
    size_t current_idx = 0;
    bool current_hash_valid = false;

    for (size_t local_row = 0; local_row < record.num_rows; ++local_row) {
        size_t global_row = record.start_row + local_row;
        if (global_row >= num_valid_rows) {
            break;
        }
        RowSlice row(d_trace + global_row, trace_height);
        COL_WRITE_VALUE(
            row, MerkleVerifyCols, proof_idx, Fp(static_cast<uint32_t>(record.proof_idx))
        );
        bool proof_start =
            record.proof_idx < num_proofs && global_row == proof_row_starts[record.proof_idx];
        COL_WRITE_VALUE(row, MerkleVerifyCols, is_proof_start, bool_to_fp(proof_start));
        COL_WRITE_VALUE(
            row,
            MerkleVerifyCols,
            merkle_proof_idx,
            Fp(static_cast<uint32_t>(record.merkle_proof_idx))
        );
        COL_WRITE_VALUE(row, MerkleVerifyCols, is_valid, Fp::one());
        COL_WRITE_VALUE(row, MerkleVerifyCols, is_first_merkle, bool_to_fp(local_row == 0));
        COL_WRITE_VALUE(
            row, MerkleVerifyCols, is_last_merkle, bool_to_fp(local_row + 1 == record.num_rows)
        );
        COL_WRITE_VALUE(
            row, MerkleVerifyCols, commit_major, Fp(static_cast<uint32_t>(record.commit_major))
        );
        COL_WRITE_VALUE(
            row, MerkleVerifyCols, commit_minor, Fp(static_cast<uint32_t>(record.commit_minor))
        );
        COL_WRITE_VALUE(
            row, MerkleVerifyCols, total_depth, Fp(static_cast<uint32_t>(record.depth + k + 1))
        );

        Fp poseidon_state[POSEIDON2_WIDTH];
        bool is_combining = local_row < num_leaves - 1;

        if (is_combining) {
            CombinationIndices indices;
            bool has_indices = compute_combination_indices(k, local_row, indices);
            if (!has_indices) {
                continue;
            }
            copy_digest(poseidon_state, leaf_layer + indices.left_source_index * DIGEST_SIZE);
            copy_digest(
                poseidon_state + DIGEST_SIZE, leaf_layer + indices.right_source_index * DIGEST_SIZE
            );
            COL_WRITE_ARRAY(row, MerkleVerifyCols, left, poseidon_state);
            COL_WRITE_ARRAY(row, MerkleVerifyCols, right, poseidon_state + DIGEST_SIZE);
#pragma unroll
            for (size_t col = 0; col < POSEIDON2_WIDTH; ++col) {
                poseidon_inputs[global_row * POSEIDON2_WIDTH + col] = poseidon_state[col];
            }
            poseidon2::poseidon2_mix(poseidon_state);
            copy_digest(leaf_layer + indices.result_index * DIGEST_SIZE, poseidon_state);
            COL_WRITE_VALUE(row, MerkleVerifyCols, is_combining_leaves, Fp::one());
            COL_WRITE_VALUE(
                row, MerkleVerifyCols, leaf_sub_idx, Fp(static_cast<uint32_t>(indices.result_index))
            );
            COL_WRITE_VALUE(
                row, MerkleVerifyCols, idx, Fp(static_cast<uint32_t>(record.merkle_idx))
            );
            COL_WRITE_VALUE(
                row, MerkleVerifyCols, idx_parity, Fp(static_cast<uint32_t>(record.merkle_idx & 1))
            );
            COL_WRITE_VALUE(
                row, MerkleVerifyCols, height, Fp(static_cast<uint32_t>(indices.source_layer))
            );
            COL_WRITE_VALUE(row, MerkleVerifyCols, recv_left, Fp::one());
            COL_WRITE_VALUE(row, MerkleVerifyCols, recv_right, Fp::one());
        } else {
            size_t pos = local_row + 1 - num_leaves;
            const Fp *sibling = siblings + record.siblings_offset + pos * DIGEST_SIZE;
            if (!current_hash_valid) {
#pragma unroll
                for (size_t limb = 0; limb < DIGEST_SIZE; ++limb) {
                    current_hash[limb] = leaf_layer[limb];
                }
                current_idx = record.merkle_idx;
                current_hash_valid = true;
            }
            bool left_is_cur = (current_idx % 2) == 0;
#pragma unroll
            for (size_t limb = 0; limb < DIGEST_SIZE; ++limb) {
                poseidon_state[limb] = left_is_cur ? current_hash[limb] : sibling[limb];
                poseidon_state[limb + DIGEST_SIZE] =
                    left_is_cur ? sibling[limb] : current_hash[limb];
            }
            COL_WRITE_ARRAY(row, MerkleVerifyCols, left, poseidon_state);
            COL_WRITE_ARRAY(row, MerkleVerifyCols, right, poseidon_state + DIGEST_SIZE);
#pragma unroll
            for (size_t col = 0; col < POSEIDON2_WIDTH; ++col) {
                poseidon_inputs[global_row * POSEIDON2_WIDTH + col] = poseidon_state[col];
            }
            poseidon2::poseidon2_mix(poseidon_state);
            copy_digest(current_hash, poseidon_state);
            COL_WRITE_VALUE(row, MerkleVerifyCols, is_combining_leaves, Fp::zero());
            COL_WRITE_VALUE(row, MerkleVerifyCols, leaf_sub_idx, Fp::zero());
            COL_WRITE_VALUE(row, MerkleVerifyCols, idx, Fp(static_cast<uint32_t>(current_idx)));
            COL_WRITE_VALUE(
                row, MerkleVerifyCols, idx_parity, Fp(static_cast<uint32_t>(current_idx & 1))
            );
            COL_WRITE_VALUE(row, MerkleVerifyCols, height, Fp(static_cast<uint32_t>(pos + k)));
            COL_WRITE_VALUE(row, MerkleVerifyCols, recv_left, bool_to_fp(left_is_cur));
            COL_WRITE_VALUE(row, MerkleVerifyCols, recv_right, bool_to_fp(!left_is_cur));
            current_idx >>= 1;
        }
        COL_WRITE_ARRAY(row, MerkleVerifyCols, output, poseidon_state);
    }
}

extern "C" int _merkle_verify_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    const MerkleVerifyRecord *records,
    size_t num_records,
    const Fp *leaf_hashes,
    const Fp *siblings,
    size_t num_leaves,
    size_t k,
    Fp *poseidon_inputs,
    size_t num_valid_rows,
    const size_t *proof_row_starts,
    size_t num_proofs,
    Fp *leaf_scratch
) {
    if (num_records == 0 || num_valid_rows == 0) {
        return cudaSuccess;
    }
    auto [grid, block] = kernel_launch_params(num_records, 512);
    cukernel_merkle_verify_tracegen<PoseidonParams::SBOX_REGS><<<grid, block>>>(
        d_trace,
        height,
        width,
        records,
        num_records,
        leaf_hashes,
        siblings,
        num_leaves,
        k,
        poseidon_inputs,
        num_valid_rows,
        proof_row_starts,
        num_proofs,
        leaf_scratch
    );
    return CHECK_KERNEL();
}
