use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;

use super::MerkleTreeCols;
use crate::circuit::deferral::{DEF_INTERNAL_TAG, DEF_LEAF_TAG};

/// Build Merkle-tree rows from leaf `(left_child, right_child)` inputs using
/// Poseidon2 compression over BabyBear.
///
/// The number of leaf rows must be a power of two and non-zero.
/// The returned vector has length `2 * num_leaves`, with the final row being the
/// terminal row (`send_type = receive_type = 0`).
pub fn generate_cols_from_leaf_children(
    leaf_children: Vec<([F; DIGEST_SIZE], [F; DIGEST_SIZE])>,
    tagged: bool,
) -> Vec<MerkleTreeCols<F>> {
    let num_leaves = leaf_children.len();
    debug_assert!(num_leaves > 0);
    debug_assert!(num_leaves.is_power_of_two());

    let compress = |left: [F; DIGEST_SIZE], right: [F; DIGEST_SIZE], tag: [u8; DIGEST_SIZE]| {
        let left = if tagged {
            poseidon2_compress_with_capacity(tag.map(F::from_u8), left).0
        } else {
            left
        };
        poseidon2_compress_with_capacity(left, right).0
    };

    let mut rows = Vec::with_capacity(2 * num_leaves);
    let mut next_layer = Vec::with_capacity(num_leaves);

    for (row_idx, (left, right)) in leaf_children.into_iter().enumerate() {
        let parent = compress(left, right, DEF_LEAF_TAG);
        rows.push(MerkleTreeCols {
            row_idx: F::from_usize(row_idx),
            send_type: if num_leaves == 1 { F::TWO } else { F::ONE },
            receive_type: F::ONE,
            parent,
            is_right_child: F::from_bool(row_idx & 1 == 1),
            left_child: left,
            right_child: right,
        });
        next_layer.push(parent);
    }

    let mut row_idx = num_leaves;
    while next_layer.len() > 1 {
        let parent_layer_len = next_layer.len() >> 1;
        for parent_idx in 0..parent_layer_len {
            let left = next_layer[2 * parent_idx];
            let right = next_layer[2 * parent_idx + 1];
            let parent = compress(left, right, DEF_INTERNAL_TAG);

            rows.push(MerkleTreeCols {
                row_idx: F::from_usize(row_idx),
                send_type: if parent_layer_len > 1 { F::ONE } else { F::TWO },
                receive_type: F::TWO,
                parent,
                is_right_child: F::from_bool(row_idx & 1 == 1),
                left_child: left,
                right_child: right,
            });

            row_idx += 1;
            next_layer[parent_idx] = parent;
        }
        next_layer.truncate(parent_layer_len);
    }

    rows.push(MerkleTreeCols {
        row_idx: F::from_usize(row_idx),
        send_type: F::ZERO,
        receive_type: F::ZERO,
        parent: [F::ZERO; DIGEST_SIZE],
        is_right_child: F::from_bool(row_idx & 1 == 1),
        left_child: [F::ZERO; DIGEST_SIZE],
        right_child: [F::ZERO; DIGEST_SIZE],
    });

    rows
}
