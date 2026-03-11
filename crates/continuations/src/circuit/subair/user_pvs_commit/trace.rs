use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;

use super::UserPvsCommitCols;
use crate::utils::digests_to_poseidon2_input;

/// Build `UserPvsCommitCols` rows from raw user public values.
///
/// Each leaf consumes one digest (`DIGEST_SIZE` values), where the right child is zero.
/// Returns all rows (including the terminal invalid row) and Poseidon2 lookup inputs.
pub fn generate_cols_from_user_pvs(
    user_pvs: &[F],
) -> (Vec<UserPvsCommitCols<F>>, Vec<[F; POSEIDON2_WIDTH]>) {
    // Each leaf consumes `DIGEST_SIZE` public values, which is padded and hashed before
    // being inserted into the Merkle tree. We require at least one leaf (so at least one
    // Poseidon2 hash), and a full binary tree.
    debug_assert!(user_pvs.len() >= DIGEST_SIZE);
    debug_assert!(user_pvs.len().is_multiple_of(DIGEST_SIZE));
    debug_assert!((user_pvs.len() / DIGEST_SIZE).is_power_of_two());

    let num_leaves = user_pvs.len() / DIGEST_SIZE;
    let mut rows = Vec::with_capacity(2 * num_leaves);
    let mut next_layer = Vec::with_capacity(num_leaves);
    let mut poseidon2_compress_inputs = Vec::with_capacity(2 * num_leaves - 1);

    for (row_idx, pv_digest) in user_pvs.chunks(DIGEST_SIZE).enumerate() {
        let left: [F; DIGEST_SIZE] = pv_digest[..DIGEST_SIZE].try_into().unwrap();
        let right: [F; DIGEST_SIZE] = [F::ZERO; DIGEST_SIZE];
        let parent = poseidon2_compress_with_capacity(left, right).0;
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

        rows.push(UserPvsCommitCols {
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

    let mut row_idx = next_layer.len();
    while next_layer.len() > 1 {
        let parent_layer_len = next_layer.len() >> 1;
        for parent_idx in 0..parent_layer_len {
            let left = next_layer[2 * parent_idx];
            let right = next_layer[2 * parent_idx + 1];
            let parent = poseidon2_compress_with_capacity(left, right).0;
            poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

            rows.push(UserPvsCommitCols {
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

    rows.push(UserPvsCommitCols {
        row_idx: F::from_usize(row_idx),
        send_type: F::ZERO,
        receive_type: F::ZERO,
        parent: [F::ZERO; DIGEST_SIZE],
        is_right_child: F::ONE,
        left_child: [F::ZERO; DIGEST_SIZE],
        right_child: [F::ZERO; DIGEST_SIZE],
    });

    (rows, poseidon2_compress_inputs)
}
