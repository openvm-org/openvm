use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{prover::AirProvingContext, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::{
        root::commit::{MerkleTreeCols, MAX_ENCODER_DEGREE},
        subair::generate_cols_from_leaf_children,
    },
    utils::digests_to_poseidon2_input,
};

pub fn generate_proving_ctx<SC: StarkProtocolConfig<F = F>>(
    user_pvs: Vec<F>,
) -> (AirProvingContext<CpuBackend<SC>>, Vec<[F; POSEIDON2_WIDTH]>) {
    let num_user_pvs = user_pvs.len();

    // Each leaf consumes `DIGEST_SIZE` public values, which is padded and hashed before
    // being inserted into the Merkle tree. We require at least one leaf (so at least one
    // Poseidon2 hash), and a full binary tree.
    debug_assert!(num_user_pvs >= DIGEST_SIZE);
    debug_assert!(num_user_pvs.is_multiple_of(DIGEST_SIZE));
    debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

    // One selector per leaf PV chunk (each leaf consumes 1 digest).
    let encoder = Encoder::new(num_user_pvs / DIGEST_SIZE, MAX_ENCODER_DEGREE, true);

    let num_leaves = num_user_pvs / DIGEST_SIZE;
    let leaf_children = user_pvs
        .chunks_exact(DIGEST_SIZE)
        .map(|digest| {
            (
                digest.try_into().expect("digest sized chunk"),
                [F::ZERO; DIGEST_SIZE],
            )
        })
        .collect();
    let rows = generate_cols_from_leaf_children(leaf_children);
    let poseidon2_compress_inputs = rows
        .iter()
        .take(rows.len() - 1)
        .map(|row| digests_to_poseidon2_input(row.left_child, row.right_child))
        .collect();
    debug_assert_eq!(rows.len(), 2 * num_leaves);

    let const_width = MerkleTreeCols::<u8>::width();
    let width = const_width + encoder.width();
    let mut trace = vec![F::ZERO; rows.len() * width];

    for (row_idx, (chunk, row)) in trace.chunks_mut(width).zip(rows).enumerate() {
        let cols: &mut MerkleTreeCols<F> = chunk[..const_width].borrow_mut();
        *cols = row;

        if row_idx < num_leaves {
            chunk[const_width..].copy_from_slice(
                encoder
                    .get_flag_pt(row_idx)
                    .into_iter()
                    .map(F::from_u32)
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
        }
    }

    let common_main = RowMajorMatrix::new(trace, width);
    let ctx = AirProvingContext::simple(common_main, user_pvs);
    (ctx, poseidon2_compress_inputs)
}
