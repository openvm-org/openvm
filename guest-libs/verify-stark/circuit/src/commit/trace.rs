use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::{
    circuit::subair::{generate_cols_from_leaf_children, MerkleTreeCols},
    utils::digests_to_poseidon2_input,
};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::prover::AirProvingContext;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

pub fn generate_proving_ctx(
    user_pvs: Vec<F>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
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
    let width = MerkleTreeCols::<u8>::width();
    let mut trace = vec![F::ZERO; rows.len() * width];

    for (chunk, row) in trace.chunks_mut(width).zip(rows) {
        let cols: &mut MerkleTreeCols<F> = chunk.borrow_mut();
        *cols = row;
    }

    let common_main = RowMajorMatrix::new(trace, width);
    let ctx = AirProvingContext::simple_no_pis(common_main);
    (ctx, poseidon2_compress_inputs)
}
