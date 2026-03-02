use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::{
    root::commit::{UserPvsCommitCols, MAX_ENCODER_DEGREE},
    subair::generate_cols_from_user_pvs,
};

pub fn generate_proving_ctx(
    user_pvs: Vec<F>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
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
    let (rows, poseidon2_compress_inputs) = generate_cols_from_user_pvs(&user_pvs);
    debug_assert_eq!(rows.len(), 2 * num_leaves);

    let const_width = UserPvsCommitCols::<u8>::width();
    let width = const_width + encoder.width();
    let mut trace = vec![F::ZERO; rows.len() * width];

    for (row_idx, (chunk, row)) in trace.chunks_mut(width).zip(rows).enumerate() {
        let cols: &mut UserPvsCommitCols<F> = chunk[..const_width].borrow_mut();
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

    let common_main = ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width));
    let ctx = AirProvingContext::simple(common_main, user_pvs);
    (ctx, poseidon2_compress_inputs)
}
