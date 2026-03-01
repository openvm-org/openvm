use std::borrow::BorrowMut;

use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::nonroot::unset::UnsetPvsCols;

pub fn generate_proving_ctx(
    unset_proof_idxs: &[usize],
    child_is_app: bool,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_valid = if child_is_app {
        0
    } else {
        unset_proof_idxs.len()
    };

    let height = num_valid.next_power_of_two();
    let width = UnsetPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);

    for i in 0..num_valid {
        let chunk = chunks.next().unwrap();
        let cols: &mut UnsetPvsCols<F> = chunk.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_usize(unset_proof_idxs[i]);
    }

    AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&RowMajorMatrix::new(
        trace, width,
    )))
}
