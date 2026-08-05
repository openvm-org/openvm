use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{POSEIDON2_WIDTH, U16_CELLS_PER_PUBLIC_VALUE},
    system::public_values::{public_values_event_block_from_limbs, public_values_initial_commit},
};
use openvm_continuations::utils::digests_to_poseidon2_input;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::prover::AirProvingContext;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use super::UserPvsCommitValuesCols;

pub fn generate_proving_ctx(
    user_pvs: &[F],
    num_values: usize,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    assert!(user_pvs.len().is_multiple_of(U16_CELLS_PER_PUBLIC_VALUE));
    let capacity = user_pvs.len() / U16_CELLS_PER_PUBLIC_VALUE;
    assert!(capacity.is_power_of_two());
    assert!(num_values <= capacity);

    let width = UserPvsCommitValuesCols::<u8>::width();
    let mut trace = vec![F::ZERO; capacity * width];
    let mut commit = public_values_initial_commit::<F>(capacity);
    let mut poseidon2_compress_inputs = Vec::with_capacity(num_values);

    for (row_idx, (row, value)) in trace
        .chunks_exact_mut(width)
        .zip(user_pvs.chunks_exact(U16_CELLS_PER_PUBLIC_VALUE))
        .enumerate()
    {
        let cols: &mut UserPvsCommitValuesCols<F> = row.borrow_mut();
        cols.is_valid = F::from_bool(row_idx < num_values);
        cols.is_last = F::from_bool(row_idx + 1 == capacity);
        cols.row_idx = F::from_usize(row_idx);
        cols.len = F::from_usize(row_idx.min(num_values));
        cols.value.copy_from_slice(value);
        cols.commit_before = commit;

        if row_idx < num_values {
            let event = public_values_event_block_from_limbs(
                value.try_into().expect("public value has four limbs"),
            );
            let input = digests_to_poseidon2_input(commit, event);
            commit = poseidon2_compress_with_capacity(commit, event).0;
            poseidon2_compress_inputs.push(input);
        }
        cols.commit_after = commit;
    }

    (
        AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width)),
        poseidon2_compress_inputs,
    )
}
