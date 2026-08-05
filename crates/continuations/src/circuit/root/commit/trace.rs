use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{POSEIDON2_WIDTH, U16_CELLS_PER_PUBLIC_VALUE},
    system::public_values::{public_values_event_block_from_limbs, public_values_initial_commit},
};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{prover::AirProvingContext, StarkProtocolConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_compress_with_capacity, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use super::{UserPvsCommitCols, MAX_ENCODER_DEGREE};
use crate::utils::digests_to_poseidon2_input;

pub fn generate_proving_ctx<SC: StarkProtocolConfig<F = F>>(
    user_pvs: Vec<F>,
    num_values: usize,
) -> (AirProvingContext<CpuBackend<SC>>, Vec<[F; POSEIDON2_WIDTH]>) {
    assert!(user_pvs.len().is_multiple_of(U16_CELLS_PER_PUBLIC_VALUE));
    let capacity = user_pvs.len() / U16_CELLS_PER_PUBLIC_VALUE;
    assert!(capacity.is_power_of_two());
    assert!(num_values <= capacity);

    let encoder = Encoder::new(capacity, MAX_ENCODER_DEGREE, true);
    let cols_width = UserPvsCommitCols::<u8>::width();
    let width = cols_width + encoder.width();
    let mut trace = vec![F::ZERO; capacity * width];
    let mut commit = public_values_initial_commit::<F>(capacity);
    let mut poseidon2_compress_inputs = Vec::with_capacity(num_values);

    for (row_idx, (row, value)) in trace
        .chunks_exact_mut(width)
        .zip(user_pvs.chunks_exact(U16_CELLS_PER_PUBLIC_VALUE))
        .enumerate()
    {
        let cols: &mut UserPvsCommitCols<F> = row[..cols_width].borrow_mut();
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

        row[cols_width..].copy_from_slice(
            &encoder
                .get_flag_pt(row_idx)
                .into_iter()
                .map(F::from_u32)
                .collect::<Vec<_>>(),
        );
    }

    let public_values = std::iter::once(F::from_usize(num_values))
        .chain(user_pvs)
        .collect();
    (
        AirProvingContext::simple(RowMajorMatrix::new(trace, width), public_values),
        poseidon2_compress_inputs,
    )
}
