use std::{array::from_fn, borrow::BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::{
    deferral::verify::output::{DeferralOutputCommitCols, F_NUM_BYTES, VALS_IN_DIGEST},
    root::digests_to_poseidon2_input,
};

pub fn generate_proving_ctx(
    app_exe_commit: [F; DIGEST_SIZE],
    app_vk_commit: [F; DIGEST_SIZE],
    user_pvs: Vec<F>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    debug_assert!(DIGEST_SIZE.is_multiple_of(F_NUM_BYTES));
    debug_assert!(DIGEST_SIZE.is_multiple_of(VALS_IN_DIGEST));
    debug_assert!(user_pvs.len().is_multiple_of(VALS_IN_DIGEST));

    let mut next_f_rows = values_to_rows(&app_exe_commit);
    next_f_rows.extend(values_to_rows(&app_vk_commit));
    next_f_rows.extend(values_to_rows(&user_pvs));
    debug_assert!(!next_f_rows.is_empty());

    // One trailing invalid row is required so OutputCommitBus can receive the final state
    let min_height = next_f_rows.len() + 1;
    let height = min_height.next_power_of_two();
    let width = DeferralOutputCommitCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);

    for (row_idx, next_f) in next_f_rows.iter().copied().enumerate() {
        let row = chunks.next().unwrap();
        let cols: &mut DeferralOutputCommitCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(row_idx == 0);
        cols.next_f = next_f;
        cols.next_f_idx = F::from_usize(row_idx);
        cols.next_bytes = next_f_to_digest(next_f);
    }

    for row_idx in next_f_rows.len()..height {
        let row = chunks.next().unwrap();
        let cols: &mut DeferralOutputCommitCols<F> = row.borrow_mut();
        cols.next_f_idx = F::from_usize(row_idx);
    }

    let mut poseidon2_compress_inputs = Vec::with_capacity(next_f_rows.len().saturating_sub(1));
    let mut state = next_f_to_digest(next_f_rows[0]);

    if height > 1 {
        let row_1: &mut DeferralOutputCommitCols<F> = trace[width..2 * width].borrow_mut();
        row_1.state = state;
    }

    for &next_f in next_f_rows.iter().skip(1) {
        let next_bytes = next_f_to_digest(next_f);
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(state, next_bytes));
        state = poseidon2_compress_with_capacity(state, next_bytes).0;
    }

    if next_f_rows.len() < height {
        let first_invalid_row: &mut DeferralOutputCommitCols<F> =
            trace[next_f_rows.len() * width..(next_f_rows.len() + 1) * width].borrow_mut();
        first_invalid_row.state = state;
    }

    (
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&RowMajorMatrix::new(
            trace, width,
        ))),
        poseidon2_compress_inputs,
    )
}

fn values_to_rows(values: &[F]) -> Vec<[F; VALS_IN_DIGEST]> {
    values
        .chunks_exact(VALS_IN_DIGEST)
        .map(|chunk| chunk.try_into().unwrap())
        .collect()
}

fn next_f_to_digest(next_f: [F; VALS_IN_DIGEST]) -> [F; DIGEST_SIZE] {
    from_fn(|byte_idx| {
        let f_idx = byte_idx / F_NUM_BYTES;
        let byte_in_f = byte_idx % F_NUM_BYTES;
        let f_u32 = next_f[f_idx].as_canonical_u32();
        F::from_u8(f_u32.to_le_bytes()[byte_in_f])
    })
}
