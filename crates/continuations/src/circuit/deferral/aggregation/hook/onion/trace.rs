use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, CpuBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::deferral::aggregation::hook::onion::air::OnionHashCols,
    utils::digests_to_poseidon2_input,
};

pub type IoCommit = ([F; DIGEST_SIZE], [F; DIGEST_SIZE]);

pub struct OnionTraceCtx {
    pub proving_ctx: AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    pub poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub input_onion: [F; DIGEST_SIZE],
    pub output_onion: [F; DIGEST_SIZE],
}

pub fn generate_proving_ctx(
    def_vk_commit: [F; DIGEST_SIZE],
    io_commits: Vec<IoCommit>,
) -> OnionTraceCtx {
    let num_commits = io_commits.len();
    debug_assert!(num_commits > 0);

    let width = OnionHashCols::<u8>::width();
    let height = (num_commits + 1).next_power_of_two();
    let mut trace = vec![F::ZERO; height * width];
    let mut poseidon2_inputs = Vec::with_capacity(2 * num_commits);

    let mut current_input_onion = def_vk_commit;
    let mut current_output_onion = [F::ZERO; DIGEST_SIZE];

    for (row_idx, (input_commit, output_commit)) in io_commits.iter().copied().enumerate() {
        let cols: &mut OnionHashCols<F> =
            trace[row_idx * width..(row_idx + 1) * width].borrow_mut();
        cols.row_idx = F::from_usize(row_idx);
        cols.is_valid = F::ONE;
        cols.is_first = F::from_bool(row_idx == 0);
        cols.input_commit = input_commit;
        cols.output_commit = output_commit;
        cols.input_onion = current_input_onion;
        cols.output_onion = current_output_onion;

        poseidon2_inputs.push(digests_to_poseidon2_input(
            current_input_onion,
            input_commit,
        ));
        poseidon2_inputs.push(digests_to_poseidon2_input(
            current_output_onion,
            output_commit,
        ));

        current_input_onion = poseidon2_compress_with_capacity(current_input_onion, input_commit).0;
        current_output_onion =
            poseidon2_compress_with_capacity(current_output_onion, output_commit).0;
    }

    // First invalid row stores the final onions so the final valid row can constrain its
    // transition.
    let first_invalid_row = num_commits;
    let cols: &mut OnionHashCols<F> =
        trace[first_invalid_row * width..(first_invalid_row + 1) * width].borrow_mut();
    cols.row_idx = F::from_usize(first_invalid_row);
    cols.input_onion = current_input_onion;
    cols.output_onion = current_output_onion;

    for row_idx in (first_invalid_row + 1)..height {
        let cols: &mut OnionHashCols<F> =
            trace[row_idx * width..(row_idx + 1) * width].borrow_mut();
        cols.row_idx = F::from_usize(row_idx);
    }

    OnionTraceCtx {
        proving_ctx: AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(
            &RowMajorMatrix::new(trace, width),
        )),
        poseidon2_inputs,
        input_onion: current_input_onion,
        output_onion: current_output_onion,
    }
}
