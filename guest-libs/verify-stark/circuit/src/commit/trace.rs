use std::borrow::BorrowMut;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::circuit::subair::{generate_cols_from_user_pvs, UserPvsCommitCols};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::prover::AirProvingContext;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

pub fn generate_proving_ctx(
    user_pvs: Vec<F>,
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
) {
    let (rows, poseidon2_compress_inputs) = generate_cols_from_user_pvs(&user_pvs);
    let width = UserPvsCommitCols::<u8>::width();
    let mut trace = vec![F::ZERO; rows.len() * width];

    for (chunk, row) in trace.chunks_mut(width).zip(rows) {
        let cols: &mut UserPvsCommitCols<F> = chunk.borrow_mut();
        *cols = row;
    }

    let common_main = RowMajorMatrix::new(trace, width);
    let ctx = AirProvingContext::simple_no_pis(common_main);
    (ctx, poseidon2_compress_inputs)
}
