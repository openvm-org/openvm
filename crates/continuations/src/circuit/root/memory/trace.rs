use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::PUBLIC_VALUES_AS},
};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    p3_util::log2_strict_usize, prover::AirProvingContext, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{circuit::root::memory::UserPvsInMemoryCols, utils::digests_to_poseidon2_input};

pub fn generate_proving_input<SC: StarkProtocolConfig<F = F>>(
    user_pv_commit: [F; DIGEST_SIZE],
    merkle_proof: &[[F; DIGEST_SIZE]],
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
) -> (AirProvingContext<CpuBackend<SC>>, Vec<[F; POSEIDON2_WIDTH]>) {
    let merkle_proof_len = merkle_proof.len();
    let num_layers = merkle_proof_len + 1;
    let height = num_layers.next_power_of_two();
    let width = UserPvsInMemoryCols::<u8>::width();

    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);
    let mut current = user_pv_commit;

    /*
     * We can determine the public values' location in memory (and thus location in
     * the memory merkle tree) from PUBLIC_VALUES_AS, the memory dimensions, and the
     * number of user public values.
     */
    let pv_start_idx = memory_dimensions.label_to_index((PUBLIC_VALUES_AS, 0));
    let pv_height = log2_strict_usize(num_user_pvs / DIGEST_SIZE);
    let merkle_path_branch_bits = pv_start_idx >> pv_height;
    let mut current_branch_bits = 0;

    let mut poseidon2_compress_inputs = Vec::with_capacity(merkle_proof_len);

    for (i, &sibling) in merkle_proof.iter().enumerate() {
        let chunk = chunks.next().unwrap();
        let cols: &mut UserPvsInMemoryCols<F> = chunk.borrow_mut();
        let is_right_child = merkle_path_branch_bits & (1 << i) != 0;
        current_branch_bits += (is_right_child as usize) << i;

        cols.is_valid = if i == 0 { F::TWO } else { F::ONE };
        cols.is_right_child = F::from_bool(is_right_child);
        cols.node_commit = current;
        cols.sibling = sibling;
        cols.row_idx_exp_2 = F::from_usize(1 << i);
        cols.merkle_path_branch_bits = F::from_usize(current_branch_bits);

        let left = if is_right_child { sibling } else { current };
        let right = if is_right_child { current } else { sibling };
        current = poseidon2_compress_with_capacity(left, right).0;
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));
    }

    let last_chunk = chunks.next().unwrap();
    let last_row: &mut UserPvsInMemoryCols<F> = last_chunk.borrow_mut();
    last_row.is_valid = F::ONE;
    last_row.node_commit = current;
    last_row.row_idx_exp_2 = F::from_usize(1 << merkle_proof_len);
    last_row.merkle_path_branch_bits = F::from_usize(current_branch_bits);

    (
        AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width)),
        poseidon2_compress_inputs,
    )
}
