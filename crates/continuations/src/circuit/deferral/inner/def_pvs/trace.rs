use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
};

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    circuit::deferral::{
        inner::def_pvs::air::DeferralAggPvsCols, DeferralAggregationPvs, DeferralCircuitPvs,
        DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::zero_hash,
};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_merkle_depth: Option<usize>,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    let is_wrapper = child_merkle_depth.is_none();
    debug_assert!(!is_wrapper || num_proofs == 1);

    let num_rows = if is_wrapper { 1usize } else { 2usize };
    let width = DeferralAggPvsCols::<u8>::width();

    debug_assert!((1..=2).contains(&num_proofs));

    let mut trace = vec![F::ZERO; num_rows * width];
    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut DeferralAggPvsCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_present = F::ONE;
        cols.has_verifier_pvs = F::from_bool(child_is_def);

        if child_is_def {
            let child_pvs: &DeferralAggregationPvs<F> =
                proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();
            cols.merkle_commit = child_pvs.merkle_commit;
        } else {
            let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
                .as_slice()
                .borrow();
            let commit_values = once(child_pvs.input_commit)
                .chain(
                    proof
                        .trace_vdata
                        .iter()
                        .flatten()
                        .flat_map(|vdata| vdata.cached_commitments.iter().copied()),
                )
                .flatten()
                .collect_vec();
            let folded_input_commit = poseidon2_hash_slice(&commit_values).0;
            cols.child_pvs = DeferralCircuitPvs {
                input_commit: folded_input_commit,
                output_commit: child_pvs.output_commit,
            };
            cols.merkle_commit =
                poseidon2_compress_with_capacity(folded_input_commit, child_pvs.output_commit).0;
        }
    }

    if num_rows == 2 && num_proofs == 1 {
        let cols: &mut DeferralAggPvsCols<F> = trace[width..2 * width].borrow_mut();
        cols.proof_idx = F::ONE;
        cols.has_verifier_pvs = F::from_bool(child_is_def);
        cols.merkle_commit = zero_hash(child_merkle_depth.unwrap() + 1);
    }

    let mut public_values = vec![F::ZERO; DeferralAggregationPvs::<u8>::width()];
    let pvs: &mut DeferralAggregationPvs<F> = public_values.as_mut_slice().borrow_mut();

    let first_row: &DeferralAggPvsCols<F> = trace[..width].borrow();
    if is_wrapper {
        pvs.merkle_commit = first_row.merkle_commit;
    } else {
        let second_row: &DeferralAggPvsCols<F> = trace[width..2 * width].borrow();
        pvs.merkle_commit =
            poseidon2_compress_with_capacity(first_row.merkle_commit, second_row.merkle_commit).0;
    }

    AirProvingContext {
        cached_mains: vec![],
        common_main: RowMajorMatrix::new(trace, width),
        public_values,
    }
}
