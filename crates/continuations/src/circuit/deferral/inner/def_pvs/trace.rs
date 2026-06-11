use std::{
    borrow::{Borrow, BorrowMut},
    iter::once,
};

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::utils::poseidon2_hash_slice;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::deferral::{
    inner::def_pvs::air::{DeferralAggPvsAir, DeferralAggPvsCols},
    utils::{def_internal_compress, def_leaf_compress, def_zero_hash},
    DeferralAggregationPvs, DeferralCircuitPvs, DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    MAX_DEF_AGG_MERKLE_DEPTH,
};

pub struct DeferralAggPvsTraceCtx {
    pub proving_ctx: AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    pub range_check_inputs: Vec<usize>,
}

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_agg: bool,
    child_merkle_depth: Option<usize>,
) -> DeferralAggPvsTraceCtx {
    let num_proofs = proofs.len();
    let is_wrapper = child_merkle_depth.is_none();
    debug_assert!(!is_wrapper || num_proofs == 1);

    let num_rows = if is_wrapper { 1usize } else { 2usize };
    let width = DeferralAggPvsCols::<u8>::width();

    debug_assert!((1..=2).contains(&num_proofs));

    let mut trace = vec![F::ZERO; num_rows * width];
    let mut num_def_circuit_proofs = F::ZERO;
    let mut merkle_depth = F::ZERO;
    let mut def_idx = F::ZERO;

    for (proof_idx, (proof, chunk)) in proofs.iter().zip(trace.chunks_exact_mut(width)).enumerate()
    {
        let cols: &mut DeferralAggPvsCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_present = F::ONE;
        cols.has_verifier_pvs = F::from_bool(child_is_agg);

        if child_is_agg {
            let child_pvs: &DeferralAggregationPvs<F> =
                proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();
            cols.merkle_commit = child_pvs.merkle_commit;
            cols.child_pvs.input_commit[0] = child_pvs.num_def_circuit_proofs;
            cols.child_pvs.input_commit[1] = child_pvs.merkle_depth;
            cols.child_pvs.def_idx = child_pvs.def_idx;
            num_def_circuit_proofs += child_pvs.num_def_circuit_proofs;
            if proof_idx == 0 {
                merkle_depth = child_pvs.merkle_depth;
                def_idx = child_pvs.def_idx;
            } else {
                debug_assert_eq!(merkle_depth, child_pvs.merkle_depth);
                debug_assert_eq!(def_idx, child_pvs.def_idx);
            }
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
                def_idx: child_pvs.def_idx,
            };
            let (tagged_input_commit, merkle_commit) =
                def_leaf_compress(folded_input_commit, child_pvs.output_commit);
            cols.tagged_input_commit = tagged_input_commit;
            cols.merkle_commit = merkle_commit;
            num_def_circuit_proofs += F::ONE;
            if proof_idx == 0 {
                def_idx = child_pvs.def_idx;
            } else {
                debug_assert_eq!(def_idx, child_pvs.def_idx);
            }
        }
    }

    if num_rows == 2 && num_proofs == 1 {
        let cols: &mut DeferralAggPvsCols<F> = trace[width..2 * width].borrow_mut();
        cols.proof_idx = F::ONE;
        cols.has_verifier_pvs = F::from_bool(child_is_agg);
        for (dst, value) in cols
            .merkle_commit
            .iter_mut()
            .zip(DeferralAggPvsAir::depth_encoder().get_flag_pt(child_merkle_depth.unwrap()))
        {
            *dst = F::from_u32(value);
        }
    }

    let mut public_values = vec![F::ZERO; DeferralAggregationPvs::<u8>::width()];
    let pvs: &mut DeferralAggregationPvs<F> = public_values.as_mut_slice().borrow_mut();

    if is_wrapper {
        let first_row: &DeferralAggPvsCols<F> = trace[..width].borrow();
        pvs.merkle_commit = first_row.merkle_commit;
        pvs.merkle_depth = merkle_depth;
    } else {
        let right_child = if num_proofs == 1 {
            def_zero_hash(child_merkle_depth.unwrap() + 1)
        } else {
            let second_row: &DeferralAggPvsCols<F> = trace[width..2 * width].borrow();
            second_row.merkle_commit
        };
        let first_row: &mut DeferralAggPvsCols<F> = trace[..width].borrow_mut();
        let (tagged_left_merkle, merkle_commit) =
            def_internal_compress(first_row.merkle_commit, right_child);
        first_row.tagged_left_merkle = tagged_left_merkle;
        pvs.merkle_commit = merkle_commit;
        pvs.merkle_depth = merkle_depth + F::ONE;
    }
    pvs.num_def_circuit_proofs = num_def_circuit_proofs;
    pvs.def_idx = def_idx;

    let merkle_depth = pvs.merkle_depth.as_canonical_u32() as usize;
    let max_depth_minus_merkle_depth = MAX_DEF_AGG_MERKLE_DEPTH
        .checked_sub(merkle_depth)
        .expect("deferral aggregation merkle depth exceeds max depth");

    DeferralAggPvsTraceCtx {
        proving_ctx: AirProvingContext {
            cached_mains: vec![],
            common_main: RowMajorMatrix::new(trace, width),
            public_values,
        },
        range_check_inputs: vec![max_depth_minus_merkle_depth],
    }
}
