use std::borrow::{Borrow, BorrowMut};

use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::deferral::{
    inner::input::air::InputCommitCols, DeferralAggregationPvs, DeferralCircuitPvs,
    DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
};

type CachedCommitRow<F> = (usize, usize, [F; DIGEST_SIZE]);

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_agg: bool,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_proofs = proofs.len();
    debug_assert!((1..=2).contains(&num_proofs));

    let rows_per_proof = proofs
        .iter()
        .map(|proof| {
            if child_is_agg {
                1
            } else {
                count_cached_commitments(proof) + 1
            }
        })
        .collect::<Vec<_>>();
    let num_rows = rows_per_proof.iter().sum::<usize>();
    let height = num_rows.next_power_of_two();
    let width = InputCommitCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_idx = 0usize;

    for (proof_idx, proof) in proofs.iter().enumerate() {
        let initial_commit = if child_is_agg {
            let child_pvs: &DeferralAggregationPvs<F> =
                proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();
            child_pvs.merkle_commit
        } else {
            let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
                .as_slice()
                .borrow();
            child_pvs.input_commit
        };
        let cached_rows = if child_is_agg {
            Vec::new()
        } else {
            collect_cached_rows(proof)
        };

        let mut capacity = [F::ZERO; DIGEST_SIZE];

        for (row_in_proof, (air_idx, cached_idx, current_commit)) in
            std::iter::once((0usize, 0usize, initial_commit))
                .chain(cached_rows.into_iter())
                .enumerate()
        {
            let cols: &mut InputCommitCols<F> =
                trace[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row_in_proof == 0);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.has_verifier_pvs = F::from_bool(child_is_agg);
            cols.air_idx = F::from_usize(air_idx);
            cols.cached_idx = F::from_usize(cached_idx);
            cols.current_commit = current_commit;
            cols.capacity = capacity;

            if child_is_agg {
                cols.res_left = [F::ZERO; DIGEST_SIZE];
                cols.res_right = [F::ZERO; DIGEST_SIZE];
            } else {
                let (res_left, res_right) =
                    poseidon2_compress_with_capacity(cols.current_commit, cols.capacity);
                cols.res_left = res_left;
                cols.res_right = res_right;
                capacity = res_right;
            }
            row_idx += 1;
        }
    }

    AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width))
}

fn count_cached_commitments(proof: &Proof<BabyBearPoseidon2Config>) -> usize {
    proof
        .trace_vdata
        .iter()
        .flatten()
        .map(|vd| vd.cached_commitments.len())
        .sum()
}

fn collect_cached_rows(proof: &Proof<BabyBearPoseidon2Config>) -> Vec<CachedCommitRow<F>> {
    proof
        .trace_vdata
        .iter()
        .enumerate()
        .flat_map(|(air_idx, vdata)| {
            vdata.iter().flat_map(move |vd| {
                vd.cached_commitments
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(cached_idx, cached_commit)| (air_idx, cached_idx, cached_commit))
            })
        })
        .collect()
}
