use std::sync::Arc;

use halo2_base::{gates::range::RangeChip, utils::biguint_to_fe, AssignedValue, Context};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config as RootConfig, Bn254Scalar},
    openvm_stark_backend::{
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        p3_field::PrimeField,
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};

use crate::{
    field::baby_bear::{BabyBearChip, BabyBearExtChip, BabyBearWire, BABY_BEAR_BITS},
    stages::{
        batch_constraints::{
            constrain_batch_constraints_verification, load_batch_constraint_proof_wire,
            load_gkr_proof_wire, BatchConstraintProofWire, GkrProofWire,
        },
        stacked_reduction::{
            constrain_stacked_reduction, load_stacking_proof_wire, StackingProofWire,
        },
        whir::{constrain_whir_verification, load_whir_proof_wire, WhirProofWire},
    },
    transcript::{digest_wire_from_root, TranscriptGadget},
    Fr,
};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct ProofWire {
    pub common_main_commit_root: AssignedValue<Fr>,
    pub public_values: Vec<Vec<BabyBearWire>>,
    pub cached_commitment_roots: Vec<Vec<AssignedValue<Fr>>>,
    pub gkr: GkrProofWire,
    pub batch: BatchConstraintProofWire,
    pub stacking: StackingProofWire,
    pub whir: WhirProofWire,
}

pub(crate) fn digest_scalar_to_fr(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

/// Load proof data into Halo2 cells. `log_heights_per_air` must match this circuit's fixed heights;
/// host-side asserts enforce that every `proof.trace_vdata[air_id].log_height` agrees.
pub fn load_proof_wire(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    proof: &Proof<RootConfig>,
    log_heights_per_air: &[usize],
) -> ProofWire {
    assert_eq!(
        proof.trace_vdata.len(),
        log_heights_per_air.len(),
        "proof.trace_vdata length must match log_heights_per_air"
    );
    for (air_id, (tv, &expected_log_height)) in proof
        .trace_vdata
        .iter()
        .zip(log_heights_per_air.iter())
        .enumerate()
    {
        let Some(vd) = tv.as_ref() else {
            panic!("static verifier proof must include trace_vdata for air_id {air_id}");
        };
        assert_eq!(
            vd.log_height, expected_log_height,
            "trace log_height mismatch for air_id {air_id}: proof has {}, circuit expects {}",
            vd.log_height, expected_log_height
        );
    }

    let base_chip = Arc::new(BabyBearChip::new(Arc::new(range.clone())));
    let ext_chip = BabyBearExtChip::new(base_chip.clone());

    let common_main_commit_root =
        ctx.load_witness(digest_scalar_to_fr(proof.common_main_commit[0]));

    let public_values = proof
        .public_values
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| base_chip.load_witness(ctx, value))
                .collect()
        })
        .collect();

    let cached_commitment_roots = proof
        .trace_vdata
        .iter()
        .map(|vdata| {
            if let Some(vdata) = vdata {
                vdata
                    .cached_commitments
                    .iter()
                    .map(|commit| ctx.load_witness(digest_scalar_to_fr(commit[0])))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let gkr = load_gkr_proof_wire(ctx, &base_chip, &ext_chip, &proof.gkr_proof);
    let batch = load_batch_constraint_proof_wire(ctx, &ext_chip, &proof.batch_constraint_proof);
    let stacking = load_stacking_proof_wire(ctx, &ext_chip, &proof.stacking_proof);
    let whir = load_whir_proof_wire(ctx, &base_chip, &ext_chip, &proof.whir_proof);

    ProofWire {
        common_main_commit_root,
        public_values,
        cached_commitment_roots,
        gkr,
        batch,
        stacking,
        whir,
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_preamble(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    transcript: &mut TranscriptGadget,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    log_heights_per_air: &[usize],
    public_values: &[Vec<BabyBearWire>],
    cached_commitment_roots: &[Vec<AssignedValue<Fr>>],
    statement_public_inputs: [AssignedValue<Fr>; 2],
) {
    let base_chip = BabyBearChip::new(Arc::new(range.clone()));

    transcript.observe_commit(
        ctx,
        &base_chip,
        &digest_wire_from_root(statement_public_inputs[0]),
    );
    transcript.observe_commit(
        ctx,
        &base_chip,
        &digest_wire_from_root(statement_public_inputs[1]),
    );

    for air_idx in 0..mvk.inner.per_air.len() {
        if !mvk.inner.per_air[air_idx].is_required {
            // Static verifier: every AIR in the child VK has a trace (see crate `lib.rs`).
            let presence_flag = ctx.load_constant(Fr::one());
            transcript.observe(
                ctx,
                &base_chip,
                &BabyBearWire {
                    value: presence_flag,
                    max_bits: 1,
                },
            );
        }

        if let Some(preprocessed) = mvk.inner.per_air[air_idx].preprocessed_data.as_ref() {
            let preprocessed_root = ctx.load_constant(digest_scalar_to_fr(preprocessed.commit[0]));
            transcript.observe_commit(ctx, &base_chip, &digest_wire_from_root(preprocessed_root));
        } else {
            // Fixed circuit parameter (not loaded from the proof witness).
            let log_height = ctx.load_constant(Fr::from(log_heights_per_air[air_idx] as u64));
            transcript.observe(
                ctx,
                &base_chip,
                &BabyBearWire {
                    value: log_height,
                    max_bits: BABY_BEAR_BITS,
                },
            );
        }

        for root in &cached_commitment_roots[air_idx] {
            transcript.observe_commit(ctx, &base_chip, &digest_wire_from_root(*root));
        }

        for value in &public_values[air_idx] {
            transcript.observe(ctx, &base_chip, value);
        }
    }
}

/// Run the full static verifier pipeline on pre-loaded witness data.
///
/// `trace_id_to_air_id` and `log_heights_per_air` are fixed for this circuit (host-side). They must
/// match the child proof shape: `log_heights_per_air.len() == mvk.inner.per_air.len()`, and
/// `trace_id_to_air_id` must list every `air_id` exactly once in descending-`log_height` order
/// (tie-break: ascending `air_id`).
///
/// `stacked_layouts` must be the layout vector fixed for this circuit (same as stored on
/// [`crate::StaticVerifierCircuit`]).
///
/// Returns the two statement public inputs as assigned cells:
/// `[mvk_pre_hash_root, common_main_commit_root]`.
pub fn constrained_verify(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    proof_wire: ProofWire,
    trace_id_to_air_id: &[usize],
    log_heights_per_air: &[usize],
    stacked_layouts: &[StackedLayout],
) -> [AssignedValue<Fr>; 2] {
    assert_eq!(
        log_heights_per_air.len(),
        mvk.inner.per_air.len(),
        "log_heights_per_air must match VK per_air count"
    );
    let l_skip = mvk.inner.params.l_skip;

    let mvk_pre_hash_root = ctx.load_constant(digest_scalar_to_fr(mvk.pre_hash[0]));
    let statement_public_inputs = [mvk_pre_hash_root, proof_wire.common_main_commit_root];

    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| log_heights_per_air[air_id] as isize - l_skip as isize)
        .collect();

    let mut transcript = TranscriptGadget::new(ctx);
    observe_preamble(
        ctx,
        range,
        &mut transcript,
        mvk,
        log_heights_per_air,
        &proof_wire.public_values,
        &proof_wire.cached_commitment_roots,
        statement_public_inputs,
    );

    let base_chip = Arc::new(BabyBearChip::new(Arc::new(range.clone())));
    let ext_chip = BabyBearExtChip::new(base_chip);

    let batch = constrain_batch_constraints_verification(
        ctx,
        range,
        &mut transcript,
        &mvk.inner,
        &proof_wire.gkr,
        &proof_wire.batch,
        &n_per_trace,
        trace_id_to_air_id,
        proof_wire.public_values,
    );

    let need_rot_per_commit = get_need_rot_per_commit(&mvk.inner, trace_id_to_air_id);
    let stacked_reduction = constrain_stacked_reduction(
        ctx,
        &ext_chip,
        &mut transcript,
        &proof_wire.stacking,
        stacked_layouts,
        &need_rot_per_commit,
        l_skip,
        mvk.inner.params.n_stack,
        &batch.column_openings,
        &batch.r,
    );

    let u_cube = {
        let u = &stacked_reduction.u;
        assert!(!u.is_empty());
        let mut u_cube = Vec::with_capacity(l_skip + u.len().saturating_sub(1));
        let mut power = *u.first().unwrap();
        for _ in 0..l_skip {
            u_cube.push(power);
            power = ext_chip.square(ctx, power);
        }
        u_cube.extend(u.iter().skip(1).copied());
        u_cube
    };

    let initial_commitment_roots = {
        let common_main_root = statement_public_inputs[1];
        let mut commits = vec![common_main_root];
        for &air_id in trace_id_to_air_id {
            if let Some(preprocessed) = &mvk.inner.per_air[air_id].preprocessed_data {
                commits.push(ctx.load_constant(digest_scalar_to_fr(preprocessed.commit[0])));
            }
            commits.extend(proof_wire.cached_commitment_roots[air_id].iter().copied());
        }
        commits
    };

    constrain_whir_verification(
        ctx,
        &ext_chip,
        &mut transcript,
        &mvk.inner,
        &proof_wire.whir,
        &stacked_reduction.stacking_openings,
        &initial_commitment_roots,
        &u_cube,
    );

    statement_public_inputs
}

/// Helper function, purely on out-of-circuit values.
fn get_need_rot_per_commit(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    trace_id_to_air_id: &[usize],
) -> Vec<Vec<bool>> {
    let mut need_rot_per_commit = vec![trace_id_to_air_id
        .iter()
        .map(|&air_id| mvk0.per_air[air_id].params.need_rot)
        .collect::<Vec<_>>()];
    for &air_id in trace_id_to_air_id {
        let need_rot = mvk0.per_air[air_id].params.need_rot;
        if mvk0.per_air[air_id].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        let cached_len = mvk0.per_air[air_id].params.width.cached_mains.len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }
    need_rot_per_commit
}
