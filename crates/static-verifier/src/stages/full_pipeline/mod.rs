use std::sync::Arc;

use halo2_base::{gates::range::RangeChip, utils::biguint_to_fe, AssignedValue, Context};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config as RootConfig, Bn254Scalar},
    openvm_stark_backend::{
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        p3_field::PrimeField,
        proof::Proof,
    },
};

use crate::{
    field::baby_bear::{BabyBearChip, BabyBearExtChip, BabyBearWire, BABY_BEAR_BITS},
    stages::{
        batch_constraints::{
            constrain_batch_constraints_verification, load_batch_constraint_proof_wire,
            load_gkr_proof_wire, BatchConstraintProofWire, GkrProofWire,
        },
        proof_shape::compute_trace_id_to_air_id,
        stacked_reduction::{
            constrain_stacked_reduction, load_stacking_proof_wire, stacked_reduction_layouts,
            StackingProofWire,
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

pub fn load_proof_wire(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    proof: &Proof<RootConfig>,
) -> ProofWire {
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
    proof: &Proof<RootConfig>,
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
            let presence_flag =
                ctx.load_constant(Fr::from(proof.trace_vdata[air_idx].is_some() as u64));
            transcript.observe(
                ctx,
                &base_chip,
                &BabyBearWire {
                    value: presence_flag,
                    max_bits: 1,
                },
            );
        }

        if proof.trace_vdata[air_idx].is_some() {
            if let Some(preprocessed) = mvk.inner.per_air[air_idx].preprocessed_data.as_ref() {
                let preprocessed_root =
                    ctx.load_constant(digest_scalar_to_fr(preprocessed.commit[0]));
                transcript.observe_commit(
                    ctx,
                    &base_chip,
                    &digest_wire_from_root(preprocessed_root),
                );
            } else {
                let log_height = ctx.load_constant(Fr::from(
                    proof.trace_vdata[air_idx]
                        .as_ref()
                        .expect("present air must include trace vdata")
                        .log_height as u64,
                ));
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
        }

        for value in &public_values[air_idx] {
            transcript.observe(ctx, &base_chip, value);
        }
    }
}

/// Run the full static verifier pipeline on pre-loaded witness data.
///
/// Returns the two statement public inputs as assigned cells:
/// `[mvk_pre_hash_root, common_main_commit_root]`.
pub fn constrained_verify(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    proof: &Proof<RootConfig>,
    proof_wire: ProofWire,
) -> [AssignedValue<Fr>; 2] {
    let l_skip = mvk.inner.params.l_skip;
    let trace_id_to_air_id = compute_trace_id_to_air_id(&mvk.inner, proof);

    let mvk_pre_hash_root = ctx.load_constant(digest_scalar_to_fr(mvk.pre_hash[0]));
    let statement_public_inputs = [mvk_pre_hash_root, proof_wire.common_main_commit_root];

    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            proof.trace_vdata[air_id]
                .as_ref()
                .expect("present air must have trace vdata")
                .log_height as isize
                - l_skip as isize
        })
        .collect();

    let mut transcript = TranscriptGadget::new(ctx);
    observe_preamble(
        ctx,
        range,
        &mut transcript,
        mvk,
        proof,
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
        &trace_id_to_air_id,
        proof_wire.public_values,
    );

    let layouts = stacked_reduction_layouts(&mvk.inner, proof);
    let need_rot_per_commit = get_need_rot_per_commit(&mvk.inner, proof, &trace_id_to_air_id);
    let stacked_reduction = constrain_stacked_reduction(
        ctx,
        &ext_chip,
        &mut transcript,
        &proof_wire.stacking,
        &layouts,
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
        for &air_id in &trace_id_to_air_id {
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
    proof: &Proof<RootConfig>,
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
        let cached_len = proof.trace_vdata[air_id]
            .as_ref()
            .expect("present air must have trace vdata")
            .cached_commitments
            .len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }
    need_rot_per_commit
}
