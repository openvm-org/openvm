pub(crate) mod witness;

use std::sync::Arc;

use halo2_base::{gates::range::RangeChip, utils::biguint_to_fe, AssignedValue, Context};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config as RootConfig, Bn254Scalar},
    openvm_stark_backend::{
        keygen::types::MultiStarkVerifyingKey, p3_field::PrimeField, proof::Proof,
    },
};

use crate::{
    field::baby_bear::{BabyBearChip, BabyBearExtChip, BabyBearWire, BABY_BEAR_BITS},
    stages::{
        batch_constraints::{
            compute_trace_id_to_air_id, constrain_batch_from_proof_inputs,
            AssignedBatchIntermediates, BatchConstraintError,
        },
        full_pipeline::witness::get_need_rot_per_commit,
        proof_shape::{
            derive_and_constrain_proof_shape, derive_proof_shape_rules,
            AssignedProofShapeIntermediates, ProofShapePreambleError,
        },
        stacked_reduction::{
            constrain_stacked_reduction_from_proof_inputs, AssignedStackedReductionIntermediates,
            StackedReductionConstraintError,
        },
        whir::{constrain_whir_from_proof_inputs, AssignedWhirIntermediates, WhirError},
    },
    transcript::{digest_wire_from_root, TranscriptGadget},
    Fr,
};

#[cfg(test)]
mod tests;

#[derive(Debug, PartialEq, Eq)]
pub enum PipelineError {
    ProofShape(ProofShapePreambleError),
    Batch(BatchConstraintError),
    StackedReduction(StackedReductionConstraintError),
    Whir(WhirError),
}

impl From<ProofShapePreambleError> for PipelineError {
    fn from(value: ProofShapePreambleError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<BatchConstraintError> for PipelineError {
    fn from(value: BatchConstraintError) -> Self {
        Self::Batch(value)
    }
}

impl From<StackedReductionConstraintError> for PipelineError {
    fn from(value: StackedReductionConstraintError) -> Self {
        Self::StackedReduction(value)
    }
}

impl From<WhirError> for PipelineError {
    fn from(value: WhirError) -> Self {
        Self::Whir(value)
    }
}

#[derive(Clone, Debug)]
pub struct AssignedPipelineIntermediates {
    pub proof_shape: AssignedProofShapeIntermediates,
    pub batch: AssignedBatchIntermediates,
    pub stacked_reduction: AssignedStackedReductionIntermediates,
    pub whir: AssignedWhirIntermediates,
    pub statement_public_inputs: [AssignedValue<Fr>; 2],
}

#[derive(Clone, Debug)]
struct AssignedPreambleState {
    statement_public_inputs: [AssignedValue<Fr>; 2],
    public_values: Vec<Vec<BabyBearWire>>,
}

fn digest_scalar_to_fr(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

fn observe_preamble_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    transcript: &mut TranscriptGadget,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    proof: &Proof<RootConfig>,
    proof_shape: &AssignedProofShapeIntermediates,
) -> AssignedPreambleState {
    let base_chip = BabyBearChip::new(Arc::new(range.clone()));
    let statement_public_inputs = [
        ctx.load_witness(digest_scalar_to_fr(mvk.pre_hash[0])),
        ctx.load_witness(digest_scalar_to_fr(proof.common_main_commit[0])),
    ];

    transcript.observe_commit(
        ctx,
        range,
        &base_chip,
        &digest_wire_from_root(statement_public_inputs[0]),
    );
    transcript.observe_commit(
        ctx,
        range,
        &base_chip,
        &digest_wire_from_root(statement_public_inputs[1]),
    );

    let public_values = proof
        .public_values
        .iter()
        .enumerate()
        .map(|(air_idx, values)| {
            if !mvk.inner.per_air[air_idx].is_required {
                transcript.observe(
                    ctx,
                    range,
                    &base_chip,
                    &BabyBearWire {
                        value: proof_shape.air_presence_flags[air_idx],
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
                        range,
                        &base_chip,
                        &digest_wire_from_root(preprocessed_root),
                    );
                } else {
                    transcript.observe(
                        ctx,
                        range,
                        &base_chip,
                        &BabyBearWire {
                            value: proof_shape.air_log_heights[air_idx],
                            max_bits: BABY_BEAR_BITS,
                        },
                    );
                }

                for commit in &proof.trace_vdata[air_idx]
                    .as_ref()
                    .expect("present air must include trace vdata")
                    .cached_commitments
                {
                    let root = ctx.load_witness(digest_scalar_to_fr(commit[0]));
                    transcript.observe_commit(ctx, range, &base_chip, &digest_wire_from_root(root));
                }
            }

            values
                .iter()
                .map(|&value| {
                    let value = base_chip.load_witness(ctx, value);
                    transcript.observe(ctx, range, &base_chip, &value);
                    value
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    AssignedPreambleState {
        statement_public_inputs,
        public_values,
    }
}

pub fn constrained_verify(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &RootConfig,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    proof: &Proof<RootConfig>,
) -> Result<AssignedPipelineIntermediates, PipelineError> {
    let l_skip = mvk.inner.params.l_skip;
    let base_chip = Arc::new(BabyBearChip::new(Arc::new(range.clone())));
    let ext_chip = BabyBearExtChip::new(base_chip);
    let proof_shape = derive_and_constrain_proof_shape(ctx, ext_chip.base(), config, mvk, proof)?;
    let trace_id_to_air_id = compute_trace_id_to_air_id(&mvk.inner, proof);

    let mut transcript = TranscriptGadget::new(ctx);
    let preamble = observe_preamble_assigned(ctx, range, &mut transcript, mvk, proof, &proof_shape);

    let batch = constrain_batch_from_proof_inputs(
        ctx,
        range,
        &mut transcript,
        &mvk.inner,
        proof,
        &trace_id_to_air_id,
        &proof_shape.trace_id_to_air_id,
        preamble.public_values,
    )?;

    let layouts = derive_proof_shape_rules(&mvk.inner, proof)
        .map_err(|err| PipelineError::ProofShape(ProofShapePreambleError::ProofShape(err)))?
        .layouts;
    let need_rot_per_commit = get_need_rot_per_commit(&mvk.inner, proof, &trace_id_to_air_id)?;
    let stacked_reduction = constrain_stacked_reduction_from_proof_inputs(
        ctx,
        &ext_chip,
        &mut transcript,
        &proof.stacking_proof,
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
        let common_main_root = preamble.statement_public_inputs[1];
        let mut commits = vec![common_main_root];
        for &air_id in &trace_id_to_air_id {
            if let Some(preprocessed) = &mvk.inner.per_air[air_id].preprocessed_data {
                commits.push(ctx.load_constant(digest_scalar_to_fr(preprocessed.commit[0])));
            }
            commits.extend(
                proof.trace_vdata[air_id]
                    .as_ref()
                    .expect("trace-id schedule must reference present airs")
                    .cached_commitments
                    .iter()
                    .map(|commit| ctx.load_witness(digest_scalar_to_fr(commit[0]))),
            );
        }
        commits
    };

    let whir = constrain_whir_from_proof_inputs(
        ctx,
        &ext_chip,
        &mut transcript,
        config,
        &proof.whir_proof,
        &stacked_reduction.stacking_openings,
        &initial_commitment_roots,
        &u_cube,
    );

    Ok(AssignedPipelineIntermediates {
        proof_shape,
        batch,
        stacked_reduction,
        whir,
        statement_public_inputs: preamble.statement_public_inputs,
    })
}
