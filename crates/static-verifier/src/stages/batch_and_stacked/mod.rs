use halo2_base::{
    Context,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as NativeConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof},
};

use crate::{
    circuit::Fr,
    stages::{
        batch_constraints::{
            AssignedBatchIntermediates, BatchConstraintError, BatchIntermediates,
            constrain_batch_intermediates_unchecked, derive_batch_intermediates,
        },
        stacked_reduction::{
            AssignedStackedReductionIntermediates, StackedReductionConstraintError,
            StackedReductionIntermediates, constrain_stacked_reduction_intermediates,
            derive_stacked_reduction_intermediates,
        },
    },
};

#[derive(Debug, PartialEq, Eq)]
pub enum BatchAndStackedError {
    Batch(BatchConstraintError),
    StackedReduction(StackedReductionConstraintError),
}

impl From<BatchConstraintError> for BatchAndStackedError {
    fn from(value: BatchConstraintError) -> Self {
        Self::Batch(value)
    }
}

impl From<StackedReductionConstraintError> for BatchAndStackedError {
    fn from(value: StackedReductionConstraintError) -> Self {
        Self::StackedReduction(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchAndStackedIntermediates {
    pub batch: BatchIntermediates,
    pub stacked_reduction: StackedReductionIntermediates,
}

#[derive(Clone, Debug)]
pub struct AssignedBatchAndStackedIntermediates {
    pub batch: AssignedBatchIntermediates,
    pub stacked_reduction: AssignedStackedReductionIntermediates,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawBatchAndStackedWitnessState {
    pub intermediates: BatchAndStackedIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedBatchAndStackedState {
    pub batch_consistency_residual: crate::gadgets::baby_bear::BabyBearExtVar,
    pub stacked_final_residual: crate::gadgets::baby_bear::BabyBearExtVar,
}

#[derive(Clone, Debug)]
pub struct CheckedBatchAndStackedWitnessState {
    pub assigned: AssignedBatchAndStackedIntermediates,
    pub derived: DerivedBatchAndStackedState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BatchAndStackedStrictOwnership {
    pub batch_trace_id_to_air_id: Vec<usize>,
    pub batch_total_interactions: u64,
    pub batch_n_logup: usize,
    pub batch_n_max: usize,
    pub batch_degree: usize,
    pub batch_l_skip: usize,
    pub stacked_l_skip: usize,
}

pub fn derive_batch_and_stacked_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<BatchAndStackedIntermediates, BatchAndStackedError> {
    let batch = derive_batch_intermediates(config, mvk, proof)?;
    let stacked_reduction = derive_stacked_reduction_intermediates(config, mvk, proof)?;
    Ok(BatchAndStackedIntermediates {
        batch,
        stacked_reduction,
    })
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_batch_and_stacked_intermediates_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &BatchAndStackedIntermediates,
) -> AssignedBatchAndStackedIntermediates {
    let batch = constrain_batch_intermediates_unchecked(ctx, range, &actual.batch);
    let stacked_reduction =
        constrain_stacked_reduction_intermediates(ctx, range, &actual.stacked_reduction);

    AssignedBatchAndStackedIntermediates {
        batch,
        stacked_reduction,
    }
}

pub(crate) fn derive_batch_and_stacked_strict_ownership(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<BatchAndStackedStrictOwnership, BatchAndStackedError> {
    let expected = derive_batch_and_stacked_intermediates(config, mvk, proof)?;
    Ok(BatchAndStackedStrictOwnership {
        batch_trace_id_to_air_id: expected.batch.trace_id_to_air_id,
        batch_total_interactions: expected.batch.total_interactions,
        batch_n_logup: expected.batch.n_logup,
        batch_n_max: expected.batch.n_max,
        batch_degree: expected.batch.batch_degree,
        batch_l_skip: expected.batch.l_skip,
        stacked_l_skip: expected.stacked_reduction.l_skip,
    })
}

fn constrain_batch_and_stacked_strict_metadata(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
    assigned: &AssignedBatchAndStackedIntermediates,
    ownership: &BatchAndStackedStrictOwnership,
) {
    let gate = range.gate();
    assert_eq!(
        assigned.batch.trace_id_to_air_id.len(),
        ownership.batch_trace_id_to_air_id.len(),
        "strict ownership trace-id schedule length mismatch",
    );
    for (&assigned_air_id, &expected_air_id) in assigned
        .batch
        .trace_id_to_air_id
        .iter()
        .zip(ownership.batch_trace_id_to_air_id.iter())
    {
        gate.assert_is_const(ctx, &assigned_air_id, &Fr::from(expected_air_id as u64));
    }
    gate.assert_is_const(
        ctx,
        &assigned.batch.total_interactions,
        &Fr::from(ownership.batch_total_interactions),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.n_logup,
        &Fr::from(ownership.batch_n_logup as u64),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.n_max,
        &Fr::from(ownership.batch_n_max as u64),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.batch_degree,
        &Fr::from(ownership.batch_degree as u64),
    );

    let batch_l_skip = ctx.load_witness(Fr::from(raw.intermediates.batch.l_skip as u64));
    gate.assert_is_const(ctx, &batch_l_skip, &Fr::from(ownership.batch_l_skip as u64));
    let stacked_l_skip =
        ctx.load_witness(Fr::from(raw.intermediates.stacked_reduction.l_skip as u64));
    gate.assert_is_const(
        ctx,
        &stacked_l_skip,
        &Fr::from(ownership.stacked_l_skip as u64),
    );
}

/// Standalone batch-and-stacked derive+constrain wrapper is internal; external callers must use
/// transcript-owned stage composition (`stages::full_pipeline`) as the acceptance boundary.
pub(crate) fn derive_and_constrain_batch_and_stacked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedBatchAndStackedIntermediates, BatchAndStackedError> {
    let raw = derive_raw_batch_and_stacked_witness_state(config, mvk, proof)?;
    let ownership = derive_batch_and_stacked_strict_ownership(config, mvk, proof)?;
    Ok(constrain_checked_batch_and_stacked_witness_state_strict(ctx, range, &raw, &ownership).assigned)
}

pub(crate) fn derive_raw_batch_and_stacked_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawBatchAndStackedWitnessState, BatchAndStackedError> {
    Ok(RawBatchAndStackedWitnessState {
        intermediates: derive_batch_and_stacked_intermediates(config, mvk, proof)?,
    })
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_checked_batch_and_stacked_witness_state_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
) -> CheckedBatchAndStackedWitnessState {
    let assigned = constrain_batch_and_stacked_intermediates_unchecked(ctx, range, &raw.intermediates);
    let derived = DerivedBatchAndStackedState {
        batch_consistency_residual: assigned.batch.consistency_residual.clone(),
        stacked_final_residual: assigned.stacked_reduction.final_residual.clone(),
    };
    CheckedBatchAndStackedWitnessState { assigned, derived }
}

pub(crate) fn constrain_checked_batch_and_stacked_witness_state_strict(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
    ownership: &BatchAndStackedStrictOwnership,
) -> CheckedBatchAndStackedWitnessState {
    let checked = constrain_checked_batch_and_stacked_witness_state_unchecked(ctx, range, raw);
    constrain_batch_and_stacked_strict_metadata(ctx, range, raw, &checked.assigned, ownership);
    checked
}

#[cfg(test)]
mod tests;
