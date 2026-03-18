use halo2_base::{
    gates::range::RangeChip,
    Context,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as NativeConfig,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof},
};

use crate::{
    circuit::Fr,
    stages::{
        batch_constraints::{
            constrain_batch_intermediates_unchecked, derive_batch_intermediates,
            AssignedBatchIntermediates, BatchConstraintError, BatchIntermediates,
        },
        stacked_reduction::{
            constrain_stacked_reduction_intermediates, derive_stacked_reduction_intermediates,
            AssignedStackedReductionIntermediates, StackedReductionConstraintError,
            StackedReductionIntermediates,
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

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_checked_batch_and_stacked_witness_state_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
) -> CheckedBatchAndStackedWitnessState {
    let assigned =
        constrain_batch_and_stacked_intermediates_unchecked(ctx, range, &raw.intermediates);
    let derived = DerivedBatchAndStackedState {
        batch_consistency_residual: assigned.batch.consistency_residual.clone(),
        stacked_final_residual: assigned.stacked_reduction.final_residual.clone(),
    };
    CheckedBatchAndStackedWitnessState { assigned, derived }
}

#[cfg(test)]
mod tests;
