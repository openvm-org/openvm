use std::sync::Arc;

use derivative::Derivative;
use itertools::Itertools;
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

pub use super::trace::{ProverTraceData, TraceCommitter};
use crate::{
    config::Com,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    prover::opener::OpeningProof,
    rap::AnyRap,
};

/// All commitments to a multi-matrix STARK that are not preprocessed.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct Commitments<SC: StarkGenericConfig> {
    /// Multiple commitments for the main trace.
    /// For each RAP, each part of a partitioned matrix trace matrix
    /// must belong to one of these commitments.
    pub main_trace: Vec<Com<SC>>,
    /// One shared commitment for all trace matrices across all RAPs
    /// in a single challenge phase `i` after observing the commits to
    /// `preprocessed`, `main_trace`, and `after_challenge[..i]`
    pub after_challenge: Vec<Com<SC>>,
    /// Shared commitment for all quotient polynomial evaluations
    pub quotient: Com<SC>,
}

/// The full proof for multiple RAPs where trace matrices are committed into
/// multiple commitments, where each commitment is multi-matrix.
///
/// Includes the quotient commitments and FRI opening proofs for the constraints as well.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct Proof<SC: StarkGenericConfig> {
    /// The PCS commitments
    pub commitments: Commitments<SC>,
    /// Opening proofs separated by partition, but this may change
    pub opening: OpeningProof<SC>,
    /// Proof data for each AIR
    pub per_air: Vec<AirProofData<SC>>,
}

#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "SC::Challenge: Clone"))]
pub struct AirProofData<SC: StarkGenericConfig> {
    pub air_id: usize,
    /// height of trace matrix.
    pub degree: usize,
    /// For each challenge phase with trace, the values to expose to the verifier in that phase
    pub exposed_values_after_challenge: Vec<Vec<SC::Challenge>>,
    // The public values to expose to the verifier
    pub public_values: Vec<Val<SC>>,
}

/// Proof input
pub struct ProofInput<SC: StarkGenericConfig> {
    /// (AIR id, AIR input)
    pub per_air: Vec<(usize, AirProofInput<SC>)>,
}

impl<SC: StarkGenericConfig> ProofInput<SC> {
    pub fn new(per_air: Vec<(usize, AirProofInput<SC>)>) -> Self {
        Self { per_air }
    }
    pub fn into_air_proof_input_vec(self) -> Vec<AirProofInput<SC>> {
        self.per_air.into_iter().map(|(_, x)| x).collect()
    }
}

#[derive(Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct CommittedTraceData<SC: StarkGenericConfig> {
    pub raw_data: Arc<RowMajorMatrix<Val<SC>>>,
    pub prover_data: ProverTraceData<SC>,
}

/// Necessary input for proving a single AIR.
#[derive(Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct AirProofInput<SC: StarkGenericConfig> {
    pub air: Arc<dyn AnyRap<SC>>,
    /// Prover data for cached main traces
    pub cached_mains_pdata: Vec<ProverTraceData<SC>>,
    pub raw: AirProofRawInput<Val<SC>>,
}

/// Raw input for proving a single AIR.
#[derive(Clone, Debug)]
pub struct AirProofRawInput<F: Field> {
    /// Cached main trace matrices
    pub cached_mains: Vec<Arc<RowMajorMatrix<F>>>,
    /// Common main trace matrix
    pub common_main: Option<RowMajorMatrix<F>>,
    /// Public values
    pub public_values: Vec<F>,
}

impl<SC: StarkGenericConfig> Proof<SC> {
    pub fn get_air_ids(&self) -> Vec<usize> {
        self.per_air.iter().map(|p| p.air_id).collect()
    }
    pub fn get_public_values(&self) -> Vec<Vec<Val<SC>>> {
        self.per_air
            .iter()
            .map(|p| p.public_values.clone())
            .collect()
    }
}

impl<SC: StarkGenericConfig> ProofInput<SC> {
    pub fn sort(&mut self) {
        self.per_air.sort_by_key(|p| p.0);
    }
}

impl<SC: StarkGenericConfig> MultiStarkVerifyingKey<SC> {
    pub fn validate(&self, proof_input: &ProofInput<SC>) -> bool {
        if !proof_input
            .per_air
            .iter()
            .all(|input| input.0 < self.per_air.len())
        {
            return false;
        }
        if !proof_input
            .per_air
            .iter()
            .tuple_windows()
            .all(|(a, b)| a.0 < b.0)
        {
            return false;
        }
        true
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    pub fn validate(&self, proof_input: &ProofInput<SC>) -> bool {
        self.get_vk().validate(proof_input)
    }
}

impl<F: Field> AirProofRawInput<F> {
    pub fn height(&self) -> usize {
        let mut height = None;
        for m in self.cached_mains.iter() {
            if let Some(h) = height {
                assert_eq!(h, m.height());
            } else {
                height = Some(m.height());
            }
        }
        let common_h = self.common_main.as_ref().map(|trace| trace.height());
        if let Some(h) = height {
            if let Some(common_h) = common_h {
                assert_eq!(h, common_h);
            }
            h
        } else {
            common_h.unwrap_or(0)
        }
    }
}
