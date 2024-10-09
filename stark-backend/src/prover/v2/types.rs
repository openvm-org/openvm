use derivative::Derivative;
use itertools::Itertools;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    config::Com,
    keygen::v2::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    prover::{opener::OpeningProof, trace::ProverTraceData, types::Commitments},
    rap::AnyRap,
};

/// The full proof for multiple RAPs where trace matrices are committed into
/// multiple commitments, where each commitment is multi-matrix.
///
/// Includes the quotient commitments and FRI opening proofs for the constraints as well.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofV2<SC: StarkGenericConfig> {
    /// The PCS commitments
    pub commitments: Commitments<SC>,
    /// Opening proofs separated by partition, but this may change
    pub opening: OpeningProof<SC>,
    /// Proof data for each AIR
    pub per_air: Vec<AirProofData<SC>>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
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
pub struct ProofInput<'a, SC: StarkGenericConfig> {
    /// (AIR id, AIR input)
    pub per_air: Vec<(usize, AirProofInput<'a, SC>)>,
}

#[derive(Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct CommittedTraceData<SC: StarkGenericConfig> {
    pub raw_data: RowMajorMatrix<Val<SC>>,
    pub prover_data: ProverTraceData<SC>,
}

/// Necessary input for proving a single AIR.
#[derive(Derivative)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
pub struct AirProofInput<'a, SC: StarkGenericConfig> {
    pub air: &'a dyn AnyRap<SC>,
    /// Cached main trace matrices
    pub cached_mains: Vec<CommittedTraceData<SC>>,
    /// Common main trace matrix
    pub common_main: Option<RowMajorMatrix<Val<SC>>>,
    /// Public values
    pub public_values: Vec<Val<SC>>,
}

pub trait Chip<SC: StarkGenericConfig> {
    fn air(&self) -> &dyn AnyRap<SC>;
    /// Generate all necessary input for proving a single AIR.
    fn generate_air_proof_input(&self) -> AirProofInput<SC>;
    fn generate_air_proof_input_with_id(&self, air_id: usize) -> (usize, AirProofInput<SC>) {
        (air_id, self.generate_air_proof_input())
    }
}

impl<SC: StarkGenericConfig> ProofV2<SC> {
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

impl<'a, SC: StarkGenericConfig> ProofInput<'a, SC> {
    pub fn sort(&mut self) {
        self.per_air.sort_by_key(|p| p.0);
    }
}

impl<SC: StarkGenericConfig> MultiStarkVerifyingKeyV2<SC> {
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

impl<SC: StarkGenericConfig> MultiStarkProvingKeyV2<SC> {
    pub fn validate(&self, proof_input: &ProofInput<SC>) -> bool {
        self.get_vk().validate(proof_input)
    }
}
