// Keygen V2 API for STARK backend
// Changes:
// - All AIRs can be optional
use std::sync::Arc;

use derivative::Derivative;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::symbolic::SymbolicConstraints,
    config::{Com, PcsProverData},
};

/// Widths of different parts of trace matrix
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    pub preprocessed: Option<usize>,
    pub cached_mains: Vec<usize>,
    pub common_main: usize,
    /// Width counted by extension field elements, _not_ base field elements
    pub after_challenge: Vec<usize>,
}

impl TraceWidth {
    /// Returns the widths of all main traces, including the common main trace if it exists.
    pub fn main_widths(&self) -> Vec<usize> {
        let mut ret = self.cached_mains.clone();
        if self.common_main != 0 {
            ret.push(self.common_main);
        }
        ret
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkVerifyingParams {
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub num_challenges_to_sample: Vec<usize>,
}

/// Verifying key for a single STARK (corresponding to single AIR matrix)
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<SC>>,
    /// Parameters of the STARK
    pub params: StarkVerifyingParams,
    /// Symbolic constraints of the AIR in all challenge phases. This is
    /// a serialization of the constraints in the AIR.
    pub symbolic_constraints: SymbolicConstraints<Val<SC>>,
    /// The factor to multiple the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct MultiStarkVerifyingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkVerifyingKey<SC>>,
}

/// Proving key for a single STARK (corresponding to single AIR matrix)
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    /// Verifying key
    pub vk: StarkVerifyingKey<SC>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<ProverOnlySinglePreprocessedData<SC>>,
    /// Number of interactions to bundle in permutation trace
    pub interaction_chunk_size: usize,
}

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct MultiStarkProvingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkProvingKey<SC>>,
    /// Maximum degree of constraints (excluding logup constraints) across all AIRs
    pub max_constraint_degree: usize,
}

impl<SC: StarkGenericConfig> StarkVerifyingKey<SC> {
    pub fn num_cached_mains(&self) -> usize {
        self.params.width.cached_mains.len()
    }

    pub fn has_common_main(&self) -> bool {
        self.params.width.common_main != 0
    }

    pub fn has_interaction(&self) -> bool {
        !self.symbolic_constraints.interactions.is_empty()
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    pub fn get_vk(&self) -> MultiStarkVerifyingKey<SC> {
        MultiStarkVerifyingKey {
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
        }
    }
}
impl<SC: StarkGenericConfig> MultiStarkVerifyingKey<SC> {
    pub fn num_challenges_to_sample(&self) -> Vec<usize> {
        self.full_view().num_challenges_to_sample()
    }
}

/// Prover only data for preprocessed trace for a single AIR.
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct ProverOnlySinglePreprocessedData<SC: StarkGenericConfig> {
    /// Preprocessed trace matrix.
    pub trace: RowMajorMatrix<Val<SC>>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: Arc<PcsProverData<SC>>,
}

/// Verifier data for preprocessed trace for a single AIR.
///
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct VerifierSinglePreprocessedData<SC: StarkGenericConfig> {
    /// Commitment to the preprocessed trace.
    pub commit: Com<SC>,
}
