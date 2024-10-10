/// Keygen V2 API for STARK backend
/// Changes:
/// - All AIRs can be optional
use derivative::Derivative;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::symbolic::SymbolicConstraints,
    config::{Com, PcsProverData},
    keygen::types::{
        ProverOnlySinglePreprocessedData, StarkVerifyingParams, VerifierSinglePreprocessedData,
    },
};

/// Verifying key for a single STARK (corresponding to single AIR matrix)
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct StarkVerifyingKeyV2<SC: StarkGenericConfig> {
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
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct MultiStarkVerifyingKeyV2<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkVerifyingKeyV2<SC>>,
}

/// Proving key for a single STARK (corresponding to single AIR matrix)
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct StarkProvingKeyV2<SC: StarkGenericConfig> {
    /// Type name of the AIR, for display purposes only
    pub air_name: String,
    /// Verifying key
    pub vk: StarkVerifyingKeyV2<SC>,
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
pub struct MultiStarkProvingKeyV2<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkProvingKeyV2<SC>>,
    /// Maximum degree of constraints (excluding logup constraints) across all AIRs
    pub max_constraint_degree: usize,
}

impl<SC: StarkGenericConfig> StarkVerifyingKeyV2<SC> {
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

impl<SC: StarkGenericConfig> MultiStarkProvingKeyV2<SC> {
    pub fn get_vk(&self) -> MultiStarkVerifyingKeyV2<SC> {
        MultiStarkVerifyingKeyV2 {
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
        }
    }
}
impl<SC: StarkGenericConfig> MultiStarkVerifyingKeyV2<SC> {
    pub fn num_challenges_to_sample(&self) -> Vec<usize> {
        self.full_view().num_challenges_to_sample()
    }
}
