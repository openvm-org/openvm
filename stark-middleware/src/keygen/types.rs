use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    commit::MatrixCommitmentGraph,
    config::{Com, PcsProverData},
};

/// Proving key for a single STARK (corresponding to single AIR matrix)
///
/// !! This is not the full proving key right now. It is missing AIR constraints
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct StarkProvingKey<SC: StarkGenericConfig> {
    /// Verifying key
    pub vk: StarkVerifyingKey<SC>,
    /// Prover only data for preprocessed trace
    pub preprocessed_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

/// Verifying key for a single STARK (corresponding to single AIR matrix)
///
/// !! This is not the full proving key right now. It is missing AIR constraints
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    /// Height of trace matrix.
    pub degree: usize,
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<SC>>,
    /// [MatrixCommitmentGraph] for partitioned main trace matrix
    pub main_graph: MatrixCommitmentGraph,
    /// The factor to multiple the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
}

/// Prover only data for preprocessed trace for a single AIR.
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct ProverOnlySinglePreprocessedData<SC: StarkGenericConfig> {
    /// Preprocessed trace matrix.
    pub trace: RowMajorMatrix<Val<SC>>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: PcsProverData<SC>,
}

/// Verifier data for preprocessed trace for a single AIR.
///
/// Currently assumes each AIR has it's own preprocessed commitment
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct VerifierSinglePreprocessedData<SC: StarkGenericConfig> {
    /// Commitment to the preprocessed trace.
    pub commit: Com<SC>,
}

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key.
///
/// !! This is not the full proving key right now. It is missing AIR constraints
/// in the ProverRap trait
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "PcsProverData<SC>: Serialize",
    deserialize = "PcsProverData<SC>: Deserialize<'de>"
))]
pub struct ChipsetProvingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkProvingKey<SC>>,
    /// Number of multi-matrix commitments that hold commitments to the partitioned main trace matrices across all AIRs.
    pub num_main_trace_commitments: usize,
}

impl<SC: StarkGenericConfig> ChipsetProvingKey<SC> {
    pub fn preprocessed_commits(&self) -> impl Iterator<Item = &Com<SC>> {
        self.per_air
            .iter()
            .filter_map(|pk| pk.vk.preprocessed_data.as_ref())
            .map(|data| &data.commit)
    }

    pub fn preprocessed_traces(&self) -> impl Iterator<Item = Option<RowMajorMatrixView<Val<SC>>>> {
        self.per_air.iter().map(|pk| {
            pk.preprocessed_data
                .as_ref()
                .map(|data| data.trace.as_view())
        })
    }
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key.
///
/// !! This is not the full verifying key right now. It is missing AIR constraints
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct ChipsetVerifyingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkVerifyingKey<SC>>,
    /// Number of multi-matrix commitments that hold commitments to the partitioned main trace matrices across all AIRs.
    pub num_main_trace_commitments: usize,
}
