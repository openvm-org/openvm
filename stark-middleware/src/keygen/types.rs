use derivative::Derivative;
use itertools::Itertools;
use p3_air::BaseAir;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::symbolic::SymbolicRapBuilder,
    commit::MatrixCommitmentPointers,
    config::{Com, PcsProverData},
    interaction::Chip,
    rap::Rap,
};

/// Widths of different parts of trace matrix
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceWidth {
    pub preprocessed: Option<usize>,
    pub partitioned_main: Vec<usize>,
    pub after_challenge: Vec<usize>,
}

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
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct StarkVerifyingKey<SC: StarkGenericConfig> {
    /// Height of trace matrix.
    pub degree: usize,
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedData<SC>>,
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// [MatrixCommitmentPointers] for partitioned main trace matrix
    pub main_graph: MatrixCommitmentPointers,
    /// The factor to multiple the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub(crate) num_challenges_to_sample: Vec<usize>,
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
pub struct MultiStarkProvingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkProvingKey<SC>>,
    /// Number of multi-matrix commitments that hold commitments to the partitioned main trace matrices across all AIRs.
    pub num_main_trace_commitments: usize,
    /// Mapping from commit_idx to global AIR index for matrix in commitment, in oder.
    pub main_commit_to_air_graph: CommitmentToAirGraph,
    /// The number of challenges to sample in each challenge phase.
    /// The length determines the global number of challenge phases.
    pub num_challenges_to_sample: Vec<usize>,
}

impl<SC: StarkGenericConfig> Default for MultiStarkProvingKey<SC> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<SC: StarkGenericConfig> MultiStarkProvingKey<SC> {
    /// Empty with 1 main trace commitment
    pub fn empty() -> Self {
        Self {
            per_air: Vec::new(),
            num_main_trace_commitments: 1,
            main_commit_to_air_graph: CommitmentToAirGraph {
                commit_to_air_index: vec![vec![]],
            },
            num_challenges_to_sample: Vec::new(),
        }
    }

    pub fn new(
        per_air: Vec<StarkProvingKey<SC>>,
        num_main_trace_commitments: usize,
        num_challenges_to_sample: Vec<usize>,
    ) -> Self {
        let air_matrices = per_air
            .iter()
            .map(|pk| pk.vk.main_graph.clone())
            .collect_vec();
        let main_commit_to_air_graph =
            create_commit_to_air_graph(&air_matrices, num_main_trace_commitments);
        Self {
            per_air,
            num_main_trace_commitments,
            main_commit_to_air_graph,
            num_challenges_to_sample,
        }
    }

    pub fn vk(&self) -> MultiStarkVerifyingKey<SC> {
        MultiStarkVerifyingKey {
            per_air: self.per_air.iter().map(|pk| pk.vk.clone()).collect(),
            main_commit_to_air_graph: self.main_commit_to_air_graph.clone(),
            num_main_trace_commitments: self.num_main_trace_commitments,
            num_challenges_to_sample: self.num_challenges_to_sample.clone(),
        }
    }

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
pub struct MultiStarkVerifyingKey<SC: StarkGenericConfig> {
    pub per_air: Vec<StarkVerifyingKey<SC>>,
    /// Number of multi-matrix commitments that hold commitments to the partitioned main trace matrices across all AIRs.
    pub num_main_trace_commitments: usize,
    /// Mapping from commit_idx to global AIR index for matrix in commitment, in oder.
    pub main_commit_to_air_graph: CommitmentToAirGraph,
    /// The number of challenges to sample in each challenge phase.
    /// The length determines the global number of challenge phases.
    pub num_challenges_to_sample: Vec<usize>,
}

impl<SC: StarkGenericConfig> MultiStarkVerifyingKey<SC> {
    pub fn new(
        per_air: Vec<StarkVerifyingKey<SC>>,
        num_main_trace_commitments: usize,
        num_challenges_to_sample: Vec<usize>,
    ) -> Self {
        let air_matrices = per_air.iter().map(|vk| vk.main_graph.clone()).collect_vec();
        let main_commit_to_air_graph =
            create_commit_to_air_graph(&air_matrices, num_main_trace_commitments);
        Self {
            per_air,
            num_main_trace_commitments,
            main_commit_to_air_graph,
            num_challenges_to_sample,
        }
    }
}

/// Assuming all AIRs are ordered and each have an index,
/// then in a system with multiple multi-matrix commitments, then
/// commit_to_air_index[commit_idx][matrix_idx] = global AIR index that the matrix corresponding to matrix_idx belongs to
#[derive(Clone, Serialize, Deserialize)]
pub struct CommitmentToAirGraph {
    pub commit_to_air_index: Vec<Vec<usize>>,
}

pub(super) fn create_commit_to_air_graph(
    air_matrices: &[MatrixCommitmentPointers],
    num_total_commitments: usize,
) -> CommitmentToAirGraph {
    let mut commit_to_air_index = vec![vec![0; air_matrices.len()]; num_total_commitments];
    for (air_idx, m) in air_matrices.iter().enumerate() {
        for ptr in &m.matrix_ptrs {
            commit_to_air_index[ptr.commit_index][ptr.matrix_index] = air_idx;
        }
    }
    CommitmentToAirGraph {
        commit_to_air_index,
    }
}

/// RAP trait to extract fixed data about the RAP for keygen
pub trait SymbolicRap<SC: StarkGenericConfig>:
    BaseAir<Val<SC>> + Chip<Val<SC>> + Rap<SymbolicRapBuilder<Val<SC>>>
{
}

impl<SC: StarkGenericConfig, T> SymbolicRap<SC> for T where
    T: BaseAir<Val<SC>> + Chip<Val<SC>> + Rap<SymbolicRapBuilder<Val<SC>>>
{
}
