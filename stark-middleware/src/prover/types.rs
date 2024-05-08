use p3_air::Air;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::{
        debug::DebugConstraintBuilder, prover::ProverConstraintFolder, symbolic::SymbolicAirBuilder,
    },
    config::{Com, PcsProverData},
    interaction::InteractiveAir,
    rap::Rap,
    verifier::types::VerifierSingleRapMetadata,
};

use super::opener::OpeningProof;

/// Prover trace data for multiple AIRs where each AIR has partitioned main trace.
/// The different main trace parts can belong to different commitments.
pub struct ProvenMultiAirTraceData<'a, SC: StarkGenericConfig> {
    /// A list of multi-matrix commitments and their associated prover data.
    pub pcs_data: Vec<(Com<SC>, PcsProverData<SC>)>,
    // main trace, for each air, list of trace matrices and pointer to prover data for each
    /// Proven trace data for each AIR.
    pub air_traces: Vec<ProvenSingleAirTrace<'a, SC>>,
}

impl<'a, SC: StarkGenericConfig> ProvenMultiAirTraceData<'a, SC> {
    pub fn get_commit(&self, commit_index: usize) -> Option<&Com<SC>> {
        self.pcs_data.get(commit_index).map(|(commit, _)| commit)
    }

    pub fn commits(&self) -> impl Iterator<Item = &Com<SC>> {
        self.pcs_data.iter().map(|(commit, _)| commit)
    }
}

impl<'a, SC: StarkGenericConfig> Clone for ProvenMultiAirTraceData<'a, SC>
where
    PcsProverData<SC>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            pcs_data: self.pcs_data.clone(),
            air_traces: self.air_traces.clone(),
        }
    }
}

/// Partitioned main trace data for a single AIR.
///
/// We use dynamic dispatch here for the extra flexibility. The overhead is small
/// **if we ensure dynamic dispatch only once per AIR** (not true right now).
pub struct ProvenSingleAirTrace<'a, SC: StarkGenericConfig> {
    pub air: &'a dyn ProverRap<SC>,
    pub domain: Domain<SC>,
    pub partitioned_main_trace: Vec<RowMajorMatrixView<'a, Val<SC>>>,
}

impl<'a, SC: StarkGenericConfig> Clone for ProvenSingleAirTrace<'a, SC> {
    fn clone(&self) -> Self {
        Self {
            air: self.air,
            domain: self.domain,
            partitioned_main_trace: self.partitioned_main_trace.clone(),
        }
    }
}

/// Prover data for multi-matrix quotient polynomial commitment.
/// Quotient polynomials for multiple RAP matrices are committed together into a single commitment.
/// The quotient polynomials can be committed together even if the corresponding trace matrices
/// are committed separately.
pub struct ProverQuotientData<SC: StarkGenericConfig> {
    /// For each AIR, the number of quotient chunks that were committed.
    pub quotient_degrees: Vec<usize>,
    /// Quotient commitment
    pub commit: Com<SC>,
    /// Prover data for the quotient commitment
    pub data: PcsProverData<SC>,
}

#[derive(Serialize, Deserialize)]
pub struct Commitments<SC: StarkGenericConfig> {
    /// Multiple commitments, each committing to (possibly) multiple
    /// main trace matrices
    pub main_trace: Vec<Com<SC>>,
    /// Shared commitment for all permutation trace matrices
    pub perm_trace: Option<Com<SC>>,
    /// Shared commitment for all quotient polynomial evaluations
    pub quotient: Com<SC>,
}

/// The full STARK proof for a partition of multi-matrix AIRs.
/// There are multiple AIR matrices, which are partitioned by the preimage of
/// their trace commitments. In other words, multiple AIR trace matrices are committed
/// into a single commitment, and these AIRs form one part of the partition.
///
/// Includes the quotient commitments and FRI opening proofs for the constraints as well.
pub struct Proof<SC: StarkGenericConfig> {
    // TODO: this should be in verifying key
    pub rap_data: Vec<VerifierSingleRapMetadata>,
    /// The PCS commitments
    pub commitments: Commitments<SC>,
    // Opening proofs separated by partition, but this may change
    pub opening: OpeningProof<SC>,
    /// For each AIR, the cumulative sum if the AIR has interactions
    pub cumulative_sums: Vec<Option<SC::Challenge>>,
    // Should we include public values here?
}

/// RAP trait for prover dynamic dispatch use
pub trait ProverRap<SC: StarkGenericConfig>:
Air<SymbolicAirBuilder<Val<SC>>> // for quotient degree calculation
+ for<'a> InteractiveAir<ProverConstraintFolder<'a, SC>> // for permutation trace generation
    + for<'a> Rap<ProverConstraintFolder<'a, SC>> // for quotient polynomial calculation
    + for<'a> Rap<DebugConstraintBuilder<'a, SC>> // for debugging
{
}

impl<SC: StarkGenericConfig, T> ProverRap<SC> for T where
    T: Air<SymbolicAirBuilder<Val<SC>>>
        + for<'a> InteractiveAir<ProverConstraintFolder<'a, SC>>
        + for<'a> Rap<ProverConstraintFolder<'a, SC>>
        + for<'a> Rap<DebugConstraintBuilder<'a, SC>>
{
}
