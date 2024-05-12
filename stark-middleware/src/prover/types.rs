use p3_air::Air;
use p3_matrix::dense::RowMajorMatrixView;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::{
        debug::DebugConstraintBuilder, prover::ProverConstraintFolder, symbolic::SymbolicAirBuilder,
    },
    config::{Com, PcsProverData},
    interaction::InteractiveAir,
    rap::Rap,
};

use super::opener::OpeningProof;

/// Prover trace data for multiple AIRs where each AIR has partitioned main trace.
/// The different main trace parts can belong to different commitments.
pub struct ProvenMultiAirTraceData<'a, SC: StarkGenericConfig> {
    /// A list of multi-matrix commitments and their associated prover data.
    pub pcs_data: Vec<(Com<SC>, PcsProverData<SC>)>,
    // main trace, for each air, list of trace matrices and pointer to prover data for each
    /// Proven trace data for each AIR.
    pub air_traces: Vec<SingleAirProvenTrace<'a, SC>>,
}

impl<'a, SC: StarkGenericConfig> ProvenMultiAirTraceData<'a, SC> {
    pub fn get_domain(&self, air_index: usize) -> Domain<SC> {
        self.air_traces[air_index].domain
    }

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
pub struct SingleAirProvenTrace<'a, SC: StarkGenericConfig> {
    pub air: &'a dyn ProverRap<SC>,
    pub domain: Domain<SC>,
    pub partitioned_main_trace: Vec<RowMajorMatrixView<'a, Val<SC>>>,
}

impl<'a, SC: StarkGenericConfig> Clone for SingleAirProvenTrace<'a, SC> {
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

/// All commitments to a multi-matrix STARK that are not preprocessed.
#[derive(Serialize, Deserialize)]
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
pub struct Proof<SC: StarkGenericConfig> {
    /// The PCS commitments
    pub commitments: Commitments<SC>,
    // Opening proofs separated by partition, but this may change
    pub opening: OpeningProof<SC>,
    /// For each RAP, for each challenge phase with trace,
    /// the values to expose to the verifier in that phase
    pub exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
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
