use derivative::Derivative;
use p3_matrix::dense::RowMajorMatrixView;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use super::opener::OpeningProof;
use crate::{
    config::{Com, PcsProverData},
    rap::AnyRap,
};

/// Prover trace data for multiple AIRs where each AIR has partitioned main trace.
/// The different main trace parts can belong to different commitments.
#[derive(Derivative)]
#[derivative(Clone(bound = "PcsProverData<SC>: Clone"))]
pub struct MultiAirCommittedTraceData<'a, SC: StarkGenericConfig> {
    /// A list of multi-matrix commitments and their associated prover data.
    pub pcs_data: Vec<(Com<SC>, &'a PcsProverData<SC>)>,
    // main trace, for each air, list of trace matrices and pointer to prover data for each
    /// Proven trace data for each AIR.
    pub air_traces: Vec<SingleAirCommittedTrace<'a, SC>>,
}

impl<'a, SC: StarkGenericConfig> MultiAirCommittedTraceData<'a, SC> {
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

/// Partitioned main trace data for a single AIR.
///
/// We use dynamic dispatch here for the extra flexibility. The overhead is small
/// **if we ensure dynamic dispatch only once per AIR** (not true right now).
pub struct SingleAirCommittedTrace<'a, SC: StarkGenericConfig> {
    pub air: &'a dyn AnyRap<SC>,
    pub domain: Domain<SC>,
    pub partitioned_main_trace: Vec<RowMajorMatrixView<'a, Val<SC>>>,
}

impl<'a, SC: StarkGenericConfig> Clone for SingleAirCommittedTrace<'a, SC> {
    fn clone(&self) -> Self {
        Self {
            air: self.air,
            domain: self.domain,
            partitioned_main_trace: self.partitioned_main_trace.clone(),
        }
    }
}

/// All commitments to a multi-matrix STARK that are not preprocessed.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
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
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    /// For each RAP, the height of trace matrix.
    pub degrees: Vec<usize>,
    /// The PCS commitments
    pub commitments: Commitments<SC>,
    // Opening proofs separated by partition, but this may change
    pub opening: OpeningProof<SC>,
    /// For each RAP, for each challenge phase with trace,
    /// the values to expose to the verifier in that phase
    pub exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
    // Should we include public values here?
}

impl<SC: StarkGenericConfig> Proof<SC> {
    pub fn log_degrees(&self) -> Vec<usize> {
        self.degrees.iter().map(|d| log2_strict_usize(*d)).collect()
    }
}
