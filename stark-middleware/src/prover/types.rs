use p3_air::Air;
use p3_matrix::dense::RowMajorMatrix;
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

use super::{opener::OpeningProof, trace::ProvenSingleTraceView};

/// Prover data for multi-matrix trace commitments.
/// The data is for the traces committed into a single commitment.
///
/// This data can be cached and attached to other multi-matrix traces.
pub struct ProverTraceData<SC: StarkGenericConfig> {
    /// Trace matrices, possibly of different heights.
    /// We store the domain each trace was committed with respect to.
    // Memory optimization? PCS ProverData should be able to recover the domain.
    pub traces_with_domains: Vec<(Domain<SC>, RowMajorMatrix<Val<SC>>)>,
    /// Commitment to the trace matrices.
    pub commit: Com<SC>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: PcsProverData<SC>,
}

impl<SC: StarkGenericConfig> ProverTraceData<SC> {
    pub fn get(&self, index: usize) -> Option<ProvenSingleTraceView<SC>> {
        self.traces_with_domains
            .get(index)
            .map(|(domain, _)| ProvenSingleTraceView {
                domain: *domain,
                data: &self.data,
                index,
            })
    }
}

/// Prover data for multiple AIRs that share a multi-matrix trace commitment.
/// Each AIR owns a separate trace matrix of a different height, but these
/// trace matrices have been committed together using the PCS.
///
/// This struct contains references to the AIRs themselves, which hold the constraint
/// information. The AIRs must support the same [AirBuilder].
///
/// We use dynamic dispatch here for the extra flexibility. The overhead is small
/// **if we ensure dynamic dispatch only once per AIR** (not true right now).
///
/// The ordering of `trace_data.traces_with_domains` and `airs` must match.
pub struct ProvenMultiMatrixAirTrace<'a, SC: StarkGenericConfig> {
    /// Proven trace data.
    pub trace_data: &'a ProverTraceData<SC>,
    /// The AIRs that share the trace commitment.
    pub airs: Vec<&'a dyn ProverRap<SC>>,
}

impl<'a, SC: StarkGenericConfig> Clone for ProvenMultiMatrixAirTrace<'a, SC> {
    fn clone(&self) -> Self {
        Self {
            trace_data: self.trace_data,
            airs: self.airs.clone(),
        }
    }
}

impl<'a, SC: StarkGenericConfig> ProvenMultiMatrixAirTrace<'a, SC> {
    pub fn new(trace_data: &'a ProverTraceData<SC>, airs: Vec<&'a dyn ProverRap<SC>>) -> Self {
        Self { trace_data, airs }
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
+ Rap<SymbolicAirBuilder<Val<SC>>> // for quotient degree calculation
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
