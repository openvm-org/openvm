use p3_air::{Air, BaseAir};
use p3_commit::PolynomialSpace;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::{
    air_builders::{prover::ProverConstraintFolder, symbolic::SymbolicAirBuilder},
    config::{Com, PcsProof, PcsProverData},
    verifier::types::{VerifierQuotientData, VerifierTraceData},
};

use super::opener::OpenedValues;

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
    /// Expose only parts of data necessary for the verifier.
    pub fn verifier_view(&self) -> VerifierTraceData<SC> {
        let degrees = self
            .traces_with_domains
            .iter()
            .map(|(d, _)| d.size())
            .collect();
        VerifierTraceData {
            degrees,
            commit: self.commit.clone(),
        }
    }
}

/// Prover data for multiple AIRs that share a multi-matrix trace commitment.
/// Each AIR owns a separate trace matrix of a different height, but these
/// trace matrices have been committed together using the PCS.
///
/// This struct contains references to the AIRs themselves, which hold the constraint
/// information. The AIRs must support the same [AirBuilder].
///
/// We use dynamic dispatch here for the extra flexibility. The overhead is small since
/// the number of AIRs will not be more than 100 and the prover only needs to
/// dispatch once per AIR.
///
/// The ordering of `trace_data.traces` and `airs` must match.
pub struct ProvenMultiMatrixAirTrace<'a, SC: StarkGenericConfig> {
    /// Proven trace data.
    pub trace_data: &'a ProverTraceData<SC>,
    /// The AIRs that share the trace commitment.
    pub airs: Vec<&'a dyn ProverAir<SC>>,
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
    pub fn new(trace_data: &'a ProverTraceData<SC>, airs: Vec<&'a dyn ProverAir<SC>>) -> Self {
        Self { trace_data, airs }
    }
}

/// Prover data for multi-matrix quotient polynomial commitment.
/// Quotient polynomials for multiple AIRs that share a multi-matrix trace commitment
/// are committed together into a single commitment.
pub struct ProverQuotientData<SC: StarkGenericConfig> {
    /// For each AIR, the number of quotient chunks that were committed.
    pub quotient_degrees: Vec<usize>,
    /// Quotient commitment
    pub commit: Com<SC>,
    /// Prover data for the quotient commitment
    pub data: PcsProverData<SC>,
}

impl<SC: StarkGenericConfig> ProverQuotientData<SC> {
    /// Expose only parts of data necessary for the verifier.
    pub fn verifier_view(&self) -> VerifierQuotientData<SC> {
        VerifierQuotientData {
            quotient_degrees: self.quotient_degrees.clone(),
            commit: self.commit.clone(),
        }
    }
}

/// Prover data for multiple AIRs that share a single trace commitment
/// and a single quotient commitment.
pub struct ProvenDataBeforeOpening<'a, SC: StarkGenericConfig> {
    pub trace: &'a ProverTraceData<SC>,
    pub quotient: &'a ProverQuotientData<SC>,
}

/// PCS opening proof with opened values for multi-matrix AIR.
pub struct OpeningProofData<SC: StarkGenericConfig> {
    pub proof: PcsProof<SC>,
    pub values: OpenedValues<SC::Challenge>,
}

#[derive(Serialize, Deserialize)]
pub struct Commitments<SC: StarkGenericConfig> {
    pub main_trace: VerifierTraceData<SC>,
    // pub perm_trace: Com,
    // TODO: quotient can be shared across partitions I think
    pub quotient: VerifierQuotientData<SC>,
}

/// The full STARK proof for a partition of multi-matrix AIRs.
/// There are multiple AIR matrices, which are partitioned by the preimage of
/// their trace commitments. In other words, multiple AIR trace matrices are committed
/// into a single commitment, and these AIRs form one part of the partition.
///
/// Includes the quotient commitments and FRI opening proofs for the constraints as well.
pub struct PartitionedProof<SC: StarkGenericConfig> {
    // TODO: I think quotient commitment should be shared
    /// The PCS commitments
    pub commitments: Vec<Commitments<SC>>,
    // Opening proofs separated by partition, but this may change
    pub opening_proofs: Vec<OpeningProofData<SC>>,
    // Should we include public values here?
}

/// AIR trait for prover use
pub trait ProverAir<SC: StarkGenericConfig>:
    for<'a> Air<ProverConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}

impl<SC: StarkGenericConfig, T> ProverAir<SC> for T where
    T: for<'a> Air<ProverConstraintFolder<'a, SC>> + Air<SymbolicAirBuilder<Val<SC>>>
{
}
