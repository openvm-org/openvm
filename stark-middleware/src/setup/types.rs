use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::config::{Com, PcsProverData};

/// Preprocessed commitments to a batch of trace matrices, possibly of different heights.
pub struct PreprocessedTraceCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
}

impl<'pcs, SC: StarkGenericConfig> PreprocessedTraceCommitter<'pcs, SC> {
    pub fn new(pcs: &'pcs SC::Pcs) -> Self {
        Self { pcs }
    }

    /// Uses the PCS to commit to a sequence of trace matrices.
    /// The commitment will depend on the order of the matrices.
    /// The matrices may be of different heights.
    pub fn commit(
        &self,
        maybe_traces: Vec<Option<RowMajorMatrix<Val<SC>>>>,
    ) -> ProverPreprocessedTraceData<SC> {
        info_span!("commit to preprocessed trace data").in_scope(|| {
            let traces: Vec<_> = maybe_traces.iter().filter_map(|t| t.clone()).collect();

            if !traces.is_empty() {
                let traces_with_domains: Vec<_> = traces
                    .into_iter()
                    .map(|matrix| {
                        let height = matrix.height();
                        // Recomputing the domain is lightweight
                        let domain = self.pcs.natural_domain_for_degree(height);
                        (domain, matrix)
                    })
                    .collect();
                let (commit, data) = self.pcs.commit(traces_with_domains);
                ProverPreprocessedTraceData {
                    traces: maybe_traces,
                    commit: Some(commit),
                    data: Some(data),
                }
            } else {
                ProverPreprocessedTraceData {
                    traces: maybe_traces,
                    commit: None,
                    data: None,
                }
            }
        })
    }
}

/// Prover data for multi-matrix trace commitments.
/// The data is for the traces committed into a single commitment.
///
/// This data can be cached and attached to other multi-matrix traces.
pub struct ProverPreprocessedTraceData<SC: StarkGenericConfig> {
    /// Trace matrices, possibly of different heights.
    pub traces: Vec<Option<RowMajorMatrix<Val<SC>>>>,
    /// Commitment to the trace matrices.
    pub commit: Option<Com<SC>>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: Option<PcsProverData<SC>>,
}

/// Common proving key for multiple AIRs
pub struct ProvingKey<SC: StarkGenericConfig> {
    /// Prover data for multi-matrix preprocessed trace commitments
    pub trace_data: ProverPreprocessedTraceData<SC>,
}

/// Common verifying key for multiple AIRs
#[derive(Serialize, Deserialize)]
pub struct VerifyingKey<SC: StarkGenericConfig> {
    /// PCS to preprocessed traces
    pub commit: Option<Com<SC>>,
}
