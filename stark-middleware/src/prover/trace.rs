use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::info_span;

use crate::prover::types::ProverTraceData;

/// Prover that commits to a batch of trace matrices, possibly of different heights.
pub struct TraceCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
}

impl<'pcs, SC: StarkGenericConfig> TraceCommitter<'pcs, SC> {
    pub fn new(pcs: &'pcs SC::Pcs) -> Self {
        Self { pcs }
    }

    /// Uses the PCS to commit to a sequence of trace matrices.
    /// The commitment will depend on the order of the matrices.
    /// The matrices may be of different heights.
    pub fn commit(&self, traces: Vec<RowMajorMatrix<Val<SC>>>) -> ProverTraceData<SC> {
        info_span!("commit to trace data").in_scope(|| {
            let traces_with_domains: Vec<_> = traces
                .into_iter()
                .map(|matrix| {
                    let height = matrix.height();
                    // Recomputing the domain is lightweight
                    let domain = self.pcs.natural_domain_for_degree(height);
                    (domain, matrix)
                })
                .collect();
            let (commit, data) = self.pcs.commit(traces_with_domains.clone());
            ProverTraceData {
                traces_with_domains,
                commit,
                data,
            }
        })
    }
}
