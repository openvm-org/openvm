use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use crate::{config::PcsProverData, prover::types::ProverTraceData};

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

/// The PCS commits to multiple trace matrices at once, so this struct stores
/// references to get all data relevant to a single AIR trace.
pub struct ProvenSingleTraceView<'a, SC: StarkGenericConfig> {
    /// Trace domain
    pub domain: Domain<SC>,
    /// Prover data, includes LDE matrix of trace and Merkle tree.
    /// The prover data can commit to multiple trace matrices, so
    /// `index` is needed to identify this trace.
    pub data: &'a PcsProverData<SC>,
    /// The index of the trace in the prover data.
    pub index: usize,
}

/// The domain of `main` and `permutation` must be the same.
pub struct ProvenSingleRapTraceView<'a, SC: StarkGenericConfig> {
    /// Preprocessed trace data
    pub preprocessed: Option<ProvenSingleTraceView<'a, SC>>,
    /// Main trace data
    pub main: ProvenSingleTraceView<'a, SC>,
    /// Permutation trace data
    pub permutation: Option<ProvenSingleTraceView<'a, SC>>,
    /// Exposed values of the permutation
    pub permutation_exposed_values: Vec<SC::Challenge>,
}

impl<'a, SC: StarkGenericConfig> Clone for ProvenSingleTraceView<'a, SC> {
    fn clone(&self) -> Self {
        Self {
            domain: self.domain,
            data: self.data,
            index: self.index,
        }
    }
}

impl<'a, SC: StarkGenericConfig> Clone for ProvenSingleRapTraceView<'a, SC> {
    fn clone(&self) -> Self {
        Self {
            preprocessed: self.preprocessed.clone(),
            main: self.main.clone(),
            permutation: self.permutation.clone(),
            permutation_exposed_values: self.permutation_exposed_values.clone(),
        }
    }
}
