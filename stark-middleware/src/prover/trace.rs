use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use crate::{
    commit::ProvenSingleMatrixView,
    config::{Com, PcsProverData},
};

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

/// The full RAP trace consists of horizontal concatenation of multiple matrices of the same height:
/// - preprocessed trace matrix
/// - the main trace matrix is horizontally partitioned into multiple matrices,
///   where each matrix can belong to a separate matrix commitment.
/// - after each round of challenges, a trace matrix for trace allowed to use those challenges
/// Each of these matrices is allowed to be in a separate commitment.
///
/// Only the main trace matrix is allowed to be partitioned, so that different parts may belong to
/// different commitments. We do not see any use cases where the `preprocessed` or `after_challenge`
/// matrices need to be partitioned.
#[derive(Clone)]
pub struct ProvenSingleRapTraceView<'a, SC: StarkGenericConfig> {
    /// Domain of the trace matrices
    pub domain: Domain<SC>,
    // Maybe public values should be included in this struct
    /// Preprocessed trace data, if any
    pub preprocessed: Option<ProvenSingleMatrixView<'a, SC>>,
    /// Main trace data, horizontally partitioned into multiple matrices
    pub partitioned_main: Vec<ProvenSingleMatrixView<'a, SC>>,
    /// `after_challenge[i] = (matrix, exposed_values)`
    /// where `matrix` is the trace matrix which uses challenges drawn
    /// after observing commitments to `preprocessed`, `partitioned_main`, and `after_challenge[..i]`,
    /// and `exposed_values` are certain values in this phase that are exposed to the verifier.
    pub after_challenge: Vec<(ProvenSingleMatrixView<'a, SC>, Vec<SC::Challenge>)>,
}
