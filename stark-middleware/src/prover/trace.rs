use derivative::Derivative;
use itertools::Itertools;
use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::info_span;

use crate::{
    commit::CommittedSingleMatrixView,
    config::{Com, PcsProverData},
    keygen::types::MultiStarkVerifyingKey,
};

use super::types::{MultiAirCommittedTraceData, ProverRap, SingleAirCommittedTrace};

/// Stateful builder to help with computing multi-stark trace commitments
pub struct TraceCommitmentBuilder<'a, SC: StarkGenericConfig> {
    pub committer: TraceCommitter<'a, SC>,
    traces_to_commit: Vec<RowMajorMatrix<Val<SC>>>,
    committed_traces: Vec<Vec<RowMajorMatrix<Val<SC>>>>,
    data: Vec<(Com<SC>, PcsProverData<SC>)>,
}

impl<'a, SC: StarkGenericConfig> TraceCommitmentBuilder<'a, SC> {
    pub fn new(pcs: &'a SC::Pcs) -> Self {
        Self {
            committer: TraceCommitter::new(pcs),
            traces_to_commit: Vec::new(),
            committed_traces: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add trace to list of to-be-committed
    pub fn load_trace(&mut self, trace: RowMajorMatrix<Val<SC>>) {
        self.traces_to_commit.push(trace);
    }

    pub fn commit_current(&mut self) {
        let traces = std::mem::take(&mut self.traces_to_commit);
        let data = self.committer.commit(traces.clone());
        self.data.push((data.commit, data.data));
        self.committed_traces.push(traces);
    }

    /// Loads `trace` assumed to have already been committed as single matrix commitment in `data`.
    pub fn load_cached_trace(&mut self, trace: RowMajorMatrix<Val<SC>>, data: ProverTraceData<SC>) {
        self.committed_traces.push(vec![trace]);
        self.data.push((data.commit, data.data));
    }

    pub fn view<'b>(
        &'b self,
        vk: &MultiStarkVerifyingKey<SC>,
        airs: Vec<&'b dyn ProverRap<SC>>,
    ) -> MultiAirCommittedTraceData<'b, SC>
    where
        'a: 'b,
    {
        let pcs_data = self
            .data
            .iter()
            .map(|(commit, data)| (commit.clone(), data))
            .collect_vec();
        let air_traces = airs
            .into_iter()
            .zip_eq(&vk.per_air)
            .map(|(air, vk)| {
                let partitioned_main_trace = vk
                    .main_graph
                    .matrix_ptrs
                    .iter()
                    .map(|ptr| self.committed_traces[ptr.commit_index][ptr.matrix_index].as_view())
                    .collect_vec();
                SingleAirCommittedTrace {
                    air,
                    domain: self.committer.pcs.natural_domain_for_degree(vk.degree),
                    partitioned_main_trace,
                }
            })
            .collect();
        MultiAirCommittedTraceData {
            pcs_data,
            air_traces,
        }
    }
}

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
            let (commit, data) = self.pcs.commit(traces_with_domains);
            ProverTraceData { commit, data }
        })
    }
}

/// Prover data for multi-matrix trace commitments.
/// The data is for the traces committed into a single commitment.
pub struct ProverTraceData<SC: StarkGenericConfig> {
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
#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
pub struct SingleRapCommittedTraceView<'a, SC: StarkGenericConfig> {
    /// Domain of the trace matrices
    pub domain: Domain<SC>,
    // Maybe public values should be included in this struct
    /// Preprocessed trace data, if any
    pub preprocessed: Option<CommittedSingleMatrixView<'a, SC>>,
    /// Main trace data, horizontally partitioned into multiple matrices
    pub partitioned_main: Vec<CommittedSingleMatrixView<'a, SC>>,
    /// `after_challenge[i] = (matrix, exposed_values)`
    /// where `matrix` is the trace matrix which uses challenges drawn
    /// after observing commitments to `preprocessed`, `partitioned_main`, and `after_challenge[..i]`,
    /// and `exposed_values` are certain values in this phase that are exposed to the verifier.
    pub after_challenge: Vec<(CommittedSingleMatrixView<'a, SC>, Vec<SC::Challenge>)>,
}
