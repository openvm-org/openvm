use std::sync::Arc;

use derivative::Derivative;
use itertools::{izip, Itertools};
use p3_commit::Pcs;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::{
    commit::CommittedSingleMatrixView,
    config::{Com, Domain, PcsProverData, StarkGenericConfig, Val},
    keygen::view::MultiStarkProvingKeyView,
    prover::quotient::{helper::QuotientVkDataHelper, ProverQuotientData, QuotientCommitter},
    rap::AnyRap,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_quotient_traces<'a, SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    mpk: &MultiStarkProvingKeyView<SC>,
    alpha: SC::Challenge,
    challenges: &[Vec<SC::Challenge>],
    raps: Vec<impl AsRef<dyn AnyRap<SC>>>,
    public_values_per_air: &[Vec<Val<SC>>],
    domain_per_air: Vec<Domain<SC>>,
    cached_mains_pdata_per_air: &'a [Vec<ProverTraceData<SC>>],
    common_main_prover_data: &'a ProverTraceData<SC>,
    perm_prover_data: &'a Option<ProverTraceData<SC>>,
    exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
) -> ProverQuotientData<SC> {
    let trace_views = create_trace_view_per_air(
        domain_per_air,
        cached_mains_pdata_per_air,
        mpk,
        exposed_values_after_challenge,
        common_main_prover_data,
        perm_prover_data,
    );
    let quotient_committer = QuotientCommitter::new(pcs, challenges, alpha);
    let qvks = mpk
        .per_air
        .iter()
        .map(|pk| pk.get_quotient_vk_data())
        .collect_vec();
    let quotient_values =
        quotient_committer.quotient_values(raps, &qvks, &trace_views, public_values_per_air);
    // Commit to quotient polynomias. One shared commit for all quotient polynomials
    quotient_committer.commit(quotient_values)
}

fn create_trace_view_per_air<'a, SC: StarkGenericConfig>(
    domain_per_air: Vec<Domain<SC>>,
    cached_mains_pdata_per_air: &'a [Vec<ProverTraceData<SC>>],
    mpk: &'a MultiStarkProvingKeyView<SC>,
    exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
    common_main_prover_data: &'a ProverTraceData<SC>,
    perm_prover_data: &'a Option<ProverTraceData<SC>>,
) -> Vec<SingleRapCommittedTraceView<'a, SC>> {
    let mut common_main_idx = 0;
    let mut after_challenge_idx = 0;
    izip!(
        domain_per_air,
        cached_mains_pdata_per_air,
        &mpk.per_air,
        exposed_values_after_challenge,
    ).map(|(domain, cached_mains_pdata, pk, exposed_values)| {
        // The AIR will be treated as the full RAP with virtual columns after this
        let preprocessed = pk.preprocessed_data.as_ref().map(|p| {
            // TODO: currently assuming each chip has it's own preprocessed commitment
            CommittedSingleMatrixView::<SC>::new(p.data.as_ref(), 0)
        });
        let mut partitioned_main: Vec<_> = cached_mains_pdata
            .iter()
            .map(|pdata| CommittedSingleMatrixView::new(pdata.data.as_ref(), 0))
            .collect();
        if pk.vk.has_common_main() {
            partitioned_main.push(CommittedSingleMatrixView::new(
                common_main_prover_data.data.as_ref(),
                common_main_idx,
            ));
            common_main_idx += 1;
        }

        let after_challenge = exposed_values
            .into_iter()
            .map(|exposed_values| {
                let matrix = CommittedSingleMatrixView::new(
                    perm_prover_data
                        .as_ref()
                        .expect("AIR exposes after_challenge values but has no permutation trace commitment")
                        .data
                        .as_ref(),
                    after_challenge_idx,
                );
                after_challenge_idx += 1;
                (matrix, exposed_values)
            })
            .collect();

        SingleRapCommittedTraceView {
            domain,
            preprocessed,
            partitioned_main,
            after_challenge,
        }
    }).collect()
}

/// Prover that commits to a batch of trace matrices, possibly of different heights.
pub struct TraceCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
}

impl<SC: StarkGenericConfig> Clone for TraceCommitter<'_, SC> {
    fn clone(&self) -> Self {
        Self { pcs: self.pcs }
    }
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
            ProverTraceData {
                commit,
                data: Arc::new(data),
            }
        })
    }
}

/// Prover data for multi-matrix trace commitments.
/// The data is for the traces committed into a single commitment.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize, PcsProverData<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>, PcsProverData<SC>: Deserialize<'de>"
))]
pub struct ProverTraceData<SC: StarkGenericConfig> {
    /// Commitment to the trace matrices.
    pub commit: Com<SC>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    /// The data is stored as a thread-safe smart [Arc] pointer because [PcsProverData] does
    /// not implement clone and should not be cloned. The prover only needs a reference to
    /// this data, so we use a smart pointer to elide lifetime concerns.
    pub data: Arc<PcsProverData<SC>>,
}

/// A view of just the preprocessed AIR, without any after challenge columns.
pub struct PairTraceView<'a, F> {
    pub preprocessed: &'a Option<RowMajorMatrixView<'a, F>>,
    pub partitioned_main: &'a [RowMajorMatrixView<'a, F>],
    pub public_values: &'a [F],
}

/// The full RAP trace consists of horizontal concatenation of multiple matrices of the same height:
/// - preprocessed trace matrix
/// - the main trace matrix is horizontally partitioned into multiple matrices,
///   where each matrix can belong to a separate matrix commitment.
/// - after each round of challenges, a trace matrix for trace allowed to use those challenges
///
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
