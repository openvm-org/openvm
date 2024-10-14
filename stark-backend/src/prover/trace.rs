use std::sync::Arc;

use derivative::Derivative;
use itertools::{izip, Itertools};
use p3_commit::Pcs;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::{
    commit::CommittedSingleMatrixView,
    config::{Com, PcsProof, PcsProverData},
    interaction::trace::generate_permutation_trace,
    keygen::{types::StarkProvingKey, view::MultiStarkProvingKeyView},
    prover::{
        quotient::{helper::QuotientVkDataHelper, ProverQuotientData, QuotientCommitter},
        types::CommittedTraceData,
    },
    rap::AnyRap,
};

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn generate_permutation_traces_and_cumulative_sums<SC: StarkGenericConfig>(
    mpk: &MultiStarkProvingKeyView<SC>,
    challenges: &[Vec<SC::Challenge>],
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
) -> (
    Vec<Option<SC::Challenge>>,
    Vec<Option<RowMajorMatrix<SC::Challenge>>>,
)
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let perm_trace_per_air = tracing::info_span!("generate permutation traces").in_scope(|| {
        generate_permutation_trace_per_air(
            challenges,
            mpk,
            main_views_per_air,
            public_values_per_air,
        )
    });
    let cumulative_sum_per_air = extract_cumulative_sums::<SC>(&perm_trace_per_air);

    (cumulative_sum_per_air, perm_trace_per_air)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_quotient_traces<'a, SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    mpk: &MultiStarkProvingKeyView<SC>,
    alpha: SC::Challenge,
    challenges: &[Vec<SC::Challenge>],
    raps: Vec<impl AsRef<dyn AnyRap<SC>>>,
    public_values_per_air: &[Vec<Val<SC>>],
    domain_per_air: Vec<Domain<SC>>,
    cached_mains_per_air: &'a [Vec<CommittedTraceData<SC>>],
    common_main_prover_data: &'a ProverTraceData<SC>,
    perm_prover_data: &'a Option<ProverTraceData<SC>>,
    cumulative_sum_per_air: Vec<Option<SC::Challenge>>,
) -> ProverQuotientData<SC>
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let trace_views = create_trace_view_per_air(
        domain_per_air,
        cached_mains_per_air,
        mpk,
        cumulative_sum_per_air,
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

/// Returns a list of optional tuples of (permutation trace,cumulative sum) for each AIR.
fn generate_permutation_trace_per_air<SC: StarkGenericConfig>(
    challenges: &[Vec<SC::Challenge>],
    mpk: &MultiStarkProvingKeyView<SC>,
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
) -> Vec<Option<RowMajorMatrix<SC::Challenge>>>
where
    StarkProvingKey<SC>: Send + Sync,
{
    // Generate permutation traces
    let perm_challenges = challenges.first().map(|c| [c[0], c[1]]); // must have 2 challenges

    mpk.per_air
        .par_iter()
        .zip_eq(main_views_per_air.par_iter())
        .zip_eq(public_values_per_air.par_iter())
        .map(|((pk, main), public_values)| {
            let interactions = &pk.vk.symbolic_constraints.interactions;
            let preprocessed_trace = pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
            generate_permutation_trace(
                interactions,
                &preprocessed_trace,
                main,
                public_values,
                perm_challenges,
                pk.interaction_chunk_size,
            )
        })
        .collect::<Vec<_>>()
}

fn extract_cumulative_sums<SC: StarkGenericConfig>(
    perm_traces: &[Option<RowMajorMatrix<SC::Challenge>>],
) -> Vec<Option<SC::Challenge>> {
    perm_traces
        .iter()
        .map(|perm_trace| {
            perm_trace.as_ref().map(|perm_trace| {
                *perm_trace
                    .row_slice(perm_trace.height() - 1)
                    .last()
                    .unwrap()
            })
        })
        .collect()
}

fn create_trace_view_per_air<'a, SC: StarkGenericConfig>(
    domain_per_air: Vec<Domain<SC>>,
    cached_mains_per_air: &'a [Vec<CommittedTraceData<SC>>],
    mpk: &'a MultiStarkProvingKeyView<SC>,
    cumulative_sum_per_air: Vec<Option<SC::Challenge>>,
    common_main_prover_data: &'a ProverTraceData<SC>,
    perm_prover_data: &'a Option<ProverTraceData<SC>>,
) -> Vec<SingleRapCommittedTraceView<'a, SC>> {
    let mut common_main_idx = 0;
    let mut after_challenge_idx = 0;
    izip!(
        domain_per_air,
        cached_mains_per_air,
        &mpk.per_air,
        cumulative_sum_per_air,
    )
    .map(|(domain, cached_mains, pk, cumulative_sum)| {
        // The AIR will be treated as the full RAP with virtual columns after this
        let preprocessed = pk.preprocessed_data.as_ref().map(|p| {
            // TODO: currently assuming each chip has it's own preprocessed commitment
            CommittedSingleMatrixView::<SC>::new(p.data.as_ref(), 0)
        });
        let mut partitioned_main: Vec<_> = cached_mains
            .iter()
            .map(|cm| CommittedSingleMatrixView::new(cm.prover_data.data.as_ref(), 0))
            .collect();
        if pk.vk.has_common_main() {
            partitioned_main.push(CommittedSingleMatrixView::new(
                common_main_prover_data.data.as_ref(),
                common_main_idx,
            ));
            common_main_idx += 1;
        }

        // There will be either 0 or 1 after_challenge traces
        let after_challenge = if let Some(cumulative_sum) = cumulative_sum {
            let matrix = CommittedSingleMatrixView::new(
                perm_prover_data
                    .as_ref()
                    .expect("AIR uses interactions but no permutation trace commitment")
                    .data
                    .as_ref(),
                after_challenge_idx,
            );
            after_challenge_idx += 1;
            let exposed_values = vec![cumulative_sum];
            vec![(matrix, exposed_values)]
        } else {
            Vec::new()
        };
        SingleRapCommittedTraceView {
            domain,
            preprocessed,
            partitioned_main,
            after_challenge,
        }
    })
    .collect()
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
