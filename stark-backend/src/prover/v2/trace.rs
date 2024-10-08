use itertools::{izip, Itertools};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    commit::CommittedSingleMatrixView,
    config::{Com, PcsProof, PcsProverData},
    interaction::trace::generate_permutation_trace,
    keygen::v2::{types::StarkProvingKeyV2, view::MultiStarkProvingKeyV2View},
    prover::{
        commit_perm_traces,
        quotient::{helper::QuotientVKDataHelper, ProverQuotientData, QuotientCommitter},
        trace::{ProverTraceData, SingleRapCommittedTraceView},
        v2::types::CommittedTraceData,
    },
    rap::AnyRap,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_permutation_traces<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    mpk: &MultiStarkProvingKeyV2View<SC>,
    challenges: &[Vec<SC::Challenge>],
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
    domain_per_air: Vec<Domain<SC>>,
) -> (Vec<Option<SC::Challenge>>, Option<ProverTraceData<SC>>)
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
    // Commit to permutation traces: this means only 1 challenge round right now
    // One shared commit for all permutation traces
    let perm_prover_data = tracing::info_span!("commit to permutation traces")
        .in_scope(|| commit_perm_traces::<SC>(pcs, perm_trace_per_air, &domain_per_air));

    (cumulative_sum_per_air, perm_prover_data)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_quotient_traces<'a, SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    mpk: &MultiStarkProvingKeyV2View<SC>,
    alpha: SC::Challenge,
    challenges: &[Vec<SC::Challenge>],
    raps: Vec<&'a dyn AnyRap<SC>>,
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
    mpk: &MultiStarkProvingKeyV2View<SC>,
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
) -> Vec<Option<RowMajorMatrix<SC::Challenge>>>
where
    StarkProvingKeyV2<SC>: Send + Sync,
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
    mpk: &'a MultiStarkProvingKeyV2View<SC>,
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
