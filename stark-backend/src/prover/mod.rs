use std::{
    iter,
    sync::{Arc, Mutex},
};

use itertools::{izip, multiunzip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractExtensionField;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::debug::check_constraints::{check_constraints, check_logup},
    config::{Com, PcsProof, PcsProverData},
    keygen::{types::MultiStarkProvingKey, view::MultiStarkProvingKeyView},
    prover::{
        metrics::trace_metrics,
        opener::OpeningProver,
        quotient::ProverQuotientData,
        trace::{
            commit_quotient_traces, generate_permutation_traces_and_cumulative_sums,
            ProverTraceData, TraceCommitter,
        },
        types::{AirProofData, Commitments, Proof, ProofInput},
    },
    rap::AnyRap,
};

pub mod helper;
/// Metrics about trace and other statistics related to prover performance
pub mod metrics;
/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
mod trace;
pub mod types;

thread_local! {
   pub static USE_DEBUG_BUILDER: Arc<Mutex<bool>> = Arc::new(Mutex::new(true));
}

/// Proves multiple chips with interactions together.
/// This prover implementation is specialized for Interactive AIRs.
pub struct MultiTraceStarkProver<'c, SC: StarkGenericConfig> {
    pub config: &'c SC,
}

impl<'c, SC: StarkGenericConfig> MultiTraceStarkProver<'c, SC> {
    pub fn new(config: &'c SC) -> Self {
        Self { config }
    }

    pub fn pcs(&self) -> &SC::Pcs {
        self.config.pcs()
    }

    pub fn committer(&self) -> TraceCommitter<SC> {
        TraceCommitter::new(self.pcs())
    }

    /// Specialized prove for InteractiveAirs.
    /// Handles trace generation of the permutation traces.
    /// Assumes the main traces have been generated and committed already.
    ///
    /// Public values: for each AIR, a separate list of public values.
    /// The prover can support global public values that are shared among all AIRs,
    /// but we currently split public values per-AIR for modularity.
    #[instrument(name = "MultiTraceStarkProver::prove", level = "info", skip_all)]
    pub fn prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        mpk: &'a MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Proof<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        assert!(mpk.validate(&proof_input), "Invalid proof input");
        let pcs = self.config.pcs();

        let (air_ids, air_inputs): (Vec<_>, Vec<_>) = multiunzip(proof_input.per_air.into_iter());
        let (
            airs,
            cached_mains_pdata_per_air,
            cached_mains_per_air,
            common_main_per_air,
            pvs_per_air,
        ): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            multiunzip(air_inputs.into_iter().map(|input| {
                (
                    input.air,
                    input.cached_mains_pdata,
                    input.raw.cached_mains,
                    input.raw.common_main,
                    input.raw.public_values,
                )
            }));
        assert_eq!(cached_mains_pdata_per_air.len(), cached_mains_per_air.len());

        let num_air = air_ids.len();
        // Ignore unused AIRs.
        let mpk = mpk.view(air_ids);

        // Challenger must observe public values
        for pvs in &pvs_per_air {
            challenger.observe_slice(pvs);
        }

        let preprocessed_commits = mpk.vk_view().flattened_preprocessed_commits();
        challenger.observe_slice(&preprocessed_commits);

        // Commit all common main traces in a commitment. Traces inside are ordered by AIR id.
        let (common_main_trace_views, common_main_prover_data) = {
            let committer = TraceCommitter::<SC>::new(pcs);
            let (trace_views, traces): (Vec<_>, Vec<_>) = common_main_per_air
                .iter()
                .filter_map(|cm: &Option<RowMajorMatrix<_>>| cm.as_ref())
                .map(|m| (m.as_view(), m.clone()))
                .unzip();

            (trace_views, committer.commit(traces))
        };

        // Commitments order:
        // - for each air:
        //   - for each cached main trace
        //     - 1 commitment
        // - 1 commitment of all common main traces
        let main_trace_commitments: Vec<_> = cached_mains_pdata_per_air
            .iter()
            .flatten()
            .map(|pdata| &pdata.commit)
            .chain(iter::once(&common_main_prover_data.commit))
            .cloned()
            .collect();
        challenger.observe_slice(&main_trace_commitments);

        // TODO: this is not needed if there are no interactions. Number of challenge rounds should be specified in proving key
        // Generate permutation challenges
        let challenges = mpk.vk_view().sample_challenges(challenger);

        let mut common_main_idx = 0;
        let mut degree_per_air = Vec::with_capacity(num_air);
        let mut main_views_per_air = Vec::with_capacity(num_air);
        for (pk, cached_mains) in mpk.per_air.iter().zip(&cached_mains_per_air) {
            let mut main_views: Vec<_> = cached_mains.iter().map(|m| m.as_view()).collect();
            if pk.vk.has_common_main() {
                main_views.push(common_main_trace_views[common_main_idx].as_view());
                common_main_idx += 1;
            }
            degree_per_air.push(main_views[0].height());
            main_views_per_air.push(main_views);
        }
        let domain_per_air: Vec<_> = degree_per_air
            .iter()
            .map(|&degree| pcs.natural_domain_for_degree(degree))
            .collect();

        let (cumulative_sum_per_air, perm_trace_per_air) =
            generate_permutation_traces_and_cumulative_sums(
                &mpk,
                &challenges,
                &main_views_per_air,
                &pvs_per_air,
            );

        #[cfg(debug_assertions)]
        debug_constraints_and_interactions(
            &airs,
            &mpk,
            &main_views_per_air,
            &pvs_per_air,
            &perm_trace_per_air,
            &cumulative_sum_per_air,
            &challenges,
        );

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces
        let perm_prover_data = tracing::info_span!("commit to permutation traces")
            .in_scope(|| commit_perm_traces::<SC>(pcs, perm_trace_per_air, &domain_per_air));

        // Challenger needs to observe permutation_exposed_values (aka cumulative_sums)
        for cumulative_sum in cumulative_sum_per_air.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }
        // Challenger observes commitment if exists
        if let Some(data) = &perm_prover_data {
            challenger.observe(data.commit.clone());
        }
        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let quotient_data = commit_quotient_traces(
            pcs,
            &mpk,
            alpha,
            &challenges,
            airs,
            &pvs_per_air,
            domain_per_air.clone(),
            &cached_mains_pdata_per_air,
            &common_main_prover_data,
            &perm_prover_data,
            cumulative_sum_per_air.clone(),
        );

        let main_prover_data: Vec<_> = cached_mains_pdata_per_air
            .into_iter()
            .flatten()
            .chain(iter::once(common_main_prover_data))
            .collect();
        prove_raps_with_committed_traces(
            pcs,
            challenger,
            mpk,
            &main_prover_data,
            perm_prover_data,
            cumulative_sum_per_air,
            quotient_data,
            domain_per_air,
            pvs_per_air,
        )
    }
}
//
/// Proves general RAPs after all traces have been committed.
/// Soundness depends on `challenger` having already observed
/// public values, exposed values after challenge, and all
/// trace commitments.
///
/// - `challenges`: for each trace challenge phase, the challenges sampled
///
/// ## Assumptions
/// - `raps, trace_views, public_values` have same length and same order
/// - per challenge round, shared commitment for
/// all trace matrices, with matrices in increasing order of air index
#[allow(clippy::too_many_arguments)]
#[instrument(level = "info", skip_all)]
fn prove_raps_with_committed_traces<'a, SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    challenger: &mut SC::Challenger,
    mpk: MultiStarkProvingKeyView<SC>,
    main_prover_data: &[ProverTraceData<SC>],
    perm_prover_data: Option<ProverTraceData<SC>>,
    cumulative_sum_per_air: Vec<Option<SC::Challenge>>,
    quotient_data: ProverQuotientData<SC>,
    domain_per_air: Vec<Domain<SC>>,
    public_values_per_air: Vec<Vec<Val<SC>>>,
) -> Proof<SC>
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    // Observe quotient commitment
    challenger.observe(quotient_data.commit.clone());

    let after_challenge_commitments: Vec<_> = perm_prover_data
        .iter()
        .map(|data| data.commit.clone())
        .collect();
    // Collect the commitments
    let commitments = Commitments {
        main_trace: main_prover_data
            .iter()
            .map(|data| data.commit.clone())
            .collect(),
        after_challenge: after_challenge_commitments,
        quotient: quotient_data.commit.clone(),
    };

    // Draw `zeta` challenge
    let zeta: SC::Challenge = challenger.sample_ext_element();
    tracing::debug!("zeta: {zeta:?}");

    // Open all polynomials at random points using pcs
    let opener = OpeningProver::new(pcs, zeta);
    let preprocessed_data: Vec<_> = mpk
        .per_air
        .iter()
        .zip_eq(&domain_per_air)
        .flat_map(|(pk, domain)| {
            pk.preprocessed_data
                .as_ref()
                .map(|prover_data| (prover_data.data.as_ref(), *domain))
        })
        .collect();

    let mut main_prover_data_idx = 0;
    let mut main_data = Vec::with_capacity(main_prover_data.len());
    let mut common_main_domains = Vec::with_capacity(mpk.per_air.len());
    for (air_id, pk) in mpk.per_air.iter().enumerate() {
        for _ in 0..pk.vk.num_cached_mains() {
            main_data.push((
                main_prover_data[main_prover_data_idx].data.as_ref(),
                vec![domain_per_air[air_id]],
            ));
            main_prover_data_idx += 1;
        }
        if pk.vk.has_common_main() {
            common_main_domains.push(domain_per_air[air_id]);
        }
    }
    main_data.push((
        main_prover_data[main_prover_data_idx].data.as_ref(),
        common_main_domains,
    ));

    // ASSUMING: per challenge round, shared commitment for all trace matrices, with matrices in increasing order of air index
    let after_challenge_data = if let Some(perm_prover_data) = &perm_prover_data {
        let mut domains = Vec::new();
        for (air_id, pk) in mpk.per_air.iter().enumerate() {
            if pk.vk.has_interaction() {
                domains.push(domain_per_air[air_id]);
            }
        }
        vec![(perm_prover_data.data.as_ref(), domains)]
    } else {
        vec![]
    };

    let quotient_degrees = mpk
        .per_air
        .iter()
        .map(|pk| pk.vk.quotient_degree)
        .collect_vec();
    let opening = opener.open(
        challenger,
        preprocessed_data,
        main_data,
        after_challenge_data,
        &quotient_data.data,
        &quotient_degrees,
    );

    let degrees = domain_per_air
        .iter()
        .map(|domain| domain.size())
        .collect_vec();

    let exposed_values_after_challenge = cumulative_sum_per_air
        .into_iter()
        .map(|csum| {
            if let Some(csum) = csum {
                vec![vec![csum]]
            } else {
                vec![]
            }
        })
        .collect_vec();

    tracing::info!("{}", trace_metrics(&mpk.per_air, &degrees));
    #[cfg(feature = "bench-metrics")]
    trace_metrics(&mpk.per_air, &degrees).emit();

    Proof {
        commitments,
        opening,
        per_air: izip!(
            mpk.air_ids,
            degrees,
            exposed_values_after_challenge,
            public_values_per_air
        )
        .map(
            |(air_id, degree, exposed_values, public_values)| AirProofData {
                air_id,
                degree,
                public_values,
                exposed_values_after_challenge: exposed_values,
            },
        )
        .collect(),
    }
}

fn commit_perm_traces<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    perm_traces: Vec<Option<RowMajorMatrix<SC::Challenge>>>,
    domain_per_air: &[Domain<SC>],
) -> Option<ProverTraceData<SC>> {
    let flattened_traces_with_domains: Vec<_> = perm_traces
        .into_iter()
        .zip_eq(domain_per_air)
        .flat_map(|(perm_trace, domain)| perm_trace.map(|trace| (*domain, trace.flatten_to_base())))
        .collect();
    // Only commit if there are permutation traces
    if !flattened_traces_with_domains.is_empty() {
        let (commit, data) = pcs.commit(flattened_traces_with_domains);
        Some(ProverTraceData {
            commit,
            data: data.into(),
        })
    } else {
        None
    }
}

fn debug_constraints_and_interactions<SC: StarkGenericConfig>(
    raps: &[Arc<dyn AnyRap<SC>>],
    mpk: &MultiStarkProvingKeyView<SC>,
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
    perm_trace_per_air: &[Option<RowMajorMatrix<SC::Challenge>>],
    cumulative_sum_per_air: &[Option<SC::Challenge>],
    challenges: &[Vec<SC::Challenge>],
) {
    USE_DEBUG_BUILDER.with(|debug| {
        if *debug.lock().unwrap() {
            let preprocessed = izip!(
                raps,
                &mpk.per_air,
                main_views_per_air,
                public_values_per_air,
                perm_trace_per_air,
                cumulative_sum_per_air
            )
            .map(
                |(rap, pk, main, public_values, perm_trace, cumulative_sum)| {
                    let preprocessed_trace = pk
                        .preprocessed_data
                        .as_ref()
                        .map(|data| data.trace.as_view());
                    check_constraints(
                        rap.as_ref(),
                        &preprocessed_trace,
                        main,
                        &perm_trace.iter().map(|m| m.as_view()).collect_vec(),
                        challenges,
                        public_values,
                        cumulative_sum.map(|c| vec![c]).as_slice(),
                    );
                    preprocessed_trace
                },
            )
            .collect_vec();

            let (air_names, interactions): (Vec<_>, Vec<_>) = mpk
                .per_air
                .iter()
                .map(|pk| {
                    (
                        pk.air_name.clone(),
                        &pk.vk.symbolic_constraints.interactions[..],
                    )
                })
                .unzip();
            check_logup(
                &air_names,
                &interactions,
                &preprocessed,
                main_views_per_air,
                public_values_per_air,
            );
        }
    });
}
