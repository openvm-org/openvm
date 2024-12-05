use std::{
    iter,
    sync::{Arc, Mutex},
};

use itertools::{izip, multiunzip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractField;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    air_builders::debug::check_constraints::{check_constraints, check_logup},
    config::{Domain, StarkGenericConfig, Val},
    interaction::RapPhaseSeqKind,
    keygen::{types::MultiStarkProvingKey, view::MultiStarkProvingKeyView},
    prover::{
        metrics::trace_metrics,
        opener::OpeningProver,
        quotient::ProverQuotientData,
        trace::{commit_quotient_traces, ProverTraceData, TraceCommitter},
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

pub use trace::PairTraceView;

use crate::{config::RapPhaseSeqPartialProof, interaction::RapPhaseSeq};

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
    ) -> Proof<SC> {
        assert!(mpk.validate(&proof_input), "Invalid proof input");
        let pcs = self.config.pcs();
        let rap_phase_seq = self.config.rap_phase_seq();

        let (air_ids, air_inputs): (Vec<_>, Vec<_>) = proof_input.per_air.into_iter().unzip();
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
        challenger.observe_slice(
            &degree_per_air
                .iter()
                .map(|&d| Val::<SC>::from_canonical_usize(log2_strict_usize(d)))
                .collect::<Vec<_>>(),
        );
        let domain_per_air: Vec<_> = degree_per_air
            .iter()
            .map(|&degree| pcs.natural_domain_for_degree(degree))
            .collect();

        let preprocessed_trace_per_air = mpk
            .per_air
            .iter()
            .map(|pk| pk.preprocessed_data.as_ref().map(|d| d.trace.as_view()))
            .collect_vec();
        let trace_view_per_air = izip!(
            preprocessed_trace_per_air.iter(),
            main_views_per_air.iter(),
            pvs_per_air.iter()
        )
        .map(|(preprocessed, main, pvs)| PairTraceView {
            preprocessed,
            partitioned_main: main,
            public_values: pvs,
        })
        .collect_vec();

        let (constraints_per_air, rap_pk_per_air): (Vec<_>, Vec<_>) = mpk
            .per_air
            .iter()
            .map(|pk| (&pk.vk.symbolic_constraints, pk.rap_phase_seq_pk.clone()))
            .unzip();

        let (rap_phase_seq_proof, rap_phase_seq_data) = rap_phase_seq
            .partially_prove(
                challenger,
                &rap_pk_per_air,
                &constraints_per_air,
                &trace_view_per_air,
            )
            .map_or((None, None), |(p, d)| (Some(p), Some(d)));

        let (perm_trace_per_air, exposed_values_after_challenge, challenges) =
            if let Some(phase_data) = rap_phase_seq_data {
                assert_eq!(mpk.vk_view().num_phases(), 1);
                assert_eq!(
                    mpk.vk_view().num_challenges_in_phase(0),
                    phase_data.challenges.len()
                );
                (
                    phase_data.after_challenge_trace_per_air,
                    phase_data
                        .exposed_values_per_air
                        .into_iter()
                        .map(|v| v.into_iter().collect_vec())
                        .collect(),
                    vec![phase_data.challenges],
                )
            } else {
                assert_eq!(mpk.vk_view().num_phases(), 0);
                (vec![None; num_air], vec![vec![]; num_air], vec![])
            };

        #[cfg(debug_assertions)]
        debug_constraints_and_interactions(
            &airs,
            &mpk,
            &main_views_per_air,
            &pvs_per_air,
            &perm_trace_per_air,
            &exposed_values_after_challenge,
            &challenges,
            SC::RapPhaseSeq::ID,
        );

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces
        let perm_prover_data = tracing::info_span!("commit to permutation traces")
            .in_scope(|| commit_perm_traces::<SC>(pcs, perm_trace_per_air, &domain_per_air));

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
            exposed_values_after_challenge.clone(),
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
            exposed_values_after_challenge,
            quotient_data,
            domain_per_air,
            pvs_per_air,
            rap_phase_seq_proof,
        )
    }
}

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
    exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
    quotient_data: ProverQuotientData<SC>,
    domain_per_air: Vec<Domain<SC>>,
    public_values_per_air: Vec<Vec<Val<SC>>>,
    rap_phase_seq_proof: Option<RapPhaseSeqPartialProof<SC>>,
) -> Proof<SC> {
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
        rap_phase_seq_proof,
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

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn debug_constraints_and_interactions<SC: StarkGenericConfig>(
    raps: &[Arc<dyn AnyRap<SC>>],
    mpk: &MultiStarkProvingKeyView<SC>,
    main_views_per_air: &[Vec<RowMajorMatrixView<'_, Val<SC>>>],
    public_values_per_air: &[Vec<Val<SC>>],
    perm_trace_per_air: &[Option<RowMajorMatrix<SC::Challenge>>],
    exposed_values_after_challenge: &[Vec<Vec<SC::Challenge>>],
    challenges: &[Vec<SC::Challenge>],
    rap_phase_seq_kind: RapPhaseSeqKind,
) {
    USE_DEBUG_BUILDER.with(|debug| {
        if *debug.lock().unwrap() {
            let preprocessed = izip!(
                raps,
                &mpk.per_air,
                main_views_per_air,
                public_values_per_air,
                perm_trace_per_air,
                exposed_values_after_challenge
            )
            .map(
                |(rap, pk, main, public_values, perm_trace, exposed_values_after_challenge)| {
                    let preprocessed_trace = pk
                        .preprocessed_data
                        .as_ref()
                        .map(|data| data.trace.as_view());
                    tracing::debug!("Checking constraints for {}", rap.name());
                    check_constraints(
                        rap.as_ref(),
                        &rap.name(),
                        &preprocessed_trace,
                        main,
                        &perm_trace.iter().map(|m| m.as_view()).collect_vec(),
                        challenges,
                        public_values,
                        exposed_values_after_challenge,
                        rap_phase_seq_kind,
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
