use std::sync::{Arc, Mutex};

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::debug::check_constraints::{check_constraints, check_logup},
    commit::CommittedSingleMatrixView,
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkPartialProvingKey,
    prover::trace::SingleRapCommittedTraceView,
    rap::AnyRap,
};

use self::{
    opener::OpeningProver,
    quotient::QuotientCommitter,
    types::{Commitments, MultiAirCommittedTraceData, Proof},
};

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
pub mod trace;
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
        pk: &'a MultiStarkPartialProvingKey<SC>,
        main_trace_data: MultiAirCommittedTraceData<'a, SC>,
        public_values: &'a [Vec<Val<SC>>],
    ) -> Proof<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let pcs = self.config.pcs();

        // Challenger must observe public values
        for pis in public_values.iter() {
            challenger.observe_slice(pis);
        }

        let preprocessed_commits: Vec<_> = pk.preprocessed_commits().cloned().collect();
        challenger.observe_slice(&preprocessed_commits);

        // Challenger must observe all trace commitments
        let main_trace_commitments = main_trace_data.commits().cloned().collect_vec();
        assert_eq!(main_trace_commitments.len(), pk.num_main_trace_commitments);
        challenger.observe_slice(&main_trace_commitments);

        // TODO: this is not needed if there are no interactions. Number of challenge rounds should be specified in proving key
        // Generate 2 permutation challenges
        assert!(pk.num_challenges_to_sample.len() <= 1);
        let challenges: Vec<_> = pk
            .num_challenges_to_sample
            .iter()
            .map(|&num_challenges| {
                (0..num_challenges)
                    .map(|_| challenger.sample_ext_element::<SC::Challenge>())
                    .collect_vec()
            })
            .collect();

        // TODO: ===== Permutation Trace Generation should be moved to separate module ====
        // Generate permutation traces
        let (perm_traces, cumulative_sums_and_indices): (Vec<Option<_>>, Vec<Option<_>>) =
            tracing::info_span!("generate permutation traces").in_scope(|| {
                let perm_challenges = challenges.first().map(|c| [c[0], c[1]]); // must have 2 challenges
                let perm_traces = pk
                    .per_air
                    .par_iter()
                    .zip_eq(main_trace_data.air_traces.par_iter())
                    .map(|(pk, main)| {
                        let air = main.air;
                        let preprocessed_trace =
                            pk.preprocessed_data.as_ref().map(|d| d.trace.as_view());
                        air.generate_permutation_trace(
                            &preprocessed_trace,
                            &main.partitioned_main_trace,
                            perm_challenges,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut count = 0usize;
                let cumulative_sums_and_indices = perm_traces
                    .iter()
                    .map(|opt_trace| {
                        opt_trace.as_ref().map(|trace| {
                            // The cumulative sum is the element in last row of phi, which is the last column in perm_trace
                            let cumulative_sum =
                                *trace.row_slice(trace.height() - 1).last().unwrap();
                            let matrix_index = count;
                            count += 1;
                            (cumulative_sum, matrix_index)
                        })
                    })
                    .collect();
                (perm_traces, cumulative_sums_and_indices)
            });

        // Challenger needs to observe permutation_exposed_values (aka cumulative_sums)
        for (cumulative_sum, _) in cumulative_sums_and_indices.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }

        // TODO: Move to a separate MockProver
        // Debug check constraints
        #[cfg(debug_assertions)]
        USE_DEBUG_BUILDER.with(|debug| {
            if *debug.lock().unwrap() {
                let mut raps = vec![];
                let mut preprocessed = vec![];
                let mut partitioned_main = vec![];
                let mut permutation = vec![];
                for (
                    (((preprocessed_trace, main_data), perm_trace), cumulative_sum_and_index),
                    pis,
                ) in pk
                    .preprocessed_traces()
                    .zip_eq(&main_trace_data.air_traces)
                    .zip_eq(&perm_traces)
                    .zip_eq(&cumulative_sums_and_indices)
                    .zip_eq(public_values)
                {
                    let rap = main_data.air;
                    let partitioned_main_trace = &main_data.partitioned_main_trace;
                    let perm_trace = perm_trace.as_ref().map(|t| t.as_view());
                    let cumulative_sum = cumulative_sum_and_index.as_ref().map(|(sum, _)| *sum);

                    check_constraints(
                        rap,
                        &preprocessed_trace,
                        partitioned_main_trace,
                        perm_trace.as_slice(),
                        &challenges,
                        pis,
                        cumulative_sum.map(|c| vec![c]).as_slice(),
                    );

                    raps.push(rap);
                    preprocessed.push(preprocessed_trace);
                    partitioned_main.push(partitioned_main_trace.as_slice());
                    permutation.push(perm_trace);
                }
                check_logup(&raps, &preprocessed, &partitioned_main);
            }
        });

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces
        let perm_pcs_data = tracing::info_span!("commit to permutation traces").in_scope(|| {
            let flattened_traces_with_domains: Vec<_> = perm_traces
                .into_iter()
                .zip_eq(&main_trace_data.air_traces)
                .flat_map(|(perm_trace, data)| {
                    perm_trace.map(|trace| (data.domain, trace.flatten_to_base()))
                })
                .collect();
            // Only commit if there are permutation traces
            if !flattened_traces_with_domains.is_empty() {
                let (commit, data) = pcs.commit(flattened_traces_with_domains);
                // Challenger observes commitment
                challenger.observe(commit.clone());
                Some((commit, data))
            } else {
                None
            }
        });
        // Either 0 or 1 after_challenge commits, depending on if there are any permutation traces
        let after_challenge_pcs_data: Vec<_> = perm_pcs_data.into_iter().collect();
        let main_pcs_data = &main_trace_data.pcs_data;

        // Prepare the proven RAP trace views
        // Abstraction boundary: after this, we consider InteractiveAIR as a RAP with virtual columns included in the trace.
        let (raps, trace_views): (Vec<_>, Vec<_>) = main_trace_data
            .air_traces
            .into_iter()
            .zip_eq(&pk.per_air)
            .zip_eq(cumulative_sums_and_indices)
            .map(|((main, pk), cumulative_sum_and_index)| {
                // The AIR will be treated as the full RAP with virtual columns after this
                let rap = main.air;
                let domain = main.domain;
                let preprocessed = pk.preprocessed_data.as_ref().map(|p| {
                    // TODO: currently assuming each chip has it's own preprocessed commitment
                    CommittedSingleMatrixView::new(&p.data, 0)
                });
                let matrix_ptrs = &pk.vk.main_graph.matrix_ptrs;
                assert_eq!(main.partitioned_main_trace.len(), matrix_ptrs.len());
                let partitioned_main = matrix_ptrs
                    .iter()
                    .map(|ptr| {
                        CommittedSingleMatrixView::new(
                            main_pcs_data[ptr.commit_index].1,
                            ptr.matrix_index,
                        )
                    })
                    .collect_vec();

                // There will be either 0 or 1 after_challenge traces
                let after_challenge =
                    if let Some((cumulative_sum, index)) = cumulative_sum_and_index {
                        let matrix =
                            CommittedSingleMatrixView::new(&after_challenge_pcs_data[0].1, index);
                        let exposed_values = vec![cumulative_sum];
                        vec![(matrix, exposed_values)]
                    } else {
                        Vec::new()
                    };
                let trace_view = SingleRapCommittedTraceView {
                    domain,
                    preprocessed,
                    partitioned_main,
                    after_challenge,
                };
                (rap, trace_view)
            })
            .unzip();
        // === END of logic specific to Interactions/permutations, we can now deal with general RAP ===

        self.prove_raps_with_committed_traces(
            challenger,
            pk,
            raps,
            trace_views,
            main_pcs_data,
            &after_challenge_pcs_data,
            &challenges,
            public_values,
        )
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
    #[instrument(level = "debug", skip_all)]
    pub fn prove_raps_with_committed_traces<'a>(
        &self,
        challenger: &mut SC::Challenger,
        partial_pk: &'a MultiStarkPartialProvingKey<SC>,
        raps: Vec<&'a dyn AnyRap<SC>>,
        trace_views: Vec<SingleRapCommittedTraceView<'a, SC>>,
        main_pcs_data: &[(Com<SC>, &PcsProverData<SC>)],
        after_challenge_pcs_data: &[(Com<SC>, PcsProverData<SC>)],
        challenges: &[Vec<SC::Challenge>],
        public_values: &'a [Vec<Val<SC>>],
    ) -> Proof<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let pcs = self.config.pcs();
        let after_challenge_commitments: Vec<_> = after_challenge_pcs_data
            .iter()
            .map(|(commit, _)| commit.clone())
            .collect();

        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let degrees = trace_views
            .iter()
            .map(|view| view.domain.size())
            .collect_vec();
        let quotient_degrees = partial_pk
            .per_air
            .iter()
            .map(|pk| pk.vk.quotient_degree)
            .collect_vec();
        let quotient_committer = QuotientCommitter::new(pcs, challenges, alpha);
        let quotient_values = quotient_committer.quotient_values(
            raps,
            trace_views.clone(),
            &quotient_degrees,
            public_values,
        );
        // Commit to quotient polynomias. One shared commit for all quotient polynomials
        let quotient_data = quotient_committer.commit(quotient_values);

        // Observe quotient commitment
        challenger.observe(quotient_data.commit.clone());

        // Collect the commitments
        let commitments = Commitments {
            main_trace: main_pcs_data
                .iter()
                .map(|(commit, _)| commit.clone())
                .collect(),
            after_challenge: after_challenge_commitments,
            quotient: quotient_data.commit.clone(),
        };

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        // Open all polynomials at random points using pcs
        let opener = OpeningProver::new(pcs, zeta);
        let preprocessed_data: Vec<_> = trace_views
            .iter()
            .flat_map(|view| {
                view.preprocessed
                    .as_ref()
                    .map(|matrix| (matrix.data, view.domain))
            })
            .collect();

        let main_data: Vec<_> = main_pcs_data
            .iter()
            .zip_eq(&partial_pk.main_commit_to_air_graph.commit_to_air_index)
            .map(|((_, data), mat_to_air_index)| {
                let domains = mat_to_air_index
                    .iter()
                    .map(|i| trace_views[*i].domain)
                    .collect_vec();
                (*data, domains)
            })
            .collect();

        // ASSUMING: per challenge round, shared commitment for all trace matrices, with matrices in increasing order of air index
        let after_challenge_data: Vec<_> = after_challenge_pcs_data
            .iter()
            .enumerate()
            .map(|(round, (_, data))| {
                let domains = trace_views
                    .iter()
                    .flat_map(|view| (view.after_challenge.len() > round).then_some(view.domain))
                    .collect_vec();
                (data, domains)
            })
            .collect();

        let opening = opener.open(
            challenger,
            preprocessed_data,
            main_data,
            after_challenge_data,
            &quotient_data.data,
            &quotient_degrees,
        );

        let exposed_values_after_challenge = trace_views
            .into_iter()
            .map(|view| {
                view.after_challenge
                    .into_iter()
                    .map(|(_, values)| values)
                    .collect_vec()
            })
            .collect_vec();

        Proof {
            degrees,
            commitments,
            opening,
            exposed_values_after_challenge,
        }
    }
}
