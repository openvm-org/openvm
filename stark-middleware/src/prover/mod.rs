use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::debug::check_constraints::check_constraints,
    commit::ProvenSingleMatrixView,
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkProvingKey,
    prover::trace::ProvenSingleRapTraceView,
};

use self::{
    opener::OpeningProver,
    quotient::QuotientCommitter,
    types::{Commitments, Proof, ProvenMultiAirTraceData},
};

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
pub mod trace;
pub mod types;

/// Proves multiple chips with interactions together.
/// This prover implementation is specialized for Interactive AIRs.
pub struct MultiTraceStarkProver<SC: StarkGenericConfig> {
    pub config: SC,
}

impl<SC: StarkGenericConfig> MultiTraceStarkProver<SC> {
    pub fn new(config: SC) -> Self {
        Self { config }
    }

    /// Assumes the traces have been generated already.
    ///
    /// Public values is a global list shared across all AIRs.
    #[instrument(name = "MultiTraceStarkProver::prove", level = "debug", skip_all)]
    pub fn prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        pk: &'a MultiStarkProvingKey<SC>,
        main_trace_data: ProvenMultiAirTraceData<'a, SC>,
        public_values: &'a [Val<SC>],
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
        challenger.observe_slice(public_values);

        let preprocessed_commits: Vec<_> = pk.preprocessed_commits().cloned().collect();
        challenger.observe_slice(&preprocessed_commits);

        // Challenger must observe all trace commitments
        let main_trace_commitments = main_trace_data.commits().cloned().collect_vec();
        assert_eq!(main_trace_commitments.len(), pk.num_main_trace_commitments);
        challenger.observe_slice(&main_trace_commitments);

        // TODO: this is not needed if there are no interactions. Number of challenge rounds should be specified in proving key
        // Generate 2 permutation challenges
        let perm_challenges = [(); 2].map(|_| challenger.sample_ext_element::<SC::Challenge>());

        // TODO: ===== Permutation Trace Generation should be moved to separate module ====
        // Generate permutation traces
        let mut count = 0usize;
        let (perm_traces, cumulative_sums_and_indices): (Vec<Option<_>>, Vec<Option<_>>) =
            tracing::info_span!("generate permutation traces").in_scope(|| {
                pk.per_air
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
                        .map(|trace| {
                            // The cumulative sum is the element in last row of phi, which is the last column in perm_trace
                            let cumulative_sum =
                                *trace.row_slice(trace.height() - 1).last().unwrap();
                            let matrix_index = count;
                            count += 1;
                            (trace, (cumulative_sum, matrix_index))
                        })
                        .unzip()
                    })
                    .unzip()
            });

        // Challenger needs to observe permutation_exposed_values (aka cumulative_sums)
        for (cumulative_sum, _) in cumulative_sums_and_indices.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }

        // TODO: Move to a separate MockProver
        // Debug check constraints
        #[cfg(debug_assertions)]
        for (((preprocessed_trace, main_data), perm_trace), cumulative_sum_and_index) in pk
            .preprocessed_traces()
            .zip_eq(&main_trace_data.air_traces)
            .zip_eq(&perm_traces)
            .zip_eq(&cumulative_sums_and_indices)
        {
            let rap = main_data.air;
            let partitioned_main_trace = main_data.partitioned_main_trace;
            let perm_trace = perm_trace.map(|t| t.as_view());
            let cumulative_sum = cumulative_sum_and_index.as_ref().map(|(sum, _)| *sum);

            check_constraints(
                rap,
                &preprocessed_trace,
                &partitioned_main_trace,
                &perm_trace,
                &perm_challenges,
                cumulative_sum,
                public_values,
            );
        }

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces
        let perm_pcs_data = tracing::info_span!("commit to permutation traces").in_scope(|| {
            let mut count = 0;

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
                    ProvenSingleMatrixView::new(&p.data, 0)
                });
                let matrix_ptrs = &pk.vk.main_graph.matrix_ptrs;
                assert_eq!(main.partitioned_main_trace.len(), matrix_ptrs.len());
                let partitioned_main = matrix_ptrs
                    .iter()
                    .map(|ptr| {
                        ProvenSingleMatrixView::new(
                            &main_trace_data.pcs_data[ptr.commit_index].1,
                            ptr.matrix_index,
                        )
                    })
                    .collect_vec();

                // There will be either 0 or 1 after_challenge traces
                let after_challenge =
                    if let Some((cumulative_sum, index)) = cumulative_sum_and_index {
                        let matrix =
                            ProvenSingleMatrixView::new(&perm_pcs_data.as_ref().unwrap().1, index);
                        let exposed_values = vec![cumulative_sum];
                        vec![(matrix, exposed_values)]
                    } else {
                        Vec::new()
                    };
                let trace_view = ProvenSingleRapTraceView {
                    domain,
                    preprocessed,
                    partitioned_main,
                    after_challenge,
                };
                (rap, trace_view)
            })
            .unzip();
        // Either 0 or 1 after_challenge commits, depending on if there are any permutation traces
        let (after_challenge_commitments, after_challenge_prover_data): (Vec<_>, Vec<_>) =
            perm_pcs_data.into_iter().unzip();
        // === END of logic specific to Interactions/permutations, we can now deal with general RAP ===

        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let quotient_degrees = pk
            .per_air
            .iter()
            .map(|pk| pk.vk.quotient_degree)
            .collect_vec();
        let quotient_committer = QuotientCommitter::new(pcs, &[&perm_challenges], alpha);
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
            main_trace: main_trace_commitments,
            after_challenge: after_challenge_commitments,
            quotient: quotient_data.commit.clone(),
        };

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let opener = OpeningProver::new(pcs, zeta);
        let preprocessed_data: Vec<_> = trace_views
            .iter()
            .flat_map(|view| {
                view.preprocessed
                    .as_ref()
                    .map(|matrix| (matrix.data, view.domain))
            })
            .collect();

        let main_data: Vec<_> = main_trace_data
            .pcs_data
            .iter()
            .zip_eq(&pk.main_commit_to_air_graph.commit_to_air_index)
            .map(|((_, data), mat_to_air_index)| {
                let domains = mat_to_air_index
                    .iter()
                    .map(|i| trace_views[*i].domain)
                    .collect_vec();
                (data, domains)
            })
            .collect();

        // ASSUMING: per challenge round, shared commitment for all, with matrices in increasing order of air index
        let after_challenge_data: Vec<_> = after_challenge_prover_data
            .iter()
            .enumerate()
            .map(|(round, data)| {
                let domains = trace_views
                    .iter()
                    .flat_map(|view| (view.after_challenge.len() > round).then(|| view.domain))
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
            commitments,
            opening,
            exposed_values_after_challenge,
        }
    }
}
