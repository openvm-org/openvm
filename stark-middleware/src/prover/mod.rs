use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::symbolic::get_log_quotient_degree,
    config::{Com, PcsProof, PcsProverData},
    prover::trace::{ProvenSingleRapTraceView, ProvenSingleTraceView},
    verifier::types::VerifierSingleRapMetadata,
};

use self::{
    opener::OpeningProver,
    quotient::QuotientCommitter,
    types::{Commitments, Proof, ProvenMultiMatrixAirTrace},
};

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
pub mod trace;
pub mod types;

/// Proves a partition of multi-matrix AIRs.
/// This prover implementation is specialized for Interactive AIRs.
pub struct PartitionProver<SC: StarkGenericConfig> {
    pub config: SC,
}

impl<SC: StarkGenericConfig> PartitionProver<SC> {
    pub fn new(config: SC) -> Self {
        Self { config }
    }

    /// Assumes the traces have been generated already.
    ///
    /// Public values is a global list shared across all AIRs.
    #[instrument(name = "PartitionProver::prove", level = "debug", skip_all)]
    pub fn prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        // TODO: proving key,
        partition: Vec<ProvenMultiMatrixAirTrace<'a, SC>>,
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

        // TODO: preprocessed (aka proving key)

        // Challenger must observe public values
        challenger.observe_slice(public_values);
        // Challenger must observe all trace commitments
        let main_trace_commitments = partition
            .iter()
            .map(|p| p.trace_data.commit.clone())
            .collect_vec();
        challenger.observe_slice(&main_trace_commitments);

        // TODO: this is not needed if there are no interactions
        // Generate 2 permutation challenges
        let perm_challenges = [(); 2].map(|_| challenger.sample_ext_element::<SC::Challenge>());
        // TODO: refactor this
        // Flatten partitions and generate permutation traces
        let (rap_mains, perm_traces): (Vec<_>, Vec<_>) =
            tracing::info_span!("generate permutation traces").in_scope(|| {
                partition
                    .par_iter()
                    .flat_map(|part| {
                        part.airs
                            .par_iter()
                            .zip_eq(part.trace_data.traces_with_domains.par_iter())
                            .enumerate()
                            .map(|(index, (air, (domain, trace)))| {
                                let main = ProvenSingleTraceView {
                                    domain: *domain,
                                    data: &part.trace_data.data,
                                    index,
                                };
                                let perm_trace = air
                                    .generate_permutation_trace(&trace.as_view(), perm_challenges);
                                ((air, main), perm_trace)
                            })
                    })
                    .unzip()
            });
        // Skip matrices with no permutation traces
        let mut perm_traces_with_domains = Vec::new();
        let cumulative_sums_and_indices: Vec<Option<_>> = perm_traces
            .into_iter()
            .map(|trace| {
                if trace.width() != 0 && trace.height() != 0 {
                    let height = trace.height();
                    let sum = *trace.row_slice(height - 1).last().unwrap();
                    let domain = pcs.natural_domain_for_degree(height);
                    let index = perm_traces_with_domains.len();
                    perm_traces_with_domains.push((domain, trace.flatten_to_base()));
                    Some((sum, index))
                } else {
                    None
                }
            })
            .collect();
        let cumulative_sums = cumulative_sums_and_indices
            .iter()
            .map(|c| c.map(|(sum, _)| sum))
            .collect_vec();
        // Challenger needs to observe permutation_exposed_values (aka cumulative_sums)
        for cumulative_sum in cumulative_sums.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }

        // Commit to permutation traces
        // One shared commit for all permutation traces
        let perm_domains = perm_traces_with_domains
            .iter()
            .map(|(domain, _)| *domain)
            .collect_vec();
        let perm_pcs_data = tracing::info_span!("commit to permutation traces").in_scope(|| {
            // Only commit if there are permutation traces
            if !perm_traces_with_domains.is_empty() {
                let (commit, data) = pcs.commit(perm_traces_with_domains);
                challenger.observe(commit.clone());
                Some((commit, data))
            } else {
                None
            }
        });
        let perm_data = perm_pcs_data.as_ref().map(|(_, data)| (data, perm_domains));

        // Prepare the proven RAP trace views
        let (raps, trace_views): (Vec<_>, Vec<_>) = rap_mains
            .into_iter()
            .zip_eq(cumulative_sums_and_indices)
            .map(|((rap, main), cumulative_sum_and_index)| {
                let (permutation, exposed_values) =
                    if let Some((cumulative_sum, index)) = cumulative_sum_and_index {
                        let (data, domains) = perm_data.as_ref().unwrap();
                        let perm = Some(ProvenSingleTraceView {
                            domain: domains[index],
                            data: *data,
                            index,
                        });
                        (perm, vec![cumulative_sum])
                    } else {
                        (None, vec![])
                    };
                let trace_view = ProvenSingleRapTraceView {
                    main,
                    permutation,
                    permutation_exposed_values: exposed_values,
                };
                (rap, trace_view)
            })
            .unzip();

        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let quotient_committer = QuotientCommitter::new(pcs, &perm_challenges, alpha);
        let quotient_degrees = raps
            .iter()
            .map(|&rap| {
                // TODO: preprocessed width
                let d = get_log_quotient_degree::<Val<SC>, _>(rap, 0, public_values.len());
                1 << d
            })
            .collect_vec();
        let quotient_values = quotient_committer.quotient_values(
            raps,
            trace_views.clone(),
            &quotient_degrees,
            public_values,
        );
        // Commit to quotient polynomias. One shared commit for all quotient polynomials
        let quotient_data = quotient_committer.commit(quotient_values);

        // Observe quotient commitments
        challenger.observe(quotient_data.commit.clone());

        // Collect the commitments
        let commitments = Commitments {
            main_trace: main_trace_commitments,
            perm_trace: perm_pcs_data.as_ref().map(|(commit, _)| commit.clone()),
            quotient: quotient_data.commit.clone(),
        };
        // Book-keeping, build verifier metadata.
        // TODO: this should be in proving key gen
        let main_trace_ptrs = partition.iter().enumerate().flat_map(|(i, part)| {
            (0..part.trace_data.traces_with_domains.len())
                .map(|j| (i, j))
                .collect_vec()
        });
        let rap_data = trace_views
            .into_iter()
            .zip_eq(main_trace_ptrs)
            .zip_eq(quotient_degrees.iter())
            .enumerate()
            .map(
                |(index, ((view, main_trace_ptr), &quotient_degree))| VerifierSingleRapMetadata {
                    degree: view.main.domain.size(),
                    quotient_degree,
                    main_trace_ptr,
                    perm_trace_index: view.permutation.map(|p| p.index),
                    index,
                },
            )
            .collect::<Vec<_>>();

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let opener = OpeningProver::new(pcs, zeta);
        let main_data = partition
            .iter()
            .map(|part| {
                let data = &part.trace_data.data;
                let domains = part
                    .trace_data
                    .traces_with_domains
                    .iter()
                    .map(|(domain, _)| *domain)
                    .collect_vec();
                (data, domains)
            })
            .collect_vec();

        let opening = opener.open(
            challenger,
            main_data,
            perm_data,
            &quotient_data.data,
            &quotient_degrees,
        );

        Proof {
            commitments,
            opening,
            rap_data,
            cumulative_sums,
        }
    }
}
