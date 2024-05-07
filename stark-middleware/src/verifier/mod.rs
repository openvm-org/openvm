use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

pub mod constraints;
mod error;
pub mod types;

pub use error::*;

use crate::{
    prover::{opener::AdjacentOpenedValues, types::Proof},
    setup::types::VerifyingKey,
};

use self::{constraints::verify_single_rap_constraints, types::VerifierRap};

/// Verifies a partitioned proof of multi-matrix AIRs.
// TODO: Interactions
pub struct PartitionVerifier<SC: StarkGenericConfig> {
    config: SC,
}

impl<SC: StarkGenericConfig> PartitionVerifier<SC> {
    pub fn new(config: SC) -> Self {
        Self { config }
    }

    // It would be better to pass in only symbolic constraints, which can be serialized.
    /// Ordering of `raps` and `proof.rap_data` should match.
    /// Public values is a global list shared across all AIRs.
    #[instrument(name = "PartitionVerifier::verify", level = "debug", skip_all)]
    pub fn verify(
        &self,
        challenger: &mut SC::Challenger,
        vk: VerifyingKey<SC>,
        raps: Vec<&dyn VerifierRap<SC>>,
        proof: Proof<SC>,
        public_values: &[Val<SC>],
    ) -> Result<(), VerificationError> {
        // Challenger must observe public values
        challenger.observe_slice(public_values);

        // TODO: valid shape check from verifying key
        let preprocessed_commits: Vec<_> = vk
            .preprocessed_data
            .iter()
            .filter_map(|md| md.as_ref().map(|data| data.commit.clone()))
            .collect();
        challenger.observe_slice(&preprocessed_commits);

        // Observe main trace commitments
        challenger.observe_slice(&proof.commitments.main_trace);
        // Sample permutation challenges
        let perm_challenges = [(); 2].map(|_| challenger.sample_ext_element::<SC::Challenge>());

        // Observe cumulative sums
        for cumulative_sum in proof.cumulative_sums.iter().flatten() {
            challenger.observe_slice(cumulative_sum.as_base_slice());
        }
        // Observe permutation trace commitments
        if let Some(perm_trace_commit) = &proof.commitments.perm_trace {
            challenger.observe(perm_trace_commit.clone());
        }

        // Draw `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        // Observe quotient commitments
        challenger.observe(proof.commitments.quotient.clone());

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let pcs = self.config.pcs();
        let opened_values = proof.opening.values;
        // Map from opening index -> AIR index
        let index_lookup_preprocessed: Vec<_> = vk
            .preprocessed_data
            .iter()
            .enumerate()
            .filter_map(|(i, md)| md.as_ref().map(|_| i))
            .collect();
        // Partition to flat index lookup
        let mut index_lookup_main = opened_values
            .main
            .iter()
            .map(|part| vec![0; part.len()])
            .collect_vec();
        let mut index_lookup_perm = vec![0; opened_values.perm.as_ref().unwrap_or(&vec![]).len()];
        // Build domains
        let (domains, quotient_chunks_domains): (Vec<_>, Vec<Vec<_>>) = proof
            .rap_data
            .iter()
            .map(|data| {
                let degree = data.degree;
                let quotient_degree = data.quotient_degree;
                let domain = pcs.natural_domain_for_degree(degree);
                let quotient_domain = domain.create_disjoint_domain(degree * quotient_degree);
                let qc_domains = quotient_domain.split_domains(quotient_degree);
                let (i, j) = data.main_trace_ptr;
                // TODO: out of bounds error
                index_lookup_main[i][j] = data.index;
                if let Some(i) = data.perm_trace_index {
                    index_lookup_perm[i] = data.index;
                }
                (domain, qc_domains)
            })
            .unzip();
        // Verify all opening proofs
        let trace_domain_and_openings =
            |domain: Domain<SC>,
             zeta: SC::Challenge,
             values: &AdjacentOpenedValues<SC::Challenge>| {
                (
                    domain,
                    vec![
                        (zeta, values.local.clone()),
                        (domain.next_point(zeta).unwrap(), values.next.clone()),
                    ],
                )
            };
        let mut rounds: Vec<_> = opened_values
            .preprocessed
            .iter()
            .enumerate()
            .map(|(i, values)| {
                let index = index_lookup_preprocessed[i];
                let data = vk.preprocessed_data[index].as_ref().unwrap();
                let domain = pcs.natural_domain_for_degree(data.degree);
                let domain_and_openings = trace_domain_and_openings(domain, zeta, values);
                (data.commit.clone(), vec![domain_and_openings])
            })
            .collect();
        opened_values
            .main
            .iter()
            .enumerate()
            .for_each(|(i, values_per_mat)| {
                let domains_and_openings = values_per_mat
                    .iter()
                    .enumerate()
                    .map(|(j, values)| {
                        let domain = domains[index_lookup_main[i][j]];
                        trace_domain_and_openings(domain, zeta, values)
                    })
                    .collect_vec();
                rounds.push((
                    proof.commitments.main_trace[i].clone(),
                    domains_and_openings,
                ));
            });
        if let Some(values_per_mat) = &opened_values.perm {
            let domains_and_openings = values_per_mat
                .iter()
                .enumerate()
                .map(|(j, values)| {
                    let domain = domains[index_lookup_perm[j]];
                    trace_domain_and_openings(domain, zeta, values)
                })
                .collect_vec();
            rounds.push((proof.commitments.perm_trace.unwrap(), domains_and_openings));
        }
        let quotient_domains_and_openings = opened_values
            .quotient
            .iter()
            .enumerate()
            .flat_map(|(i, chunk)| {
                chunk
                    .iter()
                    .enumerate()
                    .map(|(j, values)| {
                        (quotient_chunks_domains[i][j], vec![(zeta, values.clone())])
                    })
                    .collect_vec()
            })
            .collect_vec();
        rounds.push((proof.commitments.quotient, quotient_domains_and_openings));

        pcs.verify(rounds, &proof.opening.proof, challenger)
            .map_err(|e| VerificationError::InvalidOpeningArgument(format!("{:?}", e)))?;

        // Verify each RAP's constraints
        for (i, (rap, data)) in raps.into_iter().zip_eq(proof.rap_data).enumerate() {
            let preprocessed_values = vk.preprocessed_data[i]
                .as_ref()
                .map(|_| {
                    index_lookup_preprocessed
                        .iter()
                        .position(|&j| j == i)
                        .map(|k| &opened_values.preprocessed[k])
                })
                .unwrap_or_default();
            let (main_commit_index, main_mat_index) = data.main_trace_ptr;
            let main_values = &opened_values.main[main_commit_index][main_mat_index];
            let main_domain = domains[data.index];
            let perm_values = data
                .perm_trace_index
                .map(|i| &opened_values.perm.as_ref().unwrap()[i]);
            let quotient_chunks = &opened_values.quotient[data.index];
            let qc_domains = &quotient_chunks_domains[data.index];
            let perm_exposed_values = proof.cumulative_sums[data.index].as_slice();
            verify_single_rap_constraints(
                rap,
                preprocessed_values,
                main_values,
                perm_values,
                quotient_chunks,
                main_domain,
                qc_domains,
                zeta,
                alpha,
                &perm_challenges,
                public_values,
                perm_exposed_values,
            )?;
        }

        let sum: SC::Challenge = proof
            .cumulative_sums
            .into_iter()
            .map(|c| c.unwrap_or(SC::Challenge::zero()))
            .sum();
        if sum != SC::Challenge::zero() {
            return Err(VerificationError::NonZeroCumulativeSum);
        }

        Ok(())
    }
}
