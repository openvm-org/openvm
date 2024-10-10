use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField};
use p3_uni_stark::{Domain, StarkGenericConfig};
use tracing::instrument;

pub mod constraints;
mod error;
pub mod v2;

pub use error::*;

use self::constraints::verify_single_rap_constraints;
use crate::{
    keygen::types::MultiStarkVerifyingKey,
    prover::{opener::AdjacentOpenedValues, types::Proof},
};

/// Verifies a partitioned proof of multi-matrix AIRs.
pub struct MultiTraceStarkVerifier<'c, SC: StarkGenericConfig> {
    config: &'c SC,
}

impl<'c, SC: StarkGenericConfig> MultiTraceStarkVerifier<'c, SC> {
    pub fn new(config: &'c SC) -> Self {
        Self { config }
    }

    /// Verify collection of InteractiveAIRs and check the permutation
    /// cumulative sum is equal to zero across all AIRs.
    #[instrument(name = "MultiTraceStarkVerifier::verify", level = "debug", skip_all)]
    pub fn verify(
        &self,
        challenger: &mut SC::Challenger,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        let cumulative_sums = proof
            .exposed_values_after_challenge
            .iter()
            .map(|exposed_values| {
                assert!(
                    exposed_values.len() <= 1,
                    "Verifier does not support more than 1 challenge phase"
                );
                exposed_values.first().map(|values| {
                    assert_eq!(
                        values.len(),
                        1,
                        "Only exposed value should be cumulative sum"
                    );
                    values[0]
                })
            })
            .collect_vec();

        self.verify_raps(challenger, vk, proof)?;

        // Check cumulative sum
        let sum: SC::Challenge = cumulative_sums
            .into_iter()
            .map(|c| c.unwrap_or(SC::Challenge::zero()))
            .sum();
        if sum != SC::Challenge::zero() {
            return Err(VerificationError::NonZeroCumulativeSum);
        }
        Ok(())
    }

    // It would be better to pass in only symbolic constraints, which can be serialized.
    /// Verify general RAPs without checking any relations (e.g., cumulative sum) between exposed values of different RAPs.
    ///
    /// Public values is a global list shared across all AIRs.
    ///
    /// - `num_challenges_to_sample[i]` is the number of challenges to sample in the trace challenge phase corresponding to `proof.commitments.after_challenge[i]`. This must have length equal
    /// to `proof.commitments.after_challenge`.
    #[instrument(level = "debug", skip_all)]
    pub fn verify_raps(
        &self,
        challenger: &mut SC::Challenger,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
        // public_values: &[Vec<Val<SC>>],
    ) -> Result<(), VerificationError> {
        let public_values = &proof.public_values;
        // Challenger must observe public values
        for pis in public_values {
            challenger.observe_slice(pis);
        }

        // TODO: valid shape check from verifying key

        for preprocessed_commit in vk.per_air.iter().filter_map(|vk| {
            vk.preprocessed_data
                .as_ref()
                .map(|data| data.commit.clone())
        }) {
            challenger.observe(preprocessed_commit);
        }

        // Observe main trace commitments
        challenger.observe_slice(&proof.commitments.main_trace);

        let mut challenges = Vec::new();
        for (phase_idx, (&num_to_sample, commit)) in vk
            .num_challenges_to_sample
            .iter()
            .zip_eq(&proof.commitments.after_challenge)
            .enumerate()
        {
            // Sample challenges needed in this phase
            challenges.push(
                (0..num_to_sample)
                    .map(|_| challenger.sample_ext_element::<SC::Challenge>())
                    .collect_vec(),
            );
            // For each RAP, the exposed values in current phase
            for exposed_values in &proof.exposed_values_after_challenge {
                if let Some(values) = exposed_values.get(phase_idx) {
                    // Observe exposed values (in ext field)
                    for value in values {
                        challenger.observe_slice(value.as_base_slice());
                    }
                }
            }
            // Observe single commitment to all trace matrices in this phase
            challenger.observe(commit.clone());
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
        // Build domains
        let (domains, quotient_chunks_domains): (Vec<_>, Vec<Vec<_>>) = vk
            .per_air
            .iter()
            .zip_eq(&proof.degrees)
            .map(|(vk, degree)| {
                let quotient_degree = vk.quotient_degree;
                let domain = pcs.natural_domain_for_degree(*degree);
                let quotient_domain = domain.create_disjoint_domain(degree * quotient_degree);
                let qc_domains = quotient_domain.split_domains(quotient_degree);
                (domain, qc_domains)
            })
            .unzip();
        // Verify all opening proofs
        let opened_values = &proof.opening.values;
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
        // Build the opening rounds
        // 1. First the preprocessed trace openings
        let mut rounds: Vec<_> = domains
            .iter()
            .zip_eq(&vk.per_air)
            .flat_map(|(domain, vk)| {
                vk.preprocessed_data
                    .as_ref()
                    .map(|data| (data.commit.clone(), *domain))
            }) // Assumption: each AIR with preprocessed trace has its own commitment and opening values
            .zip_eq(&opened_values.preprocessed)
            .map(|((commit, domain), values)| {
                let domain_and_openings = trace_domain_and_openings(domain, zeta, values);
                (commit, vec![domain_and_openings])
            })
            .collect();
        // 2. Then the main trace openings
        opened_values
            .main
            .iter()
            .zip_eq(&proof.commitments.main_trace)
            .enumerate()
            .for_each(|(commit_idx, (values_per_mat, commit))| {
                let domains_and_openings = values_per_mat
                    .iter()
                    .enumerate()
                    .map(|(matrix_idx, values)| {
                        let air_idx =
                            vk.main_commit_to_air_graph.commit_to_air_index[commit_idx][matrix_idx];
                        let domain = domains[air_idx];
                        trace_domain_and_openings(domain, zeta, values)
                    })
                    .collect_vec();
                rounds.push((commit.clone(), domains_and_openings));
            });
        // 3. Then after_challenge trace openings, one phase at a time
        opened_values
            .after_challenge
            .iter()
            .zip_eq(&proof.commitments.after_challenge)
            .enumerate()
            .for_each(|(phase_idx, (values_per_mat, commit))| {
                // Filter RAPs by those that have non-empty trace matrix in this phase
                let domains = vk.per_air.iter().enumerate().flat_map(|(air_idx, vk)| {
                    (*vk.width().after_challenge.get(phase_idx).unwrap_or(&0) > 0)
                        .then(|| domains[air_idx])
                });
                let domains_and_openings = domains
                    .zip_eq(values_per_mat)
                    .map(|(domain, values)| trace_domain_and_openings(domain, zeta, values))
                    .collect_vec();
                rounds.push((commit.clone(), domains_and_openings));
            });
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
        rounds.push((
            proof.commitments.quotient.clone(),
            quotient_domains_and_openings,
        ));

        pcs.verify(rounds, &proof.opening.proof, challenger)
            .map_err(|e| VerificationError::InvalidOpeningArgument(format!("{:?}", e)))?;

        let mut preprocessed_idx = 0usize; // preprocessed commit idx
        let mut after_challenge_idx = vec![0usize; vk.num_challenges_to_sample.len()];

        // Verify each RAP's constraints
        for (domain, qc_domains, quotient_chunks, vk, public_values, exposed_values) in izip!(
            domains,
            quotient_chunks_domains,
            &opened_values.quotient,
            &vk.per_air,
            public_values,
            &proof.exposed_values_after_challenge
        ) {
            let preprocessed_values = vk.preprocessed_data.as_ref().map(|_| {
                let values = &opened_values.preprocessed[preprocessed_idx];
                preprocessed_idx += 1;
                values
            });
            let partitioned_main_values = vk
                .main_graph
                .matrix_ptrs
                .iter()
                .map(|ptr| &opened_values.main[ptr.commit_index][ptr.matrix_index])
                .collect_vec();
            // loop through challenge phases of this single RAP
            let after_challenge_values = (0..vk.width().after_challenge.len())
                .map(|phase_idx| {
                    let matrix_idx = after_challenge_idx[phase_idx];
                    after_challenge_idx[phase_idx] += 1;
                    &opened_values.after_challenge[phase_idx][matrix_idx]
                })
                .collect_vec();
            verify_single_rap_constraints::<SC>(
                &vk.symbolic_constraints.constraints,
                preprocessed_values,
                partitioned_main_values,
                after_challenge_values,
                quotient_chunks,
                domain,
                &qc_domains,
                zeta,
                alpha,
                &challenges,
                public_values,
                exposed_values,
            )?;
        }

        Ok(())
    }
}
