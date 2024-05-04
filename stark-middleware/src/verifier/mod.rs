use itertools::{izip, Itertools};
use p3_air::BaseAir;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractExtensionField;
use p3_uni_stark::{StarkGenericConfig, Val};

pub mod constraints;
mod error;
pub mod types;

pub use error::*;
use tracing::instrument;

use crate::{
    prover::types::PartitionedProof, verifier::constraints::verify_single_air_constraints,
};

use self::types::VerifierAir;

/// Verifies a partitioned proof of multi-matrix AIRs.
// TODO: Interactions
pub struct PartitionVerifier<SC: StarkGenericConfig> {
    config: SC,
}

impl<SC: StarkGenericConfig> PartitionVerifier<SC> {
    pub fn new(config: SC) -> Self {
        Self { config }
    }

    // TODO: partitioned_airs records the constraints of a partition multi-matrix AIRs.
    // It would be better to pass in only symbolic constraints, which can be serialized.
    /// Public values is a global list shared across all AIRs.
    #[instrument(name = "PartitionVerifier::verify", level = "debug", skip_all)]
    pub fn verify(
        &self,
        challenger: &mut SC::Challenger,
        partitioned_airs: Vec<Vec<&dyn VerifierAir<SC>>>,
        proof: PartitionedProof<SC>,
        public_values: &[Val<SC>],
    ) -> Result<(), VerificationError> {
        // Challenger must observe public values
        for val in public_values.iter() {
            challenger.observe(val.clone());
        }
        // Check shapes of traces and quotients
        let valid_shape = partitioned_airs
            .iter()
            .zip_eq(proof.commitments.iter())
            .zip_eq(proof.opening_proofs.iter())
            .all(|((airs, commitments), opening)| {
                // In a single part of partition:
                opening
                    .values
                    .per_air
                    .iter()
                    .zip_eq(airs.iter())
                    .zip_eq(commitments.quotient.quotient_degrees.iter())
                    .all(|((opened_values, &air), &quotient_degree)| {
                        let air_width = BaseAir::<Val<SC>>::width(air);
                        opened_values.trace.local.len() == air_width
                            && opened_values.trace.next.len() == air_width
                            && opened_values.quotient_chunks.len() == quotient_degree
                            && opened_values.quotient_chunks.iter().all(|qc| {
                                qc.len() == <SC::Challenge as AbstractExtensionField<Val<SC>>>::D
                            })
                    })
            });
        if !valid_shape {
            return Err(VerificationError::InvalidProofShape);
        }

        // Observe trace commitments
        for commitment in proof.commitments.iter() {
            challenger.observe(commitment.main_trace.commit.clone());
        }
        // Draw `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        // Observe quotient commitments
        for commitment in proof.commitments.iter() {
            challenger.observe(commitment.quotient.commit.clone());
        }
        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let pcs = self.config.pcs();
        // TODO: for now we go part by part, but this will change
        for (commitments, (openings, airs)) in proof
            .commitments
            .iter()
            .zip_eq(proof.opening_proofs.iter().zip_eq(partitioned_airs))
        {
            // Verify all opening proofs
            let (main_domains, quotient_chunks_domains): (Vec<_>, Vec<_>) = commitments
                .main_trace
                .degrees
                .iter()
                .zip_eq(commitments.quotient.quotient_degrees.iter())
                .map(|(&degree, &quotient_degree)| {
                    let domain = pcs.natural_domain_for_degree(degree);
                    let quotient_domain = domain.create_disjoint_domain(degree * quotient_degree);
                    let qc_domains = quotient_domain.split_domains(quotient_degree);
                    (domain, qc_domains)
                })
                .unzip();
            let main_domains_and_openings = main_domains
                .iter()
                .zip(openings.values.per_air.iter())
                .map(|(&domain, opened_values)| {
                    (
                        domain,
                        vec![
                            (zeta, opened_values.trace.local.clone()),
                            (
                                domain.next_point(zeta).unwrap(),
                                opened_values.trace.next.clone(),
                            ),
                        ],
                    )
                })
                .collect_vec();
            let quotient_chunks_domains_and_openings = quotient_chunks_domains
                .iter()
                .flatten()
                .zip_eq(
                    openings
                        .values
                        .per_air
                        .iter()
                        .flat_map(|opened_values| &opened_values.quotient_chunks),
                )
                .map(|(&domain, opened_values)| (domain, vec![(zeta, opened_values.clone())]))
                .collect_vec();

            pcs.verify(
                vec![
                    (
                        commitments.main_trace.commit.clone(),
                        main_domains_and_openings,
                    ),
                    (
                        commitments.quotient.commit.clone(),
                        quotient_chunks_domains_and_openings,
                    ),
                ],
                &openings.proof,
                challenger,
            )
            .map_err(|e| VerificationError::InvalidOpeningArgument(format!("{:?}", e)))?;

            for (qc_domains, opened_values, &main_domain, air) in izip!(
                quotient_chunks_domains.iter(),
                openings.values.per_air.iter(),
                main_domains.iter(),
                airs
            ) {
                verify_single_air_constraints::<SC, _>(
                    air,
                    opened_values,
                    main_domain,
                    qc_domains,
                    zeta,
                    alpha,
                    public_values,
                )?;
            }
        }

        Ok(())
    }
}
