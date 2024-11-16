use std::fmt::Debug;

use derivative::Derivative;
use itertools::Itertools;
use p3_commit::{Pcs, PolynomialSpace};
use p3_uni_stark::{Domain, StarkGenericConfig};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::config::{PcsProof, PcsProverData};

pub struct OpeningProver<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
    zeta: SC::Challenge,
}

impl<'pcs, SC: StarkGenericConfig> OpeningProver<'pcs, SC> {
    pub fn new(pcs: &'pcs SC::Pcs, zeta: SC::Challenge) -> Self {
        Self { pcs, zeta }
    }

    /// Opening proof for multiple RAP matrices, where
    /// - (for now) each preprocessed trace matrix has a separate commitment
    /// - main trace matrices can have multiple commitments
    /// - for each after_challenge phase, all matrices in the phase share a commitment
    /// - quotient poly chunks are all committed together
    #[instrument(name = "FRI opening proofs", skip_all)]
    pub fn open(
        &self,
        challenger: &mut SC::Challenger,
        // For each preprocessed trace commitment, the prover data and
        // the domain of the matrix, in order
        preprocessed: Vec<(&PcsProverData<SC>, Domain<SC>)>,
        // For each main trace commitment, the prover data and
        // the domain of each matrix, in order
        main: Vec<(&PcsProverData<SC>, Vec<Domain<SC>>)>,
        // after_challenge[i] has shared commitment prover data for all matrices in that phase, and domains of those matrices, in order
        after_challenge: Vec<(&PcsProverData<SC>, Vec<Domain<SC>>)>,
        // Quotient poly commitment prover data
        quotient_data: &PcsProverData<SC>,
        // Quotient degree for each RAP committed in quotient_data, in order
        quotient_degrees: &[usize],
    ) -> OpeningProof<SC> {
        let preprocessed: Vec<_> = preprocessed
            .into_iter()
            .map(|(data, domain)| (data, vec![domain]))
            .collect();

        let zeta = self.zeta;
        let mut rounds = preprocessed
            .iter()
            .chain(main.iter())
            .chain(after_challenge.iter())
            .map(|(data, domains)| {
                let points_per_mat = domains
                    .iter()
                    .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                    .collect_vec();
                (*data, points_per_mat)
            })
            .collect_vec();

        // open every quotient chunk at zeta
        let num_chunks: usize = quotient_degrees.iter().sum();
        let quotient_opening_points = vec![vec![zeta]; num_chunks];
        rounds.push((quotient_data, quotient_opening_points));

        let (mut opening_values, opening_proof) = self.pcs.open(rounds, challenger);

        // Unflatten opening_values
        let mut quotient_openings = opening_values.pop().expect("Should have quotient opening");

        let num_after_challenge = after_challenge.len();
        let after_challenge_openings = opening_values
            .split_off(opening_values.len() - num_after_challenge)
            .into_iter()
            .map(collect_trace_openings)
            .collect_vec();
        assert_eq!(
            after_challenge_openings.len(),
            num_after_challenge,
            "Incorrect number of after challenge trace openings"
        );

        let main_openings = opening_values
            .split_off(preprocessed.len())
            .into_iter()
            .map(collect_trace_openings)
            .collect_vec();
        assert_eq!(
            main_openings.len(),
            main.len(),
            "Incorrect number of main trace openings"
        );

        let preprocessed_openings = opening_values
            .into_iter()
            .map(|values| {
                let mut openings = collect_trace_openings(values);
                openings
                    .pop()
                    .expect("Preprocessed trace should be opened at 1 point")
            })
            .collect_vec();
        assert_eq!(
            preprocessed_openings.len(),
            preprocessed.len(),
            "Incorrect number of preprocessed trace openings"
        );

        // Unflatten quotient openings
        let quotient_openings = quotient_degrees
            .iter()
            .map(|&chunk_size| {
                quotient_openings
                    .drain(..chunk_size)
                    .map(|mut op| {
                        op.pop()
                            .expect("quotient chunk should be opened at 1 point")
                    })
                    .collect_vec()
            })
            .collect_vec();

        OpeningProof {
            proof: opening_proof,
            values: OpenedValues {
                preprocessed: preprocessed_openings,
                main: main_openings,
                after_challenge: after_challenge_openings,
                quotient: quotient_openings,
            },
        }
    }
}

fn collect_trace_openings<Challenge: Debug>(
    ops: Vec<Vec<Vec<Challenge>>>,
) -> Vec<AdjacentOpenedValues<Challenge>> {
    ops.into_iter()
        .map(|op| {
            let [local, next] = op.try_into().expect("Should have 2 openings");
            AdjacentOpenedValues { local, next }
        })
        .collect()
}

/// PCS opening proof with opened values for multi-matrix AIR.
#[derive(Serialize, Deserialize, Derivative)]
#[serde(bound = "")]
#[derivative(Clone(bound = "SC::Challenge: Clone"))]
pub struct OpeningProof<SC: StarkGenericConfig> {
    pub proof: PcsProof<SC>,
    pub values: OpenedValues<SC::Challenge>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    /// For each preprocessed trace commitment, the opened values
    pub preprocessed: Vec<AdjacentOpenedValues<Challenge>>,
    /// For each main trace commitment, for each matrix in commitment, the
    /// opened values
    pub main: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    /// For each phase after challenge, there is shared commitment.
    /// For each commitment, if any, for each matrix in the commitment, the opened values,
    pub after_challenge: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    /// For each RAP, for each quotient chunk in quotient poly, the opened values
    pub quotient: Vec<Vec<Vec<Challenge>>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AdjacentOpenedValues<Challenge> {
    pub local: Vec<Challenge>,
    pub next: Vec<Challenge>,
}
