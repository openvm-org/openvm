use std::fmt::Debug;

use itertools::Itertools;
use p3_commit::{Pcs, PolynomialSpace};
use p3_uni_stark::{Domain, StarkGenericConfig};
use serde::{Deserialize, Serialize};

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
    /// - permutation trace matrices have multiple commitments
    /// - main trace matrices can have multiple commitments
    /// - permutation trace matrices all committed together if they exist
    /// - quotient poly chunks all committed together
    pub fn open(
        &self,
        challenger: &mut SC::Challenger,
        // For each preprocessed trace commitment, the prover data and
        // the domain of each matrix, in order
        preprocessed: Vec<(&PcsProverData<SC>, Vec<Domain<SC>>)>,
        // For each main trace commitment, the prover data and
        // the domain of each matrix, in order
        main: Vec<(&PcsProverData<SC>, Vec<Domain<SC>>)>,
        // Permutation trace commitment prover data, and the domain
        // of each matrix, in order, if permutation trace exists
        perm: Option<(&PcsProverData<SC>, Vec<Domain<SC>>)>,
        // Quotient poly commitment prover data
        quotient_data: &PcsProverData<SC>,
        // Quotient degree for each RAP, flattened
        quotient_degrees: &[usize],
    ) -> OpeningProof<SC> {
        let zeta = self.zeta;

        let mut rounds = preprocessed
            .iter()
            .chain(main.iter())
            .chain(perm.iter())
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

        let perm_openings = perm.is_some().then(|| {
            let ops = opening_values
                .pop()
                .expect("Should have permutation trace opening");
            collect_trace_openings(ops)
        });

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
            .map(collect_trace_openings)
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
                perm: perm_openings,
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
pub struct OpeningProof<SC: StarkGenericConfig> {
    pub proof: PcsProof<SC>,
    pub values: OpenedValues<SC::Challenge>,
}

#[derive(Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    /// For each preprocessed trace commitment, for each matrix in commitment, the
    /// opened values
    pub preprocessed: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    /// For each main trace commitment, for each matrix in commitment, the
    /// opened values
    pub main: Vec<Vec<AdjacentOpenedValues<Challenge>>>,
    /// For each matrix in permutation trace commitment, the opened values,
    /// if permutation trace commitment exists
    pub perm: Option<Vec<AdjacentOpenedValues<Challenge>>>,
    /// For each RAP, for each quotient chunk in quotient poly, the opened values
    pub quotient: Vec<Vec<Vec<Challenge>>>,
}

#[derive(Serialize, Deserialize)]
pub struct AdjacentOpenedValues<Challenge> {
    pub local: Vec<Challenge>,
    pub next: Vec<Challenge>,
}
