use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::symbolic::get_log_quotient_degree,
    config::{Com, PcsProof, PcsProverData},
};

use self::{
    opener::OpeningProver,
    quotient::QuotientCommitter,
    types::{Commitments, PartitionedProof, ProvenDataBeforeOpening, ProvenMultiMatrixAirTrace},
};

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;
/// Trace commitment computation
pub mod trace;
pub mod types;

/// Proves a partition of multi-matrix AIRs.
// TODO: Interactions between parts in partition.
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
    ) -> PartitionedProof<SC>
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

        // Challenger must observe all trace commitments
        for part in partition.iter() {
            challenger.observe(part.trace_data.commit.clone());
        }

        // TODO: permutation trace

        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let quotient_committer = QuotientCommitter::new(pcs, alpha);
        let quotient_data = partition
            .par_iter()
            .map(|part| {
                let quotient_degrees = part
                    .airs
                    .iter()
                    .map(|&air| {
                        let d = get_log_quotient_degree::<Val<SC>, _>(air, 0, public_values.len());
                        1 << d
                    })
                    .collect_vec();
                let qv = quotient_committer.compute_quotient_values(
                    part.clone(),
                    quotient_degrees,
                    public_values,
                );
                quotient_committer.commit(qv)
            })
            .collect::<Vec<_>>();

        // Observe all quotient commitments
        for q in quotient_data.iter() {
            challenger.observe(q.commit.clone());
        }

        let proven_partitions = partition
            .into_iter()
            .zip_eq(quotient_data.iter())
            .map(|(part, quotient)| ProvenDataBeforeOpening {
                trace: part.trace_data,
                quotient,
            })
            .collect::<Vec<_>>();
        let commitments = proven_partitions
            .iter()
            .map(|part| Commitments {
                main_trace: part.trace.verifier_view(),
                quotient: part.quotient.verifier_view(),
            })
            .collect::<Vec<_>>();

        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let opener = OpeningProver::new(pcs, zeta);
        let opening_proofs = proven_partitions
            .into_par_iter()
            .map(|part| opener.open(challenger, part))
            .collect::<Vec<_>>();

        PartitionedProof {
            commitments,
            opening_proofs,
        }
    }
}
