use core::iter::zip;
use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, Proof},
};

use crate::{
    batch_constraint::dummy::BatchConstraintDummyAir,
    system::{AirModule, BatchConstraintPreflight, BusInventory, Preflight},
};

mod dummy;

pub struct BatchConstraintModule {
    bus_inventory: BusInventory,
}

impl BatchConstraintModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        BatchConstraintModule { bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for BatchConstraintModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let sumcheck_air = BatchConstraintDummyAir {
            bc_module_bus: self.bus_inventory.bc_module_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            xi_randomness_bus: self.bus_inventory.xi_randomness_bus,
            batch_constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        vec![Arc::new(sumcheck_air) as AirRef<_>]
    }

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight<TS>,
    ) {
        let ts = &mut preflight.transcript;
        let BatchConstraintProof {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        } = &proof.batch_constraint_proof;

        let mut sumcheck_rnd = vec![];

        // Constraint batching
        let _lambda = ts.sample_ext();

        for (sum_claim_p, sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
            ts.observe_ext(*sum_claim_p);
            ts.observe_ext(*sum_claim_q);
        }
        let _mu = ts.sample_ext();

        // univariate round
        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let r0 = ts.sample_ext();
        sumcheck_rnd.push(r0);

        for polys in sumcheck_round_polys {
            for eval in polys {
                ts.observe_ext(*eval);
            }
            let ri = ts.sample_ext();
            sumcheck_rnd.push(ri);
        }

        // Common main
        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &vk.inner.per_air[*air_id].params.width;

            for col_idx in 0..width.common_main {
                let (col_opening, rot_opening) = column_openings[sort_idx][0][col_idx];
                ts.observe_ext(col_opening);
                ts.observe_ext(rot_opening);
            }
        }

        for (sort_idx, (air_id, _)) in preflight.proof_shape.sorted_trace_vdata.iter().enumerate() {
            let width = &vk.inner.per_air[*air_id].params.width;
            let widths = width.preprocessed.iter().chain(width.cached_mains.iter());

            for (i, w) in widths.enumerate() {
                for col_idx in 0..*w {
                    let (col_opening, rot_opening) = column_openings[sort_idx][i + 1][col_idx];
                    ts.observe_ext(col_opening);
                    ts.observe_ext(rot_opening);
                }
            }
        }

        preflight.batch_constraint = BatchConstraintPreflight {
            post_tidx: ts.len(),
            sumcheck_rnd,
        }
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(dummy::generate_trace(vk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
