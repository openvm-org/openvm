use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{Proof, StackingProof},
};

use crate::{
    stacking::dummy::DummyStackingAir,
    system::{AirModule, BusInventory, Preflight, StackingPreflight},
};

mod dummy;

pub struct StackingModule {
    bus_inventory: BusInventory,
}

impl StackingModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        StackingModule { bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for StackingModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let dummy_air = DummyStackingAir {
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            batch_constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            column_claims_bus: self.bus_inventory.column_claims_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            stacking_widths_bus: self.bus_inventory.stacking_widths_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        vec![Arc::new(dummy_air)]
    }

    fn run_preflight(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight<TS>,
    ) {
        let mut sumcheck_rnd = vec![];

        let ts = &mut preflight.transcript;
        let StackingProof {
            univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings,
        } = &proof.stacking_proof;

        let _mu = ts.sample_ext();

        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let u0 = ts.sample_ext();
        sumcheck_rnd.push(u0);

        for poly in sumcheck_round_polys {
            for eval in poly {
                ts.observe_ext(*eval);
            }
            let ui = ts.sample_ext();
            sumcheck_rnd.push(ui);
        }

        for matrix_openings in stacking_openings {
            for col_opening in matrix_openings {
                ts.observe_ext(*col_opening);
            }
        }

        let stacking_batching_challenge = ts.sample_ext();

        preflight.stacking = StackingPreflight {
            post_tidx: ts.len(),
            stacking_batching_challenge,
            sumcheck_rnd,
        };
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
