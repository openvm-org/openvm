use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    gkr::{gkr_round::DummyGkrRoundAir, sumcheck::DummyGkrSumcheckAir},
    system::{AirModule, BusInventory, Preflight},
};

mod gkr_round;
mod sumcheck;

pub struct GkrModule {
    bus_inventory: BusInventory,
}

impl GkrModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        GkrModule { bus_inventory }
    }
}

impl AirModule for GkrModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let gkr_verify_air = DummyGkrRoundAir {
            gkr_bus: self.bus_inventory.gkr_module_bus,
            bc_module_bus: self.bus_inventory.bc_module_bus,
            initial_zc_rnd_bus: self.bus_inventory.initial_zerocheck_randomness_bus,
            gkr_randomness_bus: self.bus_inventory.gkr_randomness_bus,
        };
        let gkr_sumcheck_air = DummyGkrSumcheckAir {
            gkr_randomness_bus: self.bus_inventory.gkr_randomness_bus,
        };
        vec![
            Arc::new(gkr_verify_air) as AirRef<_>,
            Arc::new(gkr_sumcheck_air) as AirRef<_>,
        ]
    }

    fn generate_proof_inputs(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        _public_values_per_air: &[Vec<F>],
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(gkr_round::generate_trace(proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(proof))),
                public_values: vec![],
            },
        ]
    }
}
