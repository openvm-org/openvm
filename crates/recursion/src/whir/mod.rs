use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    system::{AirModule, BusInventory, Preflight},
    whir::{circuit::WhirAir, sumcheck::WhirSumcheckAir},
};

mod circuit;
mod sumcheck;

pub struct WhirModule {
    bus_inventory: BusInventory,
}

impl WhirModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        WhirModule { bus_inventory }
    }
}

impl AirModule for WhirModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let whir_air = WhirAir {
            stacking_widths_bus: self.bus_inventory.stacking_widths_bus,
            stacking_claims_bus: self.bus_inventory.stacking_claims_bus,
            stacking_commitments_bus: self.bus_inventory.stacking_commitments_bus,
        };
        let whir_sumcheck_air = WhirSumcheckAir {
            whir_module_bus: self.bus_inventory.whir_module_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
        };
        vec![Arc::new(whir_air), Arc::new(whir_sumcheck_air)]
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        _public_values_per_air: &[Vec<F>],
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(circuit::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
        ]
    }
}
