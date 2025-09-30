use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    proof_shape::circuit::DummyProofShapeAir,
    system::{AirModule, BusInventory, Preflight},
};

mod circuit;

pub struct ProofShapeModule {
    bus_inventory: BusInventory,
}

impl ProofShapeModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        Self { bus_inventory }
    }
}

impl AirModule for ProofShapeModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let proof_shape_air = DummyProofShapeAir {
            gkr_bus: self.bus_inventory.gkr_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            stacking_commitments_bus: self.bus_inventory.stacking_commitments_bus,
            stacking_widths_bus: self.bus_inventory.stacking_widths_bus,
            public_values_bus: self.bus_inventory.public_values_bus,
        };
        vec![Arc::new(proof_shape_air) as AirRef<_>]
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        _public_values_per_air: &[Vec<F>],
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(circuit::generate_trace(vk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
