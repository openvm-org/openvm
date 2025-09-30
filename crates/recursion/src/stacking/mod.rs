use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    stacking::{
        per_air_part::DummyPerAirPartAir, per_column::DummyPerColumnAir,
        sumcheck::StackingSumcheckAir,
    },
    system::{AirModule, BusInventory, Preflight},
};

mod per_air_part;
mod per_column;
mod sumcheck;

pub struct StackingModule {
    bus_inventory: BusInventory,
}

impl StackingModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        StackingModule { bus_inventory }
    }
}

impl AirModule for StackingModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        // TODO: stacking widths bus
        let stacking_air = StackingSumcheckAir {
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            whir_module_bus: self.bus_inventory.whir_module_bus,
            batch_constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
        };
        let selector_air = DummyPerAirPartAir {
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            stacking_widths_bus: self.bus_inventory.stacking_widths_bus,
        };
        let columns_air = DummyPerColumnAir {
            column_claims_bus: self.bus_inventory.column_claims_bus,
        };
        vec![
            Arc::new(stacking_air),
            Arc::new(selector_air),
            Arc::new(columns_air),
        ]
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
                common_main: Some(Arc::new(sumcheck::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(per_air_part::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(per_column::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
        ]
    }
}
