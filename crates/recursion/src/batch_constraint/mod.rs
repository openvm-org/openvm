use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    batch_constraint::{
        columns::DummyPerColumnAir, selector::BatchConstraintSelectorAir,
        sumcheck::BatchConstraintSumcheckAir,
    },
    system::{AirModule, BusInventory, Preflight},
};

mod columns;
mod selector;
mod sumcheck;

pub struct BatchConstraintModule {
    bus_inventory: BusInventory,
}

impl BatchConstraintModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        BatchConstraintModule { bus_inventory }
    }
}

impl AirModule for BatchConstraintModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        // TODO: GkrRandomnessBus
        // TODO: ColumnClaimsBus
        let sumcheck_air = BatchConstraintSumcheckAir {
            bc_module_bus: self.bus_inventory.bc_module_bus,
            stacking_module_bus: self.bus_inventory.stacking_module_bus,
            initial_zc_randomness_bus: self.bus_inventory.initial_zerocheck_randomness_bus,
            batch_constraint_randomness_bus: self.bus_inventory.constraint_randomness_bus,
        };
        let selector_air = BatchConstraintSelectorAir {
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
        };
        let per_column_air = DummyPerColumnAir {
            column_claims_bus: self.bus_inventory.column_claims_bus,
        };
        vec![
            Arc::new(sumcheck_air) as AirRef<_>,
            Arc::new(selector_air) as AirRef<_>,
            Arc::new(per_column_air) as AirRef<_>,
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
                common_main: Some(Arc::new(sumcheck::generate_trace(proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(selector::generate_trace(vk, proof))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(columns::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
        ]
    }
}
