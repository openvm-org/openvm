use core::cmp::Reverse;
use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Config, dummy_airs::fib_air::trace,
};
use p3_field::FieldAlgebra;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    proof_shape::circuit::DummyProofShapeAir,
    system::{AirModule, BusInventory, Preflight, ProofShapePreflight},
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

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    ) {
        let ts = &mut preflight.transcript;
        ts.observe_commit(vk.pre_hash);
        ts.observe_commit(proof.common_main_commit);

        let vk = &vk.inner;

        let mut num_common_main_cells = 0;

        for (trace_shape, avk, pvs) in izip!(&proof.trace_shapes, &vk.per_air, &proof.public_values)
        {
            let is_air_present = trace_shape.is_some();

            if !avk.is_required {
                ts.observe(F::from_bool(is_air_present));
            }
            if !is_air_present {
                continue;
            }

            let trace_shape = trace_shape.as_ref().unwrap();
            num_common_main_cells += (1 << (vk.params.l_skip + trace_shape.hypercube_dim))
                * avk.params.width.common_main;

            if let Some(pdata) = avk.preprocessed_data.as_ref() {
                ts.observe_commit(pdata.commit);
            } else {
                ts.observe(F::from_canonical_usize(trace_shape.hypercube_dim));
            }
            debug_assert_eq!(
                avk.params.width.cached_mains.len(),
                trace_shape.cached_commitments.len()
            );
            for commit in &trace_shape.cached_commitments {
                ts.observe_slice(commit);
            }
            debug_assert_eq!(avk.params.num_public_values, pvs.len());
            for pv in pvs {
                ts.observe(*pv);
            }
        }

        let mut sorted_trace_shapes: Vec<_> = proof
            .trace_shapes
            .iter()
            .cloned()
            .enumerate()
            .flat_map(|(air_id, shape)| shape.map(|shape| (air_id, shape)))
            .collect();
        sorted_trace_shapes.sort_by_key(|(_, shape)| Reverse(shape.hypercube_dim));

        let n_max = proof
            .trace_shapes
            .iter()
            .flat_map(|shape| shape.as_ref().map(|shape| shape.hypercube_dim))
            .max()
            .unwrap();
        let n_logup = proof.gkr_proof.claims_per_layer.len(); // n_logup = num_layers

        let stack_height = 1 << (vk.params.l_skip + vk.params.n_stack);
        let stacked_common_width = (num_common_main_cells + stack_height - 1) / stack_height;

        preflight.proof_shape = ProofShapePreflight {
            stacked_common_width,
            sorted_trace_shapes,
            n_max,
            n_logup,
            post_tidx: ts.len(),
        };
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(circuit::generate_trace(vk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
