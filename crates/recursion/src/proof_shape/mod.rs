use core::cmp::{Reverse, max};
use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::Proof,
};

use crate::{
    proof_shape::dummy::DummyProofShapeAir,
    system::{AirModule, BusInventory, Preflight, ProofShapePreflight},
};

mod dummy;

pub struct ProofShapeModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    bus_inventory: BusInventory,
}

impl ProofShapeModule {
    pub fn new(mvk: Arc<MultiStarkVerifyingKeyV2>, bus_inventory: BusInventory) -> Self {
        Self { mvk, bus_inventory }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for ProofShapeModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let proof_shape_air = DummyProofShapeAir {
            gkr_bus: self.bus_inventory.gkr_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            air_shape_bus: self.bus_inventory.air_shape_bus,
            air_part_shape_bus: self.bus_inventory.air_part_shape_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            _public_values_bus: self.bus_inventory.public_values_bus,
        };
        vec![Arc::new(proof_shape_air) as AirRef<_>]
    }

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS) {
        ts.observe_commit(self.mvk.pre_hash);
        ts.observe_commit(proof.common_main_commit);

        let vk = &self.mvk.inner;

        let mut num_common_main_cells = 0;

        for (trace_vdata, avk, pvs) in izip!(&proof.trace_vdata, &vk.per_air, &proof.public_values)
        {
            let is_air_present = trace_vdata.is_some();

            if !avk.is_required {
                ts.observe(F::from_bool(is_air_present));
            }
            if let Some(trace_vdata) = trace_vdata {
                num_common_main_cells += (1 << (vk.params.l_skip + trace_vdata.hypercube_dim))
                    * avk.params.width.common_main;

                if let Some(pdata) = avk.preprocessed_data.as_ref() {
                    ts.observe_commit(pdata.commit);
                } else {
                    ts.observe(F::from_canonical_usize(trace_vdata.hypercube_dim));
                }
                debug_assert_eq!(
                    avk.params.width.cached_mains.len(),
                    trace_vdata.cached_commitments.len()
                );
                for commit in &trace_vdata.cached_commitments {
                    ts.observe_commit(*commit);
                }
                debug_assert_eq!(avk.params.num_public_values, pvs.len());
            }
            for pv in pvs {
                ts.observe(*pv);
            }
        }

        let mut sorted_trace_vdata: Vec<_> = proof
            .trace_vdata
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(air_id, data)| data.map(|data| (air_id, data)))
            .collect();
        sorted_trace_vdata.sort_by_key(|(_, data)| Reverse(data.hypercube_dim));

        let n_max = proof
            .trace_vdata
            .iter()
            .flat_map(|datum| datum.as_ref().map(|datum| datum.hypercube_dim))
            .max()
            .unwrap();
        let num_layers = proof.gkr_proof.claims_per_layer.len();
        let l_skip = vk.params.l_skip;
        let n_logup = num_layers.saturating_sub(l_skip);

        let stack_height = 1 << (vk.params.l_skip + vk.params.n_stack);
        let stacked_common_width = num_common_main_cells.div_ceil(stack_height);

        let logup_pow_bits = vk.params.logup_pow_bits;

        preflight.proof_shape = ProofShapePreflight {
            stacked_common_width,
            sorted_trace_vdata,
            n_global: max(n_max, n_logup),
            n_max,
            n_logup,
            l_skip,
            logup_pow_bits,
            post_tidx: ts.len(),
        };
    }

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(dummy::generate_trace(&self.mvk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
