use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    system::{AirModule, BusInventory, Preflight},
    transcript::dummy::DummyTranscriptAir,
};

mod dummy;

pub struct TranscriptModule {
    pub bus_inventory: BusInventory,
}

impl TranscriptModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        Self { bus_inventory }
    }
}

impl AirModule for TranscriptModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let transcript_air = DummyTranscriptAir {
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        vec![Arc::new(transcript_air)]
    }

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    ) {
    }

    fn generate_proof_inputs(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(dummy::generate_trace(proof, preflight))),
            public_values: vec![],
        }]
    }
}
