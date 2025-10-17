use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};

use crate::{
    system::{AirModule, BusInventory, Preflight},
    transcript::dummy::DummyTranscriptAir,
};

mod dummy;

pub struct TranscriptModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    pub bus_inventory: BusInventory,
}

impl TranscriptModule {
    pub fn new(mvk: Arc<MultiStarkVerifyingKeyV2>, bus_inventory: BusInventory) -> Self {
        Self { mvk, bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for TranscriptModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let transcript_air = DummyTranscriptAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
        };
        vec![Arc::new(transcript_air)]
    }

    fn run_preflight(&self, _proof: &Proof, _preflight: &mut Preflight<TS>) {}

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(dummy::generate_trace(&self.mvk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
