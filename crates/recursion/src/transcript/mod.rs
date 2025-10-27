use std::sync::Arc;

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{FiatShamirTranscript, TranscriptHistory},
    proof::Proof,
};

use crate::{
    system::{AirModule, BusInventory, Preflight},
    transcript::poseidon2::{Poseidon2Air, Poseidon2Cols},
    transcript::transcript::{TranscriptAir, TranscriptCols},
};

pub mod poseidon2;
pub mod transcript;

pub struct TranscriptModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    pub bus_inventory: BusInventory,
}

impl TranscriptModule {
    pub fn new(mvk: Arc<MultiStarkVerifyingKeyV2>, bus_inventory: BusInventory) -> Self {
        Self { mvk, bus_inventory }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for TranscriptModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let transcript_air = TranscriptAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            poseidon2_bus: self.bus_inventory.poseidon2_bus,
        };
        let poseidon2_air = Poseidon2Air {
            poseidon2_bus: self.bus_inventory.poseidon2_bus,
        };
        vec![Arc::new(transcript_air), Arc::new(poseidon2_air)]
    }

    fn run_preflight(&self, _proof: &Proof, _preflight: &mut Preflight, _ts: &mut TS) {}

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        // generate transcript first, and then we know what's the poseidon2 lookup that are needed
        let transcript_trace = transcript::generate_trace(proof, preflight);
        let transcript_width = TranscriptCols::<F>::width();

        let mut poseidon_trace = vec![];
        let mut poseidon_input: Option<Vec<F>> = None;
        for (i, row) in transcript_trace.chunks(transcript_width).enumerate() {
            let poseidon_state = row[transcript_width - POSEIDON2_WIDTH..].to_vec();
            if let Some(poseidon_prev) = poseidon_input {
                poseidon_trace.extend_from_slice(&poseidon_prev);
                poseidon_trace.extend_from_slice(&poseidon_state);
                poseidon_trace.push(row[transcript_width - POSEIDON2_WIDTH - 1]); // permuted
            }
            poseidon_input = Some(poseidon_state);
        }
        let poseidon2_width = Poseidon2Cols::<F, 0>::width();

        let transcript_input = AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                transcript_trace,
                transcript_width,
            ))),
            public_values: vec![],
        };
        let poseidon2_input = AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                poseidon_trace,
                poseidon2_width,
            ))),
            public_values: vec![],
        };
        vec![transcript_input, poseidon2_input]
    }
}
