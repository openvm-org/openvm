use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{Proof, WhirProof},
};

use crate::{
    system::{AirModule, BusInventory, Preflight, WhirPreflight},
    whir::dummy::DummyWhirAir,
};

mod dummy;

pub struct WhirModule {
    bus_inventory: BusInventory,
}

impl WhirModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        WhirModule { bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for WhirModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let whir_air = DummyWhirAir {
            whir_module_bus: self.bus_inventory.whir_module_bus,
            stacking_widths_bus: self.bus_inventory.stacking_widths_bus,
            stacking_claims_bus: self.bus_inventory.stacking_claims_bus,
            stacking_commitments_bus: self.bus_inventory.stacking_commitments_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        vec![Arc::new(whir_air)]
    }

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight<TS>,
    ) {
        let ts = &mut preflight.transcript;
        let WhirProof {
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            initial_round_opened_rows: _,
            initial_round_merkle_proofs: _,
            codeword_opened_rows: _,
            codeword_merkle_proofs: _,
            whir_pow_witnesses,
            final_poly,
        } = &proof.whir_proof;

        let _mu = ts.sample_ext();

        let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();

        let num_whir_rounds = whir_pow_witnesses.len();

        debug_assert_eq!(ood_values.len(), num_whir_rounds - 1);
        debug_assert_eq!(codeword_commits.len(), num_whir_rounds - 1);

        for i in 0..num_whir_rounds {
            for _ in 0..vk.inner.params.k_whir {
                if let Some(evals) = sumcheck_poly_iter.next() {
                    for eval in evals {
                        ts.observe_ext(*eval);
                    }
                    let _alpha = ts.sample_ext();
                }
            }
            if i != num_whir_rounds - 1 {
                ts.observe_slice(&codeword_commits[i]);
                let _z0 = ts.sample_ext();
                ts.observe_ext(ood_values[i]);
            } else {
                for coeff in final_poly {
                    ts.observe_ext(*coeff);
                }
            }

            ts.observe(whir_pow_witnesses[i]);
            let _pow_bits = ts.sample();

            for _ in 0..vk.inner.params.num_whir_queries {
                let _bits = ts.sample();
            }
            if i != num_whir_rounds - 1 {
                let _gamma = ts.sample_ext();
            }
        }
        debug_assert!(sumcheck_poly_iter.next().is_none());

        preflight.whir = WhirPreflight {}
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(dummy::generate_trace(vk, proof, preflight))),
            public_values: vec![],
        }]
    }
}
