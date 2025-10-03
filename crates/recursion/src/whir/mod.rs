use core::iter::zip;
use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_v2::{
    F,
    keygen::types::{MultiStarkVerifyingKeyV2, SystemParams},
    proof::{Proof, WhirProof},
};

use crate::{
    system::{AirModule, BusInventory, Preflight, WhirPreflight},
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

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    ) {
        let ts = &mut preflight.transcript;
        let WhirProof {
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            initial_round_opened_rows,
            initial_round_merkle_proofs,
            codeword_opened_rows,
            codeword_merkle_proofs,
        } = &proof.whir_proof;

        let _mu = ts.sample_ext();

        let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();
        debug_assert_eq!(ood_values.len(), codeword_commits.len());
        for (i, (ood, commit)) in zip(ood_values, codeword_commits).enumerate() {
            for _ in 0..vk.inner.params.k_whir {
                if let Some(evals) = sumcheck_poly_iter.next() {
                    for eval in evals {
                        ts.observe_ext(*eval);
                    }
                    let _alpha = ts.sample_ext();
                }
            }
            ts.observe_slice(commit);

            let _z0 = ts.sample_ext();
            ts.observe_ext(*ood);

            for j in 0..vk.inner.params.num_whir_queries {
                let _bits = ts.sample();
                if i == 0 {
                    for l in 0..initial_round_opened_rows.len() {
                        ts.observe_slice(&initial_round_opened_rows[l][j]);
                        for commit in &initial_round_merkle_proofs[l][j] {
                            ts.observe_slice(commit);
                        }
                    }
                } else {
                    for val in &codeword_opened_rows[i - 1][j] {
                        ts.observe_ext(*val);
                    }
                    for commit in &codeword_merkle_proofs[i - 1][j] {
                        ts.observe_slice(commit);
                    }
                }
            }
            let _gamma = ts.sample_ext();
        }
        debug_assert!(sumcheck_poly_iter.next().is_none());

        preflight.whir = WhirPreflight {}
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
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
