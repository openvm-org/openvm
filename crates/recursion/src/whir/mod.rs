use core::iter::zip;
use std::sync::Arc;

use itertools::izip;
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::Powers;
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
            whir_pow_witnesses,
        } = &proof.whir_proof;

        let _mu = ts.sample_ext();

        let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();
        debug_assert_eq!(ood_values.len(), codeword_commits.len());
        debug_assert_eq!(ood_values.len(), whir_pow_witnesses.len());
        for (i, (ood, commit, pow_witness)) in
            izip!(ood_values, codeword_commits, whir_pow_witnesses).enumerate()
        {
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

            ts.observe(*pow_witness);
            let _pow_bits = ts.sample();

            for j in 0..vk.inner.params.num_whir_queries {
                let _bits = ts.sample();
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
