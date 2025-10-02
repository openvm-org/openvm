use core::iter::zip;
use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    proof::{GkrProof, Proof},
};

use crate::{
    gkr::{gkr_round::DummyGkrRoundAir, sumcheck::DummyGkrSumcheckAir},
    system::{AirModule, BusInventory, GkrPreflight, Preflight},
};

mod gkr_round;
mod sumcheck;

pub struct GkrModule {
    bus_inventory: BusInventory,
}

impl GkrModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        GkrModule { bus_inventory }
    }
}

impl AirModule for GkrModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let gkr_verify_air = DummyGkrRoundAir {
            gkr_bus: self.bus_inventory.gkr_module_bus,
            bc_module_bus: self.bus_inventory.bc_module_bus,
            initial_zc_rnd_bus: self.bus_inventory.initial_zerocheck_randomness_bus,
            gkr_randomness_bus: self.bus_inventory.gkr_randomness_bus,
        };
        let gkr_sumcheck_air = DummyGkrSumcheckAir {
            gkr_randomness_bus: self.bus_inventory.gkr_randomness_bus,
        };
        vec![
            Arc::new(gkr_verify_air) as AirRef<_>,
            Arc::new(gkr_sumcheck_air) as AirRef<_>,
        ]
    }

    fn run_preflight(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight,
    ) {
        let GkrProof {
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
        } = &proof.gkr_proof;

        let ts = &mut preflight.transcript;

        let _alpha_logup = ts.sample_ext();
        let _beta_logup = ts.sample_ext();

        if !sumcheck_polys.is_empty() {
            ts.observe_ext(*q0_claim);
        }

        for (polys, claims) in zip(sumcheck_polys, claims_per_layer) {
            let _ = ts.sample_ext();

            for poly in polys {
                for eval in poly {
                    ts.observe_ext(*eval);
                }
                let _xi_i = ts.sample_ext();
            }

            ts.observe_ext(claims.p_xi_0);
            ts.observe_ext(claims.q_xi_0);
            ts.observe_ext(claims.p_xi_1);
            ts.observe_ext(claims.q_xi_1);

            let _rho = ts.sample_ext();
        }

        for _ in preflight.proof_shape.n_logup..preflight.proof_shape.n_max + 1 {
            let _ = ts.sample_ext();
        }

        preflight.gkr = GkrPreflight {
            post_tidx: ts.len(),
            input_layer_numerator_claim: EF::ZERO, // FIXME
            input_layer_denominator_claim: EF::ZERO,
        };
    }

    fn generate_proof_inputs(
        &self,
        _vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(gkr_round::generate_trace(proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(proof))),
                public_values: vec![],
            },
        ]
    }
}
