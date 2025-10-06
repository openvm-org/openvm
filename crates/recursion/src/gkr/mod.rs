use core::iter::zip;
use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{GkrProof, Proof},
};

use crate::{
    gkr::dummy::DummyGkrRoundAir,
    system::{AirModule, BusInventory, GkrPreflight, Preflight},
};

mod dummy;

pub struct GkrModule {
    bus_inventory: BusInventory,
}

impl GkrModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        GkrModule { bus_inventory }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for GkrModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let gkr_verify_air = DummyGkrRoundAir {
            gkr_bus: self.bus_inventory.gkr_module_bus,
            bc_module_bus: self.bus_inventory.bc_module_bus,
            xi_randomness_bus: self.bus_inventory.xi_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
        };
        vec![Arc::new(gkr_verify_air) as AirRef<_>]
    }

    fn run_preflight(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &mut Preflight<TS>,
    ) {
        let GkrProof {
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
            logup_pow_witness,
        } = &proof.gkr_proof;

        let ts = &mut preflight.transcript;

        ts.observe(*logup_pow_witness);
        let _pow_bits = ts.sample();

        let _alpha_logup = ts.sample_ext();
        let _beta_logup = ts.sample_ext();

        if !sumcheck_polys.is_empty() {
            ts.observe_ext(*q0_claim);
        }

        let mut xi = vec![(0, EF::ZERO); sumcheck_polys.len() + 1];

        for (i, (polys, claims)) in zip(sumcheck_polys, claims_per_layer).enumerate() {
            let is_final_layer = i == sumcheck_polys.len() - 1;

            let _ = ts.sample_ext();

            for (j, poly) in polys.iter().enumerate() {
                for eval in poly {
                    ts.observe_ext(*eval);
                }
                let xi_i = ts.sample_ext();
                if is_final_layer {
                    xi[j + 1] = (ts.len() - D_EF, xi_i);
                }
            }

            ts.observe_ext(claims.p_xi_0);
            ts.observe_ext(claims.q_xi_0);
            ts.observe_ext(claims.p_xi_1);
            ts.observe_ext(claims.q_xi_1);

            let rho = ts.sample_ext();
            if is_final_layer {
                xi[0] = (ts.len() - D_EF, rho);
            }
        }

        for _ in sumcheck_polys.len()..preflight.proof_shape.n_max + vk.inner.params.l_skip {
            xi.push((ts.len(), ts.sample_ext()));
        }

        preflight.gkr = GkrPreflight {
            post_tidx: ts.len(),
            xi,
        };
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
