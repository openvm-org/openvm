//! # GKR Air Module
//!
//! The GKR protocol reduces a fractional sum claim $\sum_{y \in H_{\ell+n}}
//! \frac{\hat{p}(y)}{\hat{q}(y)} = 0$ to evaluation claims on the input layer polynomials at a
//! random point. This is done through a layer-by-layer recursive reduction, where each layer uses a
//! sumcheck protocol.
//!
//! The GKR Air Module consists of four AIRs:
//!
//! 1. **GkrInputAir** - Handles initial setup and coordinates other AIRs
//! 2. **GkrLayerAir** - Manages layer-by-layer GKR reduction
//! 3. **GkrLayerSumcheckAir** - Executes sumcheck protocol for each layer
//! 4. **GkrXiSamplerAir** - Samples additional xi randomness challenges
//!
//! ## Architecture
//!
//! ```text
//!                                ┌─────────────────┐
//!                                │                 │───────────────────► TranscriptBus
//!                                │ GkrXiSamplerAir │
//!                                │                 │───────────────────► XiRandomnessBus
//!                                └─────────────────┘
//!                                      ▲      ┆
//!                                      ┆      ┆
//!                 GkrXiSamplerInputBus ┆      ┆ GkrXiSamplerOutputBus
//!                                      ┆      ┆
//!                                      ┆      ▼
//!                                ┌─────────────────┐
//!                                │                 │───────────────────► TranscriptBus
//!   GkrModuleBus ───────────────►│   GkrInputAir   │
//!                                │                 │───────────────────► BatchConstraintModuleBus
//!                                └─────────────────┘
//!                                      ┆      ▲
//!                                      ┆      ┆
//!                     GkrLayerInputBus ┆      ┆ GkrLayerOutputBus
//!                                      ┆      ┆
//!                                      ▼      ┆
//!                             ┌─────────────────────────┐
//!                             │                         │──────────────► TranscriptBus
//!   ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│       GkrLayerAir       │
//!   ┆                         │                         │──────────────► XiRandomnessBus
//!   ┆                         └─────────────────────────┘
//!   ┆                                  ┆      ▲
//!   ┆                                  ┆      ┆
//!   ┆              GkrSumcheckInputBus ┆      ┆ GkrSumcheckOutputBus
//!   ┆                                  ┆      ┆
//!   ┆                                  ▼      ┆
//!   ┆ GkrSumcheckChallengeBus ┌─────────────────────────┐
//!   ┆┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│                         │──────────────► TranscriptBus
//!   ┆                         │   GkrLayerSumcheckAir   │
//!   └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄►│                         │──────────────► XiRandomnessBus
//!                             └─────────────────────────┘
//! ```

use core::iter::zip;
use std::sync::Arc;

use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::FieldAlgebra;
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{interpolate_cubic_at_0123, interpolate_linear_at_01},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{GkrProof, Proof},
};

use crate::{
    gkr::{
        bus::{GkrLayerInputBus, GkrLayerOutputBus, GkrXiSamplerInputBus, GkrXiSamplerOutputBus},
        input::GkrInputAir,
        layer::GkrLayerAir,
        sumcheck::GkrLayerSumcheckAir,
        xi_sampler::GkrXiSamplerAir,
    },
    system::{AirModule, BusIndexManager, BusInventory, GkrPreflight, Preflight},
};

// Internal bus definitions
mod bus;
pub use bus::{
    GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage, GkrSumcheckInputBus,
    GkrSumcheckInputMessage, GkrSumcheckOutputBus, GkrSumcheckOutputMessage,
};

// Sub-AIRs
pub mod sub_air;

// Sub-modules for different AIRs
// mod dummy;
// TODO: rename to gkr dispatch?
mod input;
mod layer;
mod sumcheck;
mod xi_sampler;

pub struct GkrModule {
    bus_inventory: BusInventory,
    // Internal buses
    xi_sampler_input_bus: GkrXiSamplerInputBus,
    xi_sampler_output_bus: GkrXiSamplerOutputBus,
    layer_input_bus: GkrLayerInputBus,
    layer_output_bus: GkrLayerOutputBus,
    sumcheck_input_bus: GkrSumcheckInputBus,
    sumcheck_output_bus: GkrSumcheckOutputBus,
    sumcheck_challenge_bus: GkrSumcheckChallengeBus,
}

impl GkrModule {
    pub fn new(bus_inventory: BusInventory) -> Self {
        // TODO: Move to the central bus inventory or have bus inventory share the next available
        // index
        let mut bus_manager = BusIndexManager::new();
        // TODO: fix this
        for _ in 0..20 {
            bus_manager.new_bus_idx();
        }

        GkrModule {
            bus_inventory,
            layer_input_bus: GkrLayerInputBus::new(bus_manager.new_bus_idx()),
            layer_output_bus: GkrLayerOutputBus::new(bus_manager.new_bus_idx()),
            sumcheck_input_bus: GkrSumcheckInputBus::new(bus_manager.new_bus_idx()),
            sumcheck_output_bus: GkrSumcheckOutputBus::new(bus_manager.new_bus_idx()),
            sumcheck_challenge_bus: GkrSumcheckChallengeBus::new(bus_manager.new_bus_idx()),
            xi_sampler_input_bus: GkrXiSamplerInputBus::new(bus_manager.new_bus_idx()),
            xi_sampler_output_bus: GkrXiSamplerOutputBus::new(bus_manager.new_bus_idx()),
        }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for GkrModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let gkr_input_air = GkrInputAir {
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            bc_module_bus: self.bus_inventory.bc_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            layer_input_bus: self.layer_input_bus,
            layer_output_bus: self.layer_output_bus,
            xi_sampler_input_bus: self.xi_sampler_input_bus,
            xi_sampler_output_bus: self.xi_sampler_output_bus,
        };

        let gkr_layer_air = GkrLayerAir {
            xi_randomness_bus: self.bus_inventory.xi_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            layer_input_bus: self.layer_input_bus,
            layer_output_bus: self.layer_output_bus,
            sumcheck_input_bus: self.sumcheck_input_bus,
            sumcheck_challenge_bus: self.sumcheck_challenge_bus,
            sumcheck_output_bus: self.sumcheck_output_bus,
        };

        let gkr_sumcheck_air = GkrLayerSumcheckAir::new(
            self.bus_inventory.transcript_bus,
            self.bus_inventory.xi_randomness_bus,
            self.sumcheck_input_bus,
            self.sumcheck_output_bus,
            self.sumcheck_challenge_bus,
        );

        let gkr_xi_sampler_air = GkrXiSamplerAir {
            xi_randomness_bus: self.bus_inventory.xi_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            xi_sampler_input_bus: self.xi_sampler_input_bus,
            xi_sampler_output_bus: self.xi_sampler_output_bus,
        };

        // let dummy_air = dummy::DummyGkrRoundAir {
        //     gkr_bus: self.bus_inventory.gkr_module_bus,
        //     bc_module_bus: self.bus_inventory.bc_module_bus,
        //     xi_randomness_bus: self.bus_inventory.xi_randomness_bus,
        //     transcript_bus: self.bus_inventory.transcript_bus,
        // };

        vec![
            Arc::new(gkr_input_air) as AirRef<_>,
            Arc::new(gkr_layer_air) as AirRef<_>,
            Arc::new(gkr_sumcheck_air) as AirRef<_>,
            Arc::new(gkr_xi_sampler_air) as AirRef<_>,
            // Arc::new(dummy_air) as AirRef<_>,
        ]
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

        let mut xi = vec![(0, EF::ZERO); claims_per_layer.len()];
        let mut sumcheck_round_data = Vec::new();
        let mut layer_sumcheck_output = Vec::new();
        let mut layer_claim = Vec::new();
        let mut gkr_r: Vec<EF> = Vec::new();
        let mut numer_claim = EF::ZERO;
        let mut denom_claim = EF::ONE;

        if !claims_per_layer.is_empty() {
            debug_assert_eq!(sumcheck_polys.len() + 1, claims_per_layer.len());

            ts.observe_ext(*q0_claim);

            let claims = &claims_per_layer[0];

            ts.observe_ext(claims.p_xi_0);
            ts.observe_ext(claims.q_xi_0);
            ts.observe_ext(claims.p_xi_1);
            ts.observe_ext(claims.q_xi_1);

            let mu = ts.sample_ext();
            // Reduce layer 0 claims to single evaluation
            numer_claim = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
            denom_claim = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
            gkr_r = vec![mu];
        }

        for (i, (polys, claims)) in zip(sumcheck_polys, claims_per_layer.iter().skip(1)).enumerate()
        {
            let layer_idx = i + 1;
            let is_final_layer = i == sumcheck_polys.len() - 1;

            let lambda = ts.sample_ext();

            // Compute initial claim for this layer using numer_claim and denom_claim from previous
            // layer
            let mut claim = numer_claim + lambda * denom_claim;
            layer_claim.push(claim);
            let mut eq = EF::ONE;
            let mut gkr_r_prime = Vec::with_capacity(layer_idx);

            for (j, poly) in polys.iter().enumerate() {
                let claim_in = claim;
                let eq_in = eq;

                for eval in poly {
                    ts.observe_ext(*eval);
                }
                let ri = ts.sample_ext();

                // Compute claim_out via cubic interpolation
                let ev0 = claim - poly[0];
                let evals = [ev0, poly[0], poly[1], poly[2]];
                let claim_out = interpolate_cubic_at_0123(&evals, ri);

                // Update eq incrementally: eq *= xi * ri + (1 - xi) * (1 - ri)
                let xi_j = gkr_r[j];
                let eq_out = eq * (xi_j * ri + (EF::ONE - xi_j) * (EF::ONE - ri));

                sumcheck_round_data.push((layer_idx, j, claim_in, claim_out, eq_in, eq_out));

                claim = claim_out;
                eq = eq_out;
                gkr_r_prime.push(ri);

                if is_final_layer {
                    xi[j + 1] = (ts.len() - D_EF, ri);
                }
            }

            // Store the final sumcheck output for this layer (new_claim, eq_at_r_prime)
            layer_sumcheck_output.push((claim, eq));

            ts.observe_ext(claims.p_xi_0);
            ts.observe_ext(claims.q_xi_0);
            ts.observe_ext(claims.p_xi_1);
            ts.observe_ext(claims.q_xi_1);

            let mu = ts.sample_ext();
            // Reduce current layer claims to single evaluation for next layer
            numer_claim = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
            denom_claim = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);
            gkr_r = std::iter::once(mu).chain(gkr_r_prime).collect();

            if is_final_layer {
                xi[0] = (ts.len() - D_EF, mu);
            }
        }

        let post_layer_tidx = ts.len();
        for _ in sumcheck_polys.len()..preflight.proof_shape.n_max + vk.inner.params.l_skip {
            xi.push((ts.len(), ts.sample_ext()));
        }

        preflight.gkr = GkrPreflight {
            post_tidx: ts.len(),
            post_layer_tidx,
            xi,
            sumcheck_round_data,
            layer_sumcheck_output,
            layer_claim,
        };
    }

    fn generate_proof_inputs(
        &self,
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            // GkrInputAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(input::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            // GkrLayerAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(layer::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            // GkrLayerSumcheckAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            // GkrXiSamplerAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(xi_sampler::generate_trace(vk, proof, preflight))),
                public_values: vec![],
            },
            // AirProofRawInput {
            //     cached_mains: vec![],
            //     common_main: Some(Arc::new(dummy::generate_trace(vk, proof, preflight))),
            //     public_values: vec![],
            // },
        ]
    }
}
