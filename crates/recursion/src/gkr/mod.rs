//! # GKR Air Module
//!
//! The GKR protocol reduces a fractional sum claim $\sum_{y \in H_{\ell+n}}
//! \frac{\hat{p}(y)}{\hat{q}(y)} = 0$ to evaluation claims on the input layer polynomials at a
//! random point. This is done through a layer-by-layer recursive reduction, where each layer uses a
//! sumcheck protocol.
//!
//! The GKR Air Module verifies the [`GkrProof`](stark_backend_v2::proof::GkrProof) struct and
//! consists of four AIRs:
//!
//! 1. **GkrInputAir** - Handles initial setup, coordinates other AIRs, and sends final claims to batch constraint module
//! 2. **GkrLayerAir** - Manages layer-by-layer GKR reduction (verifies [`verify_gkr`](stark_backend_v2::verifier::fractional_sumcheck_gkr::verify_gkr))
//! 3. **GkrLayerSumcheckAir** - Executes sumcheck protocol for each layer (verifies [`verify_gkr_sumcheck`](stark_backend_v2::verifier::fractional_sumcheck_gkr::verify_gkr_sumcheck))
//! 4. **GkrXiSamplerAir** - Samples additional xi randomness challenges if required
//!
//! ## Architecture
//!
//! ```text
//!                                ┌─────────────────┐
//!                                │                 │───────────────────► TranscriptBus
//!                                │ GkrXiSamplerAir │
//!                                │                 │───────────────────► XiRandomnessBus
//!                                └─────────────────┘
//!                                         ▲
//!                                         ┆
//!                         GkrXiSamplerBus ┆
//!                                         ┆
//!                                         ▼
//!                                ┌─────────────────┐
//!                                │                 │───────────────────► TranscriptBus
//!                                │                 │
//!  GkrModuleBus ────────────────►│   GkrInputAir   │───────────────────► ExpBitsLenBus
//!                                │                 │
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

use openvm_stark_backend::{AirRef, p3_maybe_rayon::prelude::*, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, FieldAlgebra};
use stark_backend_v2::{
    D_EF, EF, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{interpolate_cubic_at_0123, interpolate_linear_at_01},
    poseidon2::sponge::{FiatShamirTranscript, ReadOnlyTranscript, TranscriptHistory},
    proof::{GkrProof, Proof},
};

use crate::{
    gkr::{
        bus::{GkrLayerInputBus, GkrLayerOutputBus, GkrXiSamplerBus},
        input::{GkrInputAir, GkrInputRecord},
        layer::{GkrLayerAir, GkrLayerRecord},
        sumcheck::{GkrLayerSumcheckAir, GkrSumcheckRecord},
        xi_sampler::{GkrXiSamplerAir, GkrXiSamplerRecord},
    },
    primitives::exp_bits_len::ExpBitsLenAir,
    system::{AirModule, BusIndexManager, BusInventory, GkrPreflight, Preflight},
};

// Internal bus definitions
mod bus;
pub use bus::{
    GkrSumcheckChallengeBus, GkrSumcheckChallengeMessage, GkrSumcheckInputBus,
    GkrSumcheckInputMessage, GkrSumcheckOutputBus, GkrSumcheckOutputMessage,
};

// Sub-modules for different AIRs
pub mod input;
pub mod layer;
pub mod sumcheck;
pub mod xi_sampler;

pub struct GkrModule {
    // System Params
    l_skip: usize,
    logup_pow_bits: usize,
    // Global bus inventory
    bus_inventory: BusInventory,
    exp_bits_len_air: Arc<ExpBitsLenAir>,
    // Module buses
    xi_sampler_bus: GkrXiSamplerBus,
    layer_input_bus: GkrLayerInputBus,
    layer_output_bus: GkrLayerOutputBus,
    sumcheck_input_bus: GkrSumcheckInputBus,
    sumcheck_output_bus: GkrSumcheckOutputBus,
    sumcheck_challenge_bus: GkrSumcheckChallengeBus,
}

pub struct GkrProofRecord {
    input: GkrInputRecord,
    layer: GkrLayerRecord,
    sumcheck: GkrSumcheckRecord,
    xi_sampler: GkrXiSamplerRecord,
    mus: Vec<EF>,
    q0_claim: EF,
}

impl GkrModule {
    pub fn new(
        mvk: Arc<MultiStarkVerifyingKeyV2>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        exp_bits_len_air: Arc<ExpBitsLenAir>,
    ) -> Self {
        GkrModule {
            l_skip: mvk.inner.params.l_skip,
            logup_pow_bits: mvk.inner.params.logup_pow_bits,
            bus_inventory,
            exp_bits_len_air,
            layer_input_bus: GkrLayerInputBus::new(b.new_bus_idx()),
            layer_output_bus: GkrLayerOutputBus::new(b.new_bus_idx()),
            sumcheck_input_bus: GkrSumcheckInputBus::new(b.new_bus_idx()),
            sumcheck_output_bus: GkrSumcheckOutputBus::new(b.new_bus_idx()),
            sumcheck_challenge_bus: GkrSumcheckChallengeBus::new(b.new_bus_idx()),
            xi_sampler_bus: GkrXiSamplerBus::new(b.new_bus_idx()),
        }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for GkrModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let gkr_input_air = GkrInputAir {
            l_skip: self.l_skip,
            logup_pow_bits: self.logup_pow_bits,
            gkr_module_bus: self.bus_inventory.gkr_module_bus,
            bc_module_bus: self.bus_inventory.bc_module_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            layer_input_bus: self.layer_input_bus,
            layer_output_bus: self.layer_output_bus,
            xi_sampler_bus: self.xi_sampler_bus,
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
            xi_sampler_bus: self.xi_sampler_bus,
        };

        vec![
            Arc::new(gkr_input_air) as AirRef<_>,
            Arc::new(gkr_layer_air) as AirRef<_>,
            Arc::new(gkr_sumcheck_air) as AirRef<_>,
            Arc::new(gkr_xi_sampler_air) as AirRef<_>,
        ]
    }

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight, ts: &mut TS) {
        let GkrProof {
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
            logup_pow_witness,
        } = &proof.gkr_proof;

        ts.observe(*logup_pow_witness);
        let logup_pow_sample = ts.sample();

        self.exp_bits_len_air.add_exp_bits_len(
            F::GENERATOR,
            logup_pow_sample,
            F::from_canonical_usize(self.logup_pow_bits),
            F::ONE,
        );

        let _alpha_logup = ts.sample_ext();
        let _beta_logup = ts.sample_ext();

        let mut xi = vec![(0, EF::ZERO); claims_per_layer.len()];
        let mut gkr_r = vec![EF::ZERO];
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
            let mut eq = EF::ONE;
            let mut gkr_r_prime = Vec::with_capacity(layer_idx);

            for (j, poly) in polys.iter().enumerate() {
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

                claim = claim_out;
                eq = eq_out;
                gkr_r_prime.push(ri);

                if is_final_layer {
                    xi[j + 1] = (ts.len() - D_EF, ri);
                }
            }

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

        for _ in sumcheck_polys.len()..preflight.proof_shape.n_max + self.l_skip {
            xi.push((ts.len(), ts.sample_ext()));
        }

        preflight.gkr = GkrPreflight {
            post_tidx: ts.len(),
            xi,
        };
    }

    fn generate_proof_inputs(
        &self,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProofRawInput<F>> {
        debug_assert_eq!(proofs.len(), preflights.len());

        let per_proof_records: Vec<GkrProofRecord> = proofs
            .par_iter()
            .zip(preflights.par_iter())
            .map(|(proof, preflight)| {
                let start_idx = preflight.proof_shape.post_tidx;
                let mut ts = ReadOnlyTranscript::new(&preflight.transcript, start_idx);

                let gkr_proof = &proof.gkr_proof;
                let GkrProof {
                    q0_claim,
                    claims_per_layer,
                    sumcheck_polys,
                    logup_pow_witness,
                } = gkr_proof;

                ts.observe(*logup_pow_witness);
                let logup_pow_sample = ts.sample();
                let alpha_logup = ts.sample_ext();
                let _beta_logup = ts.sample_ext();

                let xi = &preflight.gkr.xi;

                let input_layer_claim = claims_per_layer
                    .last()
                    .and_then(|last_layer| {
                        xi.first().map(|(_, rho)| {
                            let p_claim =
                                last_layer.p_xi_0 + *rho * (last_layer.p_xi_1 - last_layer.p_xi_0);
                            let q_claim =
                                last_layer.q_xi_0 + *rho * (last_layer.q_xi_1 - last_layer.q_xi_0);
                            [p_claim, q_claim]
                        })
                    })
                    .unwrap_or([EF::ZERO, alpha_logup]);

                let input_record = GkrInputRecord {
                    tidx: preflight.proof_shape.post_tidx,
                    n_logup: preflight.proof_shape.n_logup,
                    n_max: preflight.proof_shape.n_max,
                    logup_pow_witness: *logup_pow_witness,
                    logup_pow_sample,
                    input_layer_claim,
                };

                let num_layers = claims_per_layer.len();
                let sumcheck_layer_count = sumcheck_polys.len();
                let total_sumcheck_rounds: usize = sumcheck_polys.iter().map(Vec::len).sum();

                let tidx_first_gkr_layer = preflight.proof_shape.post_tidx + 2 + 2 * D_EF + D_EF;
                let mut layer_record = GkrLayerRecord {
                    tidx: tidx_first_gkr_layer,
                    layer_claims: Vec::with_capacity(num_layers),
                    lambdas: Vec::with_capacity(sumcheck_layer_count),
                    eq_at_r_primes: Vec::with_capacity(sumcheck_layer_count),
                };
                let mut mus = Vec::with_capacity(num_layers.max(1));

                let tidx_first_sumcheck_round = tidx_first_gkr_layer + 5 * D_EF + D_EF;
                let mut sumcheck_record = GkrSumcheckRecord {
                    tidx: tidx_first_sumcheck_round,
                    ris: Vec::with_capacity(total_sumcheck_rounds),
                    evals: Vec::with_capacity(total_sumcheck_rounds),
                    claims: Vec::with_capacity(sumcheck_layer_count),
                };

                let mut gkr_r: Vec<EF> = Vec::new();
                let mut numer_claim = EF::ZERO;
                let mut denom_claim = EF::ONE;

                if let Some(root_claims) = claims_per_layer.first() {
                    ts.observe_ext(*q0_claim);
                    ts.observe_ext(root_claims.p_xi_0);
                    ts.observe_ext(root_claims.q_xi_0);
                    ts.observe_ext(root_claims.p_xi_1);
                    ts.observe_ext(root_claims.q_xi_1);

                    let mu = ts.sample_ext();
                    numer_claim =
                        interpolate_linear_at_01(&[root_claims.p_xi_0, root_claims.p_xi_1], mu);
                    denom_claim =
                        interpolate_linear_at_01(&[root_claims.q_xi_0, root_claims.q_xi_1], mu);

                    gkr_r.push(mu);

                    layer_record.layer_claims.push([
                        root_claims.p_xi_0,
                        root_claims.q_xi_0,
                        root_claims.p_xi_1,
                        root_claims.q_xi_1,
                    ]);
                    mus.push(mu);
                }

                for (polys, claims) in sumcheck_polys.iter().zip(claims_per_layer.iter().skip(1)) {
                    let lambda = ts.sample_ext();
                    layer_record.lambdas.push(lambda);

                    let mut claim = numer_claim + lambda * denom_claim;
                    let mut eq_at_r_prime = EF::ONE;
                    let mut round_r = Vec::with_capacity(polys.len());

                    sumcheck_record.claims.push(claim);

                    for (round_idx, poly) in polys.iter().enumerate() {
                        for eval in poly {
                            ts.observe_ext(*eval);
                        }

                        let ri = ts.sample_ext();
                        let prev_challenge = gkr_r[round_idx];

                        let ev0 = claim - poly[0];
                        let evals = [ev0, poly[0], poly[1], poly[2]];
                        claim = interpolate_cubic_at_0123(&evals, ri);

                        let eq_factor =
                            prev_challenge * ri + (EF::ONE - prev_challenge) * (EF::ONE - ri);
                        eq_at_r_prime *= eq_factor;

                        sumcheck_record.ris.push(ri);
                        sumcheck_record.evals.push(*poly);
                        round_r.push(ri);
                    }

                    layer_record.eq_at_r_primes.push(eq_at_r_prime);

                    ts.observe_ext(claims.p_xi_0);
                    ts.observe_ext(claims.q_xi_0);
                    ts.observe_ext(claims.p_xi_1);
                    ts.observe_ext(claims.q_xi_1);

                    let mu = ts.sample_ext();
                    numer_claim = interpolate_linear_at_01(&[claims.p_xi_0, claims.p_xi_1], mu);
                    denom_claim = interpolate_linear_at_01(&[claims.q_xi_0, claims.q_xi_1], mu);

                    gkr_r.clear();
                    gkr_r.push(mu);
                    gkr_r.extend(round_r);

                    layer_record.layer_claims.push([
                        claims.p_xi_0,
                        claims.q_xi_0,
                        claims.p_xi_1,
                        claims.q_xi_1,
                    ]);
                    mus.push(mu);
                }

                let xi_sampler_record = if num_layers < xi.len() {
                    let challenges: Vec<EF> =
                        xi.iter().skip(num_layers).map(|(_, val)| *val).collect();
                    let tidx = xi[num_layers].0;
                    GkrXiSamplerRecord {
                        tidx,
                        idx: num_layers,
                        xis: challenges,
                    }
                } else {
                    GkrXiSamplerRecord::default()
                };

                GkrProofRecord {
                    input: input_record,
                    layer: layer_record,
                    sumcheck: sumcheck_record,
                    xi_sampler: xi_sampler_record,
                    mus,
                    q0_claim: *q0_claim,
                }
            })
            .collect();

        let mut input_records = Vec::with_capacity(per_proof_records.len());
        let mut layer_records = Vec::with_capacity(per_proof_records.len());
        let mut sumcheck_records = Vec::with_capacity(per_proof_records.len());
        let mut xi_sampler_records = Vec::with_capacity(per_proof_records.len());
        let mut mus_records = Vec::with_capacity(per_proof_records.len());
        let mut q0_claims = Vec::with_capacity(per_proof_records.len());

        for records in per_proof_records {
            let GkrProofRecord {
                input,
                layer,
                sumcheck,
                xi_sampler,
                mus,
                q0_claim,
            } = records;

            input_records.push(input);
            layer_records.push(layer);
            sumcheck_records.push(sumcheck);
            xi_sampler_records.push(xi_sampler);
            mus_records.push(mus);
            q0_claims.push(q0_claim);
        }

        vec![
            // GkrInputAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(input::generate_trace(input_records, &q0_claims))),
                public_values: vec![],
            },
            // GkrLayerAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(layer::generate_trace(
                    layer_records,
                    &mus_records,
                    &q0_claims,
                ))),
                public_values: vec![],
            },
            // GkrLayerSumcheckAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(
                    sumcheck_records,
                    &mus_records,
                ))),
                public_values: vec![],
            },
            // GkrXiSamplerAir proof input
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(xi_sampler::generate_trace(xi_sampler_records))),
                public_values: vec![],
            },
        ]
    }
}
