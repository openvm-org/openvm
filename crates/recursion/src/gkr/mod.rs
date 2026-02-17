//! # GKR Air Module
//!
//! The GKR protocol reduces a fractional sum claim $\sum_{y \in H_{\ell+n}}
//! \frac{\hat{p}(y)}{\hat{q}(y)} = 0$ to evaluation claims on the input layer polynomials at a
//! random point. This is done through a layer-by-layer recursive reduction, where each layer uses a
//! sumcheck protocol.
//!
//! The GKR Air Module verifies the [`GkrProof`](openvm_stark_backend::proof::GkrProof) struct and
//! consists of four AIRs:
//!
//! 1. **GkrInputAir** - Handles initial setup, coordinates other AIRs, and sends final claims to
//!    batch constraint module
//! 2. **GkrLayerAir** - Manages layer-by-layer GKR reduction (verifies
//!    [`verify_gkr`](openvm_stark_backend::verifier::fractional_sumcheck_gkr::verify_gkr))
//! 3. **GkrLayerSumcheckAir** - Executes sumcheck protocol for each layer (verifies
//!    [`verify_gkr_sumcheck`](openvm_stark_backend::verifier::fractional_sumcheck_gkr::verify_gkr_sumcheck))
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

use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_maybe_rayon::prelude::*,
    poly_common::{interpolate_cubic_at_0123, interpolate_linear_at_01},
    proof::{GkrProof, Proof},
    prover::{AirProvingContext, CpuBackend},
    AirRef, FiatShamirTranscript, ReadOnlyTranscript, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use strum::EnumCount;

use crate::{
    gkr::{
        bus::{GkrLayerInputBus, GkrLayerOutputBus, GkrXiSamplerBus},
        input::{GkrInputAir, GkrInputRecord, GkrInputTraceGenerator},
        layer::{GkrLayerAir, GkrLayerRecord, GkrLayerTraceGenerator},
        sumcheck::{GkrLayerSumcheckAir, GkrSumcheckRecord, GkrSumcheckTraceGenerator},
        xi_sampler::{GkrXiSamplerAir, GkrXiSamplerRecord, GkrXiSamplerTraceGenerator},
    },
    primitives::exp_bits_len::ExpBitsLenTraceGenerator,
    system::{
        AirModule, BusIndexManager, BusInventory, GkrPreflight, GlobalCtxCpu, Preflight,
        TraceGenModule,
    },
    tracegen::{ModuleChip, RowMajorChip},
    utils::{pow_observe_sample, pow_tidx_count},
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
    // Module buses
    xi_sampler_bus: GkrXiSamplerBus,
    layer_input_bus: GkrLayerInputBus,
    layer_output_bus: GkrLayerOutputBus,
    sumcheck_input_bus: GkrSumcheckInputBus,
    sumcheck_output_bus: GkrSumcheckOutputBus,
    sumcheck_challenge_bus: GkrSumcheckChallengeBus,
}

struct GkrBlobCpu {
    input_records: Vec<GkrInputRecord>,
    layer_records: Vec<GkrLayerRecord>,
    sumcheck_records: Vec<GkrSumcheckRecord>,
    xi_sampler_records: Vec<GkrXiSamplerRecord>,
    mus_records: Vec<Vec<EF>>,
    q0_claims: Vec<EF>,
}

impl GkrModule {
    pub fn new(
        mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        GkrModule {
            l_skip: mvk.inner.params.l_skip,
            logup_pow_bits: mvk.inner.params.logup.pow_bits,
            bus_inventory,
            layer_input_bus: GkrLayerInputBus::new(b.new_bus_idx()),
            layer_output_bus: GkrLayerOutputBus::new(b.new_bus_idx()),
            sumcheck_input_bus: GkrSumcheckInputBus::new(b.new_bus_idx()),
            sumcheck_output_bus: GkrSumcheckOutputBus::new(b.new_bus_idx()),
            sumcheck_challenge_bus: GkrSumcheckChallengeBus::new(b.new_bus_idx()),
            xi_sampler_bus: GkrXiSamplerBus::new(b.new_bus_idx()),
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
    {
        let GkrProof {
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
            logup_pow_witness,
        } = &proof.gkr_proof;

        let _logup_pow_sample = pow_observe_sample(ts, self.logup_pow_bits, *logup_pow_witness);
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
}

impl AirModule for GkrModule {
    fn num_airs(&self) -> usize {
        GkrModuleChipDiscriminants::COUNT
    }

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
}

impl GkrModule {
    #[tracing::instrument(skip_all)]
    fn generate_blob(
        &self,
        _child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[&Proof<BabyBearPoseidon2Config>],
        preflights: &[&Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
    ) -> GkrBlobCpu {
        debug_assert_eq!(proofs.len(), preflights.len());

        // NOTE: we only collect the zipped vec because rayon vs itertools has different treatment
        // of multiunzip. This could be addressed with a macro similar to parizip!
        let zipped_records: Vec<_> = proofs
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

                let logup_pow_sample =
                    pow_observe_sample(&mut ts, self.logup_pow_bits, *logup_pow_witness);
                if self.logup_pow_bits > 0 {
                    exp_bits_len_gen.add_request(
                        F::GENERATOR,
                        logup_pow_sample,
                        self.logup_pow_bits,
                    );
                }

                let alpha_logup =
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);
                let _beta_logup =
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);

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
                    alpha_logup,
                    input_layer_claim,
                };

                let num_layers = claims_per_layer.len();
                let sumcheck_layer_count = sumcheck_polys.len();
                let total_sumcheck_rounds: usize = sumcheck_polys.iter().map(Vec::len).sum();

                let logup_pow_offset = pow_tidx_count(self.logup_pow_bits);
                let tidx_first_gkr_layer =
                    preflight.proof_shape.post_tidx + logup_pow_offset + 2 * D_EF + D_EF;
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
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts, *q0_claim,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        root_claims.p_xi_0,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        root_claims.q_xi_0,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        root_claims.p_xi_1,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        root_claims.q_xi_1,
                    );

                    let mu = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);
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
                    let lambda =
                        FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);
                    layer_record.lambdas.push(lambda);

                    let mut claim = numer_claim + lambda * denom_claim;
                    let mut eq_at_r_prime = EF::ONE;
                    let mut round_r = Vec::with_capacity(polys.len());

                    sumcheck_record.claims.push(claim);

                    for (round_idx, poly) in polys.iter().enumerate() {
                        for eval in poly {
                            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                                &mut ts, *eval,
                            );
                        }

                        let ri =
                            FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);
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

                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        claims.p_xi_0,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        claims.q_xi_0,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        claims.p_xi_1,
                    );
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                        &mut ts,
                        claims.q_xi_1,
                    );

                    let mu = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut ts);
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

                (
                    input_record,
                    layer_record,
                    sumcheck_record,
                    xi_sampler_record,
                    mus,
                    *q0_claim,
                )
            })
            .collect();
        let (
            input_records,
            layer_records,
            sumcheck_records,
            xi_sampler_records,
            mus_records,
            q0_claims,
        ): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            zipped_records.into_iter().multiunzip();

        GkrBlobCpu {
            input_records,
            layer_records,
            sumcheck_records,
            xi_sampler_records,
            mus_records,
            q0_claims,
        }
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackend<BabyBearPoseidon2Config>> for GkrModule {
    type ModuleSpecificCtx<'a> = ExpBitsLenTraceGenerator;

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>> {
        let proof_refs = proofs.iter().collect_vec();
        let preflight_refs = preflights.iter().collect_vec();
        let blob = self.generate_blob(child_vk, &proof_refs, &preflight_refs, exp_bits_len_gen);

        let chips = [
            GkrModuleChip::Input,
            GkrModuleChip::Layer,
            GkrModuleChip::LayerSumcheck,
            GkrModuleChip::XiSampler,
        ];

        let span = tracing::Span::current();
        chips
            .par_iter()
            .map(|chip| {
                let _guard = span.enter();
                chip.generate_proving_ctx(
                    &blob,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
}

// To reduce the number of structs and trait implementations, we collect them into a single enum
// with enum dispatch.
#[derive(strum_macros::Display, strum::EnumDiscriminants)]
#[strum_discriminants(derive(strum_macros::EnumCount))]
#[strum_discriminants(repr(usize))]
enum GkrModuleChip {
    Input,
    Layer,
    LayerSumcheck,
    XiSampler,
}

impl GkrModuleChip {
    fn index(&self) -> usize {
        GkrModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for GkrModuleChip {
    type Ctx<'a> = GkrBlobCpu;

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        blob: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use GkrModuleChip::*;
        match self {
            Input => GkrInputTraceGenerator
                .generate_trace(&(&blob.input_records, &blob.q0_claims), required_height),
            Layer => GkrLayerTraceGenerator.generate_trace(
                &(&blob.layer_records, &blob.mus_records, &blob.q0_claims),
                required_height,
            ),
            LayerSumcheck => GkrSumcheckTraceGenerator.generate_trace(
                &(&blob.sumcheck_records, &blob.mus_records),
                required_height,
            ),
            XiSampler => GkrXiSamplerTraceGenerator
                .generate_trace(&blob.xi_sampler_records.as_slice(), required_height),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use itertools::Itertools;
    use openvm_cuda_backend::GpuBackend;
    use openvm_stark_backend::p3_maybe_rayon::prelude::*;

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        tracegen::cuda::generate_gpu_proving_ctx,
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for GkrModule {
        type ModuleSpecificCtx<'a> = ExpBitsLenTraceGenerator;

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            exp_bits_len_gen: &ExpBitsLenTraceGenerator,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            let proofs_cpu = proofs.iter().map(|proof| &proof.cpu).collect_vec();
            let preflights_cpu = preflights
                .iter()
                .map(|preflight| &preflight.cpu)
                .collect_vec();
            let blob = self.generate_blob(
                &child_vk.cpu,
                &proofs_cpu,
                &preflights_cpu,
                exp_bits_len_gen,
            );
            let chips = [
                GkrModuleChip::Input,
                GkrModuleChip::Layer,
                GkrModuleChip::LayerSumcheck,
                GkrModuleChip::XiSampler,
            ];

            let span = tracing::Span::current();
            chips
                .par_iter()
                .map(|chip| {
                    let _guard = span.enter();
                    generate_gpu_proving_ctx(
                        chip,
                        &blob,
                        required_heights.map(|heights| heights[chip.index()]),
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
                .collect()
        }
    }
}
