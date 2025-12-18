use core::{cmp, iter::zip};
use std::sync::Arc;

use itertools::{Itertools, izip};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, FieldAlgebra, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
#[cfg(not(debug_assertions))]
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{
    EF, F, SystemParams, WhirConfig,
    keygen::types::MultiStarkVerifyingKeyV2,
    poly_common::{Squarable, interpolate_quadratic_at_012},
    poseidon2::{
        CHUNK,
        sponge::{FiatShamirTranscript, TranscriptHistory},
    },
    proof::{Proof, WhirProof},
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2, poly::Mle},
};

use crate::{
    primitives::exp_bits_len::ExpBitsLenTraceGenerator,
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, ModuleChip, Preflight,
        TraceGenModule, WhirPreflight,
    },
    whir::{
        bus::{
            FinalPolyFoldingBus, FinalPolyMleEvalBus, FinalPolyQueryEvalBus, VerifyQueriesBus,
            VerifyQueryBus, WhirAlphaBus, WhirEqAlphaUBus, WhirFinalPolyBus, WhirFoldingBus,
            WhirGammaBus, WhirQueryBus, WhirSumcheckBus,
        },
        final_poly_mle_eval::FinalPolyMleEvalAir,
        final_poly_query_eval::{FinalPolyQueryEvalAir, FinalPolyQueryEvalRecord},
        folding::{FoldRecord, WhirFoldingAir},
        initial_opened_values::InitialOpenedValuesAir,
        non_initial_opened_values::NonInitialOpenedValuesAir,
        query::WhirQueryAir,
        sumcheck::SumcheckAir,
        whir_round::WhirRoundAir,
    },
};

mod bus;
mod final_poly_mle_eval;
mod final_poly_query_eval;
pub mod folding;
mod initial_opened_values;
mod non_initial_opened_values;
mod query;
mod sumcheck;
mod whir_round;

pub(crate) fn num_queries_per_round(params: &SystemParams) -> Vec<usize> {
    params
        .whir
        .rounds
        .iter()
        .map(|round| round.num_queries)
        .collect()
}

pub(crate) fn query_offsets(num_queries_per_round: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(num_queries_per_round.len() + 1);
    offsets.push(0);
    for &num_queries in num_queries_per_round {
        offsets.push(offsets.last().copied().unwrap() + num_queries);
    }
    offsets
}

pub(crate) fn total_num_queries(num_queries_per_round: &[usize]) -> usize {
    num_queries_per_round.iter().sum()
}

#[cfg(feature = "cuda")]
mod cuda_abi;

pub struct WhirModule {
    params: SystemParams,
    bus_inventory: BusInventory,

    // "execution" buses
    sumcheck_bus: WhirSumcheckBus,
    verify_queries_bus: VerifyQueriesBus,
    verify_query_bus: VerifyQueryBus,
    folding_bus: WhirFoldingBus,
    final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    final_poly_query_eval_bus: FinalPolyQueryEvalBus,

    // data buses
    alpha_bus: WhirAlphaBus,
    gamma_bus: WhirGammaBus,
    query_bus: WhirQueryBus,
    eq_alpha_u_bus: WhirEqAlphaUBus,
    final_poly_bus: WhirFinalPolyBus,
    final_poly_folding_bus: FinalPolyFoldingBus,
}

impl WhirModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKeyV2,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        let sumcheck_bus = WhirSumcheckBus::new(b.new_bus_idx());
        let alpha_bus = WhirAlphaBus::new(b.new_bus_idx());
        let gamma_bus = WhirGammaBus::new(b.new_bus_idx());
        let query_bus = WhirQueryBus::new(b.new_bus_idx());
        let verify_queries_bus = VerifyQueriesBus::new(b.new_bus_idx());
        let verify_query_bus = VerifyQueryBus::new(b.new_bus_idx());
        let eq_alpha_u_bus = WhirEqAlphaUBus::new(b.new_bus_idx());
        let folding_bus = WhirFoldingBus::new(b.new_bus_idx());
        let final_poly_mle_eval_bus = FinalPolyMleEvalBus::new(b.new_bus_idx());
        let final_poly_query_eval_bus = FinalPolyQueryEvalBus::new(b.new_bus_idx());
        let final_poly_bus = WhirFinalPolyBus::new(b.new_bus_idx());
        let final_poly_folding_bus = FinalPolyFoldingBus::new(b.new_bus_idx());
        Self {
            params: child_vk.inner.params.clone(),
            bus_inventory,
            sumcheck_bus,
            verify_queries_bus,
            verify_query_bus,
            final_poly_mle_eval_bus,
            final_poly_query_eval_bus,
            folding_bus,
            alpha_bus,
            gamma_bus,
            query_bus,
            eq_alpha_u_bus,
            final_poly_bus,
            final_poly_folding_bus,
        }
    }
}

impl WhirModule {
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS: FiatShamirTranscript + TranscriptHistory>(
        &self,
        proof: &Proof,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) {
        let WhirProof {
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            initial_round_opened_rows: _,
            initial_round_merkle_proofs: _,
            codeword_opened_values: _,
            codeword_merkle_proofs: _,
            whir_pow_witnesses,
            final_poly,
        } = &proof.whir_proof;

        let &SystemParams {
            l_skip,
            n_stack,
            whir:
                WhirConfig {
                    k: k_whir,
                    query_phase_pow_bits: _,
                    ..
                },
            log_blowup,
            ..
        } = &self.params;
        let num_queries_per_round = num_queries_per_round(&self.params);

        let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();
        let all_openings = proof
            .stacking_proof
            .stacking_openings
            .iter()
            .flatten()
            .collect_vec();
        let mu = preflight.stacking.stacking_batching_challenge;
        let mu_pows = mu.powers().take(all_openings.len()).collect_vec();
        let mut claim = all_openings
            .into_iter()
            .zip(mu.powers())
            .fold(EF::ZERO, |acc, (&opening, mu_pow)| acc + mu_pow * opening);

        let u = preflight.stacking.sumcheck_rnd[0]
            .exp_powers_of_2()
            .take(l_skip)
            .chain(preflight.stacking.sumcheck_rnd[1..].iter().copied())
            .collect_vec();

        let num_whir_rounds = whir_pow_witnesses.len();
        let mut gammas = vec![];
        let mut z0s = vec![];
        let mut alphas = vec![];
        let mut pow_samples = vec![];
        let mut queries = vec![];
        let mut tidx_per_round = vec![];
        let mut query_tidx_per_round = vec![];
        let mut initial_claim_per_round = vec![];
        let mut post_sumcheck_claims = vec![];
        let mut pre_query_claims = vec![];
        let mut zj_roots = vec![];
        let mut zjs = vec![];
        let mut yjs = vec![];
        let mut eq_partials = vec![];
        let mut eq_partial = EF::ONE;
        let mut fold_records = vec![];

        let mut log_rs_domain_size = l_skip + n_stack + log_blowup;

        debug_assert_eq!(ood_values.len(), num_whir_rounds - 1);
        debug_assert_eq!(codeword_commits.len(), num_whir_rounds - 1);

        for i in 0..num_whir_rounds {
            initial_claim_per_round.push(claim);
            tidx_per_round.push(ts.len());

            let num_round_queries = num_queries_per_round[i];
            for j in 0..k_whir {
                let evals = sumcheck_poly_iter.next().unwrap();
                let &[ev1, ev2] = evals;
                ts.observe_ext(ev1);
                ts.observe_ext(ev2);
                let ev0 = claim - ev1;
                let alpha = ts.sample_ext();
                alphas.push(alpha);

                let uj = u[i * k_whir + j];
                eq_partial *= EF::ONE - alpha - uj + alpha * uj.double();
                eq_partials.push(eq_partial);

                claim = interpolate_quadratic_at_012(&[ev0, ev1, ev2], alpha);
                post_sumcheck_claims.push(claim);
            }
            if i != num_whir_rounds - 1 {
                ts.observe_commit(codeword_commits[i]);
                z0s.push(ts.sample_ext());
                ts.observe_ext(ood_values[i]);
            } else {
                for coeff in final_poly {
                    ts.observe_ext(*coeff);
                }
            };

            ts.observe(whir_pow_witnesses[i]);
            let pow_sample = ts.sample();
            pow_samples.push(pow_sample);

            query_tidx_per_round.push(ts.len());
            let mut round_queries = vec![];
            for _ in 0..num_round_queries {
                let sample = ts.sample();
                round_queries.push(sample);
            }
            queries.extend(&round_queries);
            let gamma = ts.sample_ext();
            gammas.push(gamma);

            if let Some(&y0) = ood_values.get(i) {
                claim += gamma * y0;
            }
            let mut gamma_pows = gamma.powers().skip(2);

            let mut zj_roots_round = vec![];
            let mut zjs_round = vec![];
            let mut yjs_round = vec![];

            pre_query_claims.push(claim);
            let omega = F::two_adic_generator(log_rs_domain_size);
            for (query_idx, sample) in round_queries.into_iter().enumerate() {
                let index = sample.as_canonical_u32() & ((1 << (log_rs_domain_size - k_whir)) - 1);
                let zj_root = omega.exp_u64(index as u64);

                let zj = zj_root.exp_power_of_2(k_whir);
                let record_start = fold_records.len();
                let yj = if i == 0 {
                    let mut codeword_vals = vec![EF::ZERO; 1 << k_whir];
                    let mut mu_pow_iter = mu_pows.iter();
                    for opened_rows_per_query in proof.whir_proof.initial_round_opened_rows.iter() {
                        let opened_rows = &opened_rows_per_query[query_idx];
                        let width = opened_rows[0].len();

                        for c in 0..width {
                            let mu_pow = mu_pow_iter.next().unwrap(); // ok; mu_pows has total_width length
                            for j in 0..(1 << k_whir) {
                                codeword_vals[j] += *mu_pow * opened_rows[j][c];
                            }
                        }
                    }
                    binary_k_fold(
                        codeword_vals,
                        &alphas[alphas.len() - k_whir..],
                        zj_root,
                        i,
                        query_idx,
                        &mut fold_records,
                    )
                } else {
                    let opened_values =
                        proof.whir_proof.codeword_opened_values[i - 1][query_idx].clone();
                    binary_k_fold(
                        opened_values,
                        &alphas[alphas.len() - k_whir..],
                        zj_root,
                        i,
                        query_idx,
                        &mut fold_records,
                    )
                };
                for rec in &mut fold_records[record_start..] {
                    rec.set_final_values(zj, yj);
                }
                zj_roots_round.push(zj_root);
                zjs_round.push(zj);
                yjs_round.push(yj);

                claim += gamma_pows.next().unwrap() * yj;
            }
            log_rs_domain_size -= 1;

            zj_roots.push(zj_roots_round);
            zjs.push(zjs_round);
            yjs.push(yjs_round);

            gamma_pows.next().unwrap();
        }
        // push one for the final claim
        initial_claim_per_round.push(claim);
        debug_assert!(sumcheck_poly_iter.next().is_none());

        let f = Mle::from_coeffs(final_poly.clone());
        let t = k_whir * num_whir_rounds;
        let final_poly_at_u = f.eval_at_point(&u[t..]);
        let query_offsets = query_offsets(&num_queries_per_round);

        preflight.whir = WhirPreflight {
            num_queries_per_round,
            query_offsets,
            alphas,
            z0s,
            zj_roots,
            zjs,
            yjs,
            gammas,
            pow_samples,
            queries,
            tidx_per_round,
            query_tidx_per_round,
            initial_claim_per_round,
            pre_query_claims,
            post_sumcheck_claims,
            eq_partials,
            fold_records,
            final_poly_at_u,
        };
    }
}

struct WhirBlobCpu {
    codeword_value_accs: Vec<EF>,
    iov_rows_per_proof_psums: Vec<usize>,
    commits_per_proof_psums: Vec<usize>,
    stacking_chunks_psums: Vec<usize>,
    stacking_widths_psums: Vec<usize>,
    mu_pows: Vec<EF>, // flattened over proofs
    final_poly_query_eval_records: Vec<FinalPolyQueryEvalRecord>,
}

impl AirModule for WhirModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let params = &self.params;
        let initial_log_domain_size = params.n_stack + params.l_skip + params.log_blowup;

        let num_rounds = params.num_whir_rounds();
        let num_queries_per_round: Vec<usize> =
            params.whir.rounds.iter().map(|r| r.num_queries).collect();

        // Encoder requires at least 2 flags to work correctly
        let whir_round_encoder = Encoder::new(num_rounds.max(2), 2, false);

        let whir_round_air: AirRef<BabyBearPoseidon2Config> = Arc::new(WhirRoundAir {
            whir_module_bus: self.bus_inventory.whir_module_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            sumcheck_bus: self.sumcheck_bus,
            verify_queries_bus: self.verify_queries_bus,
            final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
            final_poly_query_eval_bus: self.final_poly_query_eval_bus,
            query_bus: self.query_bus,
            gamma_bus: self.gamma_bus,
            k: params.k_whir(),
            num_rounds,
            final_poly_len: 1 << params.log_final_poly_len(),
            pow_bits: params.whir.query_phase_pow_bits,
            generator: F::GENERATOR,
            whir_round_encoder,
            num_queries_per_round: num_queries_per_round.clone(),
        });
        let whir_sumcheck_air = SumcheckAir {
            sumcheck_bus: self.sumcheck_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            alpha_bus: self.alpha_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
            k: params.k_whir(),
        };
        let initial_round_opened_values_air = InitialOpenedValuesAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            poseidon_bus: self.bus_inventory.poseidon2_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let non_initial_round_opened_values_air = NonInitialOpenedValuesAir {
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            poseidon_bus: self.bus_inventory.poseidon2_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let query_air = WhirQueryAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            query_bus: self.query_bus,
            verify_queries_bus: self.verify_queries_bus,
            verify_query_bus: self.verify_query_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let folding_air = WhirFoldingAir {
            alpha_bus: self.alpha_bus,
            folding_bus: self.folding_bus,
            k: params.k_whir(),
        };
        let final_poly_mle_eval_air = FinalPolyMleEvalAir {
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
            final_poly_bus: self.final_poly_bus,
            folding_bus: self.final_poly_folding_bus,
            num_vars: params.log_final_poly_len(),
            num_sumcheck_rounds: params.num_whir_sumcheck_rounds(),
            num_whir_rounds: params.num_whir_rounds(),
            total_whir_queries: params
                .whir
                .rounds
                .iter()
                .map(|cfg| cfg.num_queries + 1)
                .sum(),
        };
        let final_poly_query_eval_air = FinalPolyQueryEvalAir {
            query_bus: self.query_bus,
            alpha_bus: self.alpha_bus,
            gamma_bus: self.gamma_bus,
            final_poly_bus: self.final_poly_bus,
            final_poly_query_eval_bus: self.final_poly_query_eval_bus,
            num_whir_rounds: params.num_whir_rounds(),
            k_whir: params.k_whir(),
            log_final_poly_len: params.log_final_poly_len(),
        };
        vec![
            whir_round_air,
            Arc::new(whir_sumcheck_air),
            Arc::new(query_air),
            Arc::new(initial_round_opened_values_air),
            Arc::new(non_initial_round_opened_values_air),
            Arc::new(folding_air),
            Arc::new(final_poly_mle_eval_air),
            Arc::new(final_poly_query_eval_air),
        ]
    }
}

impl WhirModule {
    #[tracing::instrument(skip_all)]
    fn generate_blob(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[&Proof],
        preflights: &[&Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
    ) -> WhirBlobCpu {
        // TODO: Move more stuff here. (Out of preflight and redundant logic in generate_trace
        // functions.)
        let SystemParams {
            l_skip,
            n_stack,
            whir,
            log_blowup,
            ..
        } = &child_vk.inner.params;
        let WhirConfig {
            k: k_whir,
            rounds: _,
            query_phase_pow_bits,
            folding_pow_bits: _,
        } = whir;
        let num_queries_per_round = num_queries_per_round(&child_vk.inner.params);
        let num_initial_queries = *num_queries_per_round.first().unwrap_or(&0);
        let num_cosets = 1 << k_whir;

        let codeword_value_accs_capacity: usize = proofs
            .iter()
            .map(|proof| {
                let total_chunks: usize = proof
                    .whir_proof
                    .initial_round_opened_rows
                    .iter()
                    .map(|c| c[0][0].len().div_ceil(CHUNK))
                    .sum();
                num_initial_queries * num_cosets * total_chunks
            })
            .sum();
        let mut codeword_value_accs = Vec::with_capacity(codeword_value_accs_capacity);

        let mut iov_rows_per_proof_psums = vec![0];
        let mut commits_per_proof_psums = vec![0];

        let mut stacking_chunks_psums = vec![0];
        let mut stacking_widths_psums = vec![0];

        let mut mu_pows = vec![];
        let final_poly_query_eval_records =
            final_poly_query_eval::build_final_poly_query_eval_records(
                &child_vk.inner.params,
                proofs,
                preflights,
            );

        let mut total_iov_rows = 0;
        let mut total_commits = 0;

        for (proof, preflight) in zip(proofs, preflights) {
            exp_bits_len_gen.add_requests(
                preflight
                    .whir
                    .pow_samples
                    .iter()
                    .map(|pow_sample| (F::GENERATOR, *pow_sample, *query_phase_pow_bits)),
            );

            let mut log_rs_domain_size = l_skip + n_stack + log_blowup;
            for window in preflight.whir.query_offsets.windows(2) {
                let start = window[0];
                let end = window[1];
                let round_queries = &preflight.whir.queries[start..end];
                let omega = F::two_adic_generator(log_rs_domain_size);
                exp_bits_len_gen.add_requests(
                    round_queries
                        .iter()
                        .copied()
                        .map(|sample| (omega, sample, log_rs_domain_size - k_whir)),
                );
                log_rs_domain_size -= 1;
            }

            let mut total_width_for_proof = 0;
            let mut total_chunks_for_proof = 0;

            let num_commits_for_proof = proof.whir_proof.initial_round_opened_rows.len();

            for openings_per_commit in proof.whir_proof.initial_round_opened_rows.iter() {
                let width = openings_per_commit[0][0].len();
                let chunks = width.div_ceil(CHUNK);

                total_width_for_proof += width;
                total_chunks_for_proof += chunks;

                let last_width = *stacking_widths_psums.last().unwrap();
                stacking_widths_psums.push(last_width + width);

                let last_chunks = *stacking_chunks_psums.last().unwrap();
                stacking_chunks_psums.push(last_chunks + chunks);
            }
            total_commits += num_commits_for_proof;
            commits_per_proof_psums.push(total_commits);

            total_iov_rows += (total_chunks_for_proof * num_initial_queries) << k_whir;
            iov_rows_per_proof_psums.push(total_iov_rows);

            let mu = preflight.stacking.stacking_batching_challenge;
            let mu_pow_offset = mu_pows.len();
            mu_pows.extend(mu.powers().take(total_width_for_proof));

            for query_idx in 0..num_initial_queries {
                let mut codeword_vals = EF::zero_vec(1 << k_whir);
                for (coset_idx, codeword_val) in codeword_vals.iter_mut().enumerate() {
                    let mut base = 0;
                    for opened_rows_per_query in proof.whir_proof.initial_round_opened_rows.iter() {
                        let opened_rows = &opened_rows_per_query[query_idx];
                        let width = opened_rows[0].len();
                        let num_chunks = width.div_ceil(CHUNK);

                        for chunk_idx in 0..num_chunks {
                            let chunk_start = chunk_idx * CHUNK;
                            let chunk_len = cmp::min(CHUNK, width - chunk_start);

                            let opened_chunk =
                                &opened_rows[coset_idx][chunk_start..chunk_start + chunk_len];

                            codeword_value_accs.push(*codeword_val);

                            for (offset, &val) in opened_chunk.iter().enumerate() {
                                *codeword_val +=
                                    mu_pows[mu_pow_offset + base + chunk_start + offset] * val;
                            }
                        }
                        base += width;
                    }
                }
            }
        }
        WhirBlobCpu {
            codeword_value_accs,
            iov_rows_per_proof_psums,
            commits_per_proof_psums,
            stacking_chunks_psums,
            stacking_widths_psums,
            mu_pows,
            final_poly_query_eval_records,
        }
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for WhirModule {
    type ModuleSpecificCtx = ExpBitsLenTraceGenerator;

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let proofs = proofs.iter().collect_vec();
        let preflights = preflights.iter().collect_vec();
        let blob = self.generate_blob(child_vk, &proofs, &preflights, exp_bits_len_gen);

        let chips = [
            WhirModuleChip::WhirRound,
            WhirModuleChip::Sumcheck,
            WhirModuleChip::Query,
            WhirModuleChip::InitialOpenedValues,
            WhirModuleChip::NonInitialOpenedValues,
            WhirModuleChip::Folding,
            WhirModuleChip::FinalPolyMleEval,
            WhirModuleChip::FinalPolyQueryEval,
        ];
        #[cfg(debug_assertions)]
        let iter = chips.iter();
        #[cfg(not(debug_assertions))]
        let iter = chips.par_iter();
        iter.map(|chip| chip.generate_trace_row_major(child_vk, &proofs, &preflights, &blob))
            .map(|trace| ColMajorMatrix::from_row_major(&trace))
            .map(AirProvingContextV2::simple_no_pis)
            .collect()
    }
}

fn binary_k_fold(
    mut values: Vec<EF>,
    alphas: &[EF],
    base_coset_shift: F,
    whir_round: usize,
    query_idx: usize,
    records: &mut Vec<FoldRecord>,
) -> EF {
    let n = values.len();
    let k = alphas.len();
    debug_assert_eq!(n, 1 << k);

    let omega_k = F::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();

    let tw = omega_k.powers().take(1 << (k - 1)).collect_vec();
    let inv_tw = omega_k_inv.powers().take(1 << (k - 1)).collect_vec();

    for (j, (&alpha, coset_shift, coset_shift_inv)) in izip!(
        alphas.iter(),
        base_coset_shift.exp_powers_of_2(),
        base_coset_shift.inverse().exp_powers_of_2()
    )
    .enumerate()
    {
        let m = n >> (j + 1);
        let (lo, hi) = values.split_at_mut(m);

        for i in 0..m {
            let x = tw[i << j] * coset_shift;
            let x_inv = inv_tw[i << j] * coset_shift_inv;
            let new_val = lo[i] + (alpha - x) * (lo[i] - hi[i]) * x_inv.halve();
            records.push(FoldRecord::new(
                whir_round,
                query_idx,
                tw[i << j],
                coset_shift,
                m,
                i,
                j + 1,
                lo[i],
                hi[i],
                new_val,
                alpha,
            ));
            lo[i] = new_val;
        }
    }
    values[0]
}

#[derive(strum_macros::Display)]
enum WhirModuleChip {
    WhirRound,
    Sumcheck,
    Query,
    InitialOpenedValues,
    NonInitialOpenedValues,
    Folding,
    FinalPolyQueryEval,
    FinalPolyMleEval,
}

impl WhirModuleChip {
    fn generate_trace_row_major(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[&Proof],
        preflights: &[&Preflight],
        blob: &WhirBlobCpu,
    ) -> RowMajorMatrix<F> {
        use WhirModuleChip::*;
        match self {
            WhirRound => whir_round::generate_trace(child_vk, proofs, preflights),
            Sumcheck => sumcheck::generate_trace(child_vk, proofs, preflights),
            Query => query::generate_trace(child_vk, proofs, preflights),
            InitialOpenedValues => initial_opened_values::generate_trace(
                &child_vk.inner.params,
                proofs,
                preflights,
                &blob.codeword_value_accs,
                &blob.iov_rows_per_proof_psums,
                &blob.commits_per_proof_psums,
                &blob.stacking_chunks_psums,
                &blob.stacking_widths_psums,
                &blob.mu_pows,
            ),
            NonInitialOpenedValues => non_initial_opened_values::generate_trace(
                child_vk,
                proofs,
                preflights,
            ),
            Folding => folding::generate_trace(child_vk, proofs, preflights),
            FinalPolyMleEval => final_poly_mle_eval::generate_trace(child_vk, proofs, preflights),
            FinalPolyQueryEval => final_poly_query_eval::generate_trace(
                child_vk,
                proofs,
                &blob.final_poly_query_eval_records,
            ),
        }
    }
}

impl ModuleChip<GlobalCtxCpu, CpuBackendV2> for WhirModuleChip {
    type ModuleSpecificCtx = WhirBlobCpu;

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        blob: &WhirBlobCpu,
    ) -> ColMajorMatrix<F> {
        let proofs = proofs.iter().collect_vec();
        let preflights = preflights.iter().collect_vec();
        ColMajorMatrix::from_row_major(&self.generate_trace_row_major(
            child_vk,
            &proofs,
            &preflights,
            blob,
        ))
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use cuda_backend_v2::GpuBackendV2;
    use openvm_cuda_backend::{base::DeviceMatrix, data_transporter::transport_matrix_to_device};
    use openvm_cuda_common::d_buffer::DeviceBuffer;
    use stark_backend_v2::poseidon2::{CHUNK, WIDTH};
    use std::cmp;

    use super::*;
    use crate::cuda::{
        GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, to_device_or_nullptr,
        vk::VerifyingKeyGpu,
    };
    use crate::whir::cuda_abi::PoseidonStatePair;

    pub(in crate::whir) struct WhirBlobGpu {
        pub zis: DeviceBuffer<F>,
        pub zi_roots: DeviceBuffer<F>,
        pub yis: DeviceBuffer<EF>,
        pub raw_queries: DeviceBuffer<F>,
        pub codeword_value_accs: DeviceBuffer<EF>,
        pub poseidon_states: DeviceBuffer<PoseidonStatePair>,
        pub iov_rows_per_proof_psums: DeviceBuffer<usize>,
        pub commits_per_proof_psums: DeviceBuffer<usize>,
        pub stacking_chunks_psums: DeviceBuffer<usize>,
        pub stacking_widths_psums: DeviceBuffer<usize>,
        pub mus: DeviceBuffer<EF>,
        pub mu_pows: DeviceBuffer<EF>,
        pub final_poly_query_eval_records: DeviceBuffer<FinalPolyQueryEvalRecord>,
        pub codeword_opened_values: DeviceBuffer<EF>,
        pub codeword_states: DeviceBuffer<F>,
        pub folding_records: DeviceBuffer<FoldRecord>,
    }

    fn build_initial_poseidon_state_pairs(
        proofs: &[&Proof],
        preflights: &[&Preflight],
    ) -> Vec<PoseidonStatePair> {
        let expected_pairs: usize = preflights
            .iter()
            .flat_map(|preflight| {
                preflight
                    .initial_row_states
                    .iter()
                    .flat_map(|commit| commit.iter().flat_map(|query| query.iter()))
            })
            .map(|coset_states| coset_states.len())
            .sum();

        let mut pairs = Vec::with_capacity(expected_pairs);
        for (proof, preflight) in proofs.iter().zip(preflights) {
            if preflight.initial_row_states.is_empty() {
                continue;
            }
            let num_commits = preflight.initial_row_states.len();
            let num_queries = preflight.initial_row_states[0].len();
            let num_cosets = preflight.initial_row_states[0][0].len();

            for query_idx in 0..num_queries {
                for coset_idx in 0..num_cosets {
                    for commit_idx in 0..num_commits {
                        let chunk_states =
                            &preflight.initial_row_states[commit_idx][query_idx][coset_idx];
                        let opened_row = &proof.whir_proof.initial_round_opened_rows[commit_idx]
                            [query_idx][coset_idx];
                        for (chunk_idx, &post_state) in chunk_states.iter().enumerate() {
                            let mut pre_state = if chunk_idx > 0 {
                                chunk_states[chunk_idx - 1]
                            } else {
                                [F::ZERO; WIDTH]
                            };
                            let chunk_start = chunk_idx * CHUNK;
                            let chunk_len = cmp::min(CHUNK, opened_row.len() - chunk_start);
                            pre_state[..chunk_len].copy_from_slice(
                                &opened_row[chunk_start..chunk_start + chunk_len],
                            );
                            pairs.push(PoseidonStatePair {
                                pre_state,
                                post_state,
                            });
                        }
                    }
                }
            }
        }
        pairs
    }

    impl WhirBlobGpu {
        // TODO: Receive PreflightGPU?
        fn new(proofs: &[&Proof], preflights: &[&Preflight], blob: &WhirBlobCpu) -> Self {
            let mus = to_device_or_nullptr(
                &preflights
                    .iter()
                    .map(|preflight| preflight.stacking.stacking_batching_challenge)
                    .collect_vec(),
            )
            .unwrap();
            let zis = to_device_or_nullptr(
                &preflights
                    .iter()
                    .flat_map(|preflight| preflight.whir.zjs.iter().flatten().copied())
                    .collect_vec(),
            )
            .unwrap();
            let zi_roots = to_device_or_nullptr(
                &preflights
                    .iter()
                    .flat_map(|preflight| preflight.whir.zj_roots.iter().flatten().copied())
                    .collect_vec(),
            )
            .unwrap();
            let yis = to_device_or_nullptr(
                &preflights
                    .iter()
                    .flat_map(|preflight| preflight.whir.yjs.iter().flatten().copied())
                    .collect_vec(),
            )
            .unwrap();
            let raw_queries = to_device_or_nullptr(
                &preflights
                    .iter()
                    .flat_map(|preflight| preflight.whir.queries.iter().copied())
                    .collect_vec(),
            )
            .unwrap();
            let iov_rows_per_proof_psums =
                to_device_or_nullptr(&blob.iov_rows_per_proof_psums).unwrap();
            let commits_per_proof_psums =
                to_device_or_nullptr(&blob.commits_per_proof_psums).unwrap();
            let stacking_chunks_psums = to_device_or_nullptr(&blob.stacking_chunks_psums).unwrap();
            let stacking_widths_psums = to_device_or_nullptr(&blob.stacking_widths_psums).unwrap();
            let mu_pows = to_device_or_nullptr(&blob.mu_pows).unwrap();
            let codeword_value_accs = to_device_or_nullptr(&blob.codeword_value_accs).unwrap();

            // Build poseidon state pairs in kernel order: [query][coset][commit][chunk]
            let poseidon_states_host = build_initial_poseidon_state_pairs(proofs, preflights);
            let poseidon_states = to_device_or_nullptr(&poseidon_states_host).unwrap();

            let folding_records_cap: usize = preflights
                .iter()
                .map(|p| p.whir.fold_records.len())
                .sum();
            let mut folding_records_host = Vec::with_capacity(folding_records_cap);
            for preflight in preflights.iter() {
                folding_records_host.extend_from_slice(&preflight.whir.fold_records);
            }
            let folding_records = to_device_or_nullptr(&folding_records_host).unwrap();
            let final_poly_query_eval_records =
                to_device_or_nullptr(&blob.final_poly_query_eval_records).unwrap();

            let codeword_opened_values_cap: usize = proofs
                .iter()
                .map(|p| {
                    p.whir_proof
                        .codeword_opened_values
                        .iter()
                        .map(|r| r.iter().map(|q| q.len()).sum::<usize>())
                        .sum::<usize>()
                })
                .sum();
            let mut codeword_opened_values_host = Vec::with_capacity(codeword_opened_values_cap);
            for proof in proofs.iter() {
                for round in proof.whir_proof.codeword_opened_values.iter() {
                    for query in round.iter() {
                        codeword_opened_values_host.extend_from_slice(query);
                    }
                }
            }
            let codeword_opened_values = to_device_or_nullptr(&codeword_opened_values_host).unwrap();

            // Must be in same order as codeword_opened_values
            let mut codeword_states_host = Vec::with_capacity(codeword_opened_values_cap * WIDTH);
            for preflight in preflights.iter() {
                for round in preflight.codeword_states.iter() {
                    for query in round.iter() {
                        for state in query.iter() {
                            codeword_states_host.extend_from_slice(state);
                        }
                    }
                }
            }
            let codeword_states = to_device_or_nullptr(&codeword_states_host).unwrap();

            WhirBlobGpu {
                zis,
                zi_roots,
                yis,
                raw_queries,
                codeword_value_accs,
                poseidon_states,
                iov_rows_per_proof_psums,
                commits_per_proof_psums,
                stacking_chunks_psums,
                stacking_widths_psums,
                mus,
                mu_pows,
                final_poly_query_eval_records,
                folding_records,
                codeword_opened_values,
                codeword_states,
            }
        }
    }

    impl WhirModuleChip {
        #[tracing::instrument(
            name = "wrapper.generate_trace",
            level = "trace",
            skip_all,
            fields(air = %self)
        )]
        fn generate_gpu_trace(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs_cpu: &[&Proof],
            preflights_cpu: &[&Preflight],
            blob: &WhirBlobGpu,
            blob_cpu: &WhirBlobCpu,
        ) -> DeviceMatrix<F> {
            match self {
                WhirModuleChip::InitialOpenedValues => initial_opened_values::cuda::generate_trace(
                    proofs_cpu.len(),
                    &blob,
                    &child_vk.system_params,
                ),
                WhirModuleChip::NonInitialOpenedValues => {
                    non_initial_opened_values::cuda::generate_trace(&blob, &child_vk.system_params)
                }
                WhirModuleChip::FinalPolyQueryEval => {
                    final_poly_query_eval::cuda::generate_trace(&blob, &child_vk.system_params)
                }
                WhirModuleChip::Folding => {
                    folding::cuda::generate_trace(&blob, &child_vk.system_params, proofs_cpu.len())
                }
                _ => {
                    // Fall back to CPU impl.
                    let mat = self.generate_trace_row_major(
                        &child_vk.cpu,
                        &proofs_cpu,
                        &preflights_cpu,
                        &blob_cpu,
                    );
                    transport_matrix_to_device(Arc::new(mat))
                }
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for WhirModule {
        type ModuleSpecificCtx = ExpBitsLenTraceGenerator;

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            exp_bits_len_gen: &ExpBitsLenTraceGenerator,
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            let proofs_cpu = &proofs.iter().map(|proof| &proof.cpu).collect_vec();
            let preflights_cpu = &preflights
                .iter()
                .map(|preflight| &preflight.cpu)
                .collect_vec();

            let blob =
                self.generate_blob(&child_vk.cpu, proofs_cpu, preflights_cpu, exp_bits_len_gen);
            let blob_gpu = WhirBlobGpu::new(proofs_cpu, preflights_cpu, &blob);

            [
                WhirModuleChip::WhirRound,
                WhirModuleChip::Sumcheck,
                WhirModuleChip::Query,
                WhirModuleChip::InitialOpenedValues,
                WhirModuleChip::NonInitialOpenedValues,
                WhirModuleChip::Folding,
                WhirModuleChip::FinalPolyMleEval,
                WhirModuleChip::FinalPolyQueryEval,
            ]
            .iter()
            .map(|chip| {
                chip.generate_gpu_trace(child_vk, proofs_cpu, preflights_cpu, &blob_gpu, &blob)
            })
            .map(|mat| AirProvingContextV2::simple_no_pis(mat))
            .collect()
        }
    }
}
