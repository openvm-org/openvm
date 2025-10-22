use std::sync::Arc;

use itertools::{Itertools, izip};
use openvm_stark_backend::{AirRef, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, FieldAlgebra, TwoAdicField};
use stark_backend_v2::{
    EF, F,
    keygen::types::{MultiStarkVerifyingKeyV2, SystemParams},
    poly_common::{Squarable, interpolate_quadratic_at_012},
    poseidon2::sponge::FiatShamirTranscript,
    proof::{Proof, WhirProof},
    prover::poly::Mle,
};

use crate::{
    primitives::exp_bits_len::ExpBitsLenAir,
    system::{AirModule, BusIndexManager, BusInventory, Preflight, WhirPreflight},
    whir::{
        bus::{
            FinalPolyMleEvalBus, FinalPolyQueryEvalBus, VerifyQueriesBus, VerifyQueryBus,
            WhirAlphaBus, WhirEqAlphaUBus, WhirFinalPolyBus, WhirFoldingBus, WhirGammaBus,
            WhirQueryBus, WhirSumcheckBus,
        },
        final_poly_mle_eval::FinalPoleMleEvalAir,
        final_poly_query_eval::FinalPolyQueryEvalAir,
        folding::WhirFoldingAir,
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
mod folding;
mod initial_opened_values;
mod non_initial_opened_values;
mod query;
mod sumcheck;
mod whir_round;

pub struct WhirModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    bus_inventory: BusInventory,
    exp_bits_len_air: Arc<ExpBitsLenAir>,

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
}

impl WhirModule {
    pub fn new(
        mvk: Arc<MultiStarkVerifyingKeyV2>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        exp_bits_len_air: Arc<ExpBitsLenAir>,
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
        Self {
            mvk,
            bus_inventory,
            exp_bits_len_air,
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
        }
    }
}

impl<TS: FiatShamirTranscript> AirModule<TS> for WhirModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let whir_round_air = WhirRoundAir {
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
            k: self.mvk.inner.params.k_whir,
            num_queries: self.mvk.inner.params.num_whir_queries,
            final_poly_len: 1 << self.mvk.inner.params.log_final_poly_len,
            pow_bits: self.mvk.inner.params.whir_pow_bits,
            generator: F::GENERATOR,
        };
        let whir_sumcheck_air = SumcheckAir {
            sumcheck_bus: self.sumcheck_bus,
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            alpha_bus: self.alpha_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
        };
        let initial_round_opened_values_air = InitialOpenedValuesAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            k: self.mvk.inner.params.k_whir,
        };
        let non_initial_round_opened_values_air = NonInitialOpenedValuesAir {
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            k: self.mvk.inner.params.k_whir,
        };
        let query_air = WhirQueryAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            query_bus: self.query_bus,
            verify_queries_bus: self.verify_queries_bus,
            verify_query_bus: self.verify_query_bus,
        };
        let folding_air = WhirFoldingAir {
            alpha_bus: self.alpha_bus,
            folding_bus: self.folding_bus,
            k: self.mvk.inner.params.k_whir,
        };
        let final_poly_mle_eval_air = FinalPoleMleEvalAir {
            stacking_randomness_bus: self.bus_inventory.stacking_randomness_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
            final_poly_bus: self.final_poly_bus,
        };
        let final_poly_query_eval_air = FinalPolyQueryEvalAir {
            query_bus: self.query_bus,
            alpha_bus: self.alpha_bus,
            gamma_bus: self.gamma_bus,
            final_poly_bus: self.final_poly_bus,
            final_poly_query_eval_bus: self.final_poly_query_eval_bus,
        };
        vec![
            Arc::new(whir_round_air),
            Arc::new(whir_sumcheck_air),
            Arc::new(initial_round_opened_values_air),
            Arc::new(non_initial_round_opened_values_air),
            Arc::new(query_air),
            Arc::new(folding_air),
            Arc::new(final_poly_mle_eval_air),
            Arc::new(final_poly_query_eval_air),
        ]
    }

    fn run_preflight(&self, proof: &Proof, preflight: &mut Preflight<TS>) {
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

        let SystemParams {
            l_skip,
            n_stack,
            k_whir,
            num_whir_queries,
            log_blowup,
            ..
        } = self.mvk.inner.params;

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

        let num_whir_rounds = whir_pow_witnesses.len();
        let mut gammas = vec![];
        let mut z0s = vec![];
        let mut alphas = vec![];
        let mut pow_samples = vec![];
        let mut queries = vec![];
        let mut query_indices = vec![];
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
        let mut initial_round_coset_vals = vec![];

        let mut log_rs_domain_size = l_skip + n_stack + log_blowup;

        debug_assert_eq!(ood_values.len(), num_whir_rounds - 1);
        debug_assert_eq!(codeword_commits.len(), num_whir_rounds - 1);

        for i in 0..num_whir_rounds {
            initial_claim_per_round.push(claim);
            tidx_per_round.push(ts.len());

            for j in 0..k_whir {
                let evals = sumcheck_poly_iter.next().unwrap();
                let &[ev1, ev2] = evals;
                ts.observe_ext(ev1);
                ts.observe_ext(ev2);
                let ev0 = claim - ev1;
                let alpha = ts.sample_ext();
                alphas.push(alpha);

                let u = preflight.stacking.sumcheck_rnd[i * k_whir + j];
                eq_partial *= EF::ONE - alpha - u + alpha * u.double();
                eq_partials.push(eq_partial);

                claim = interpolate_quadratic_at_012(&[ev0, ev1, ev2], alpha);
                post_sumcheck_claims.push(claim);
            }
            if i != num_whir_rounds - 1 {
                ts.observe_slice(&codeword_commits[i]);
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

            self.exp_bits_len_air.add_exp_bits_len(
                F::GENERATOR,
                pow_sample,
                F::from_canonical_usize(self.mvk.inner.params.whir_pow_bits),
                F::ONE,
            );

            query_tidx_per_round.push(ts.len());
            let mut round_queries = vec![];
            let mut round_query_indices = vec![];
            for _ in 0..num_whir_queries {
                let (sample, idx) = ts.sample_bits(log_rs_domain_size);
                round_queries.push(sample);
                round_query_indices.push(idx);
            }
            queries.extend(&round_queries);
            query_indices.extend(&round_query_indices);
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
            let m = self.mvk.inner.params.n_stack
                + self.mvk.inner.params.l_skip
                + self.mvk.inner.params.log_blowup;
            for (query_idx, index) in round_query_indices.into_iter().enumerate() {
                let zj_root = omega.exp_u64(index as u64);
                let zj = zj_root.exp_power_of_2(k_whir);

                self.exp_bits_len_air.add_exp_bits_len(
                    F::two_adic_generator(log_rs_domain_size - k_whir),
                    round_queries[query_idx],
                    F::from_canonical_usize(log_rs_domain_size - k_whir),
                    zj_root,
                );

                let yj = if i == 0 {
                    let mut codeword_vals = vec![EF::ZERO; 1 << k_whir];
                    let mut mu_pow_iter = mu_pows.iter();
                    for opened_rows_per_query in &proof.whir_proof.initial_round_opened_rows {
                        let opened_rows = &opened_rows_per_query[query_idx];
                        let width = opened_rows.len() / (1 << k_whir);

                        for c in 0..width {
                            let mu_pow = mu_pow_iter.next().unwrap(); // ok; mu_pows has total_width length
                            for j in 0..(1 << k_whir) {
                                codeword_vals[j] += *mu_pow * opened_rows[j * width + c];
                            }
                        }
                    }
                    initial_round_coset_vals.push(codeword_vals.clone());
                    binary_k_fold(
                        codeword_vals,
                        &alphas[alphas.len() - k_whir..],
                        zj_root,
                        i,
                        query_idx,
                        &mut fold_records,
                    )
                } else {
                    let opened_rows =
                        proof.whir_proof.codeword_opened_rows[i - 1][query_idx].clone();
                    binary_k_fold(
                        opened_rows,
                        &alphas[alphas.len() - k_whir..],
                        zj_root,
                        i,
                        query_idx,
                        &mut fold_records,
                    )
                };
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
        let u = preflight.stacking.sumcheck_rnd[0]
            .exp_powers_of_2()
            .take(self.mvk.inner.params.l_skip)
            .chain(preflight.stacking.sumcheck_rnd[1..].iter().copied())
            .collect_vec();
        let final_poly_at_u = f.eval_at_point(&u[t..]);

        preflight.whir = WhirPreflight {
            alphas,
            z0s,
            zj_roots,
            zjs,
            yjs,
            gammas,
            pow_samples,
            queries,
            query_indices,
            tidx_per_round,
            query_tidx_per_round,
            initial_claim_per_round,
            pre_query_claims,
            post_sumcheck_claims,
            eq_partials,
            fold_records,
            initial_round_coset_vals,
            final_poly_at_u,
        };
    }

    fn generate_proof_inputs(
        &self,
        proof: &Proof,
        preflight: &Preflight<TS>,
    ) -> Vec<AirProofRawInput<F>> {
        vec![
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(whir_round::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(sumcheck::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(initial_opened_values::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(non_initial_opened_values::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(query::generate_trace(&self.mvk, proof, preflight))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(folding::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(final_poly_mle_eval::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
            AirProofRawInput {
                cached_mains: vec![],
                common_main: Some(Arc::new(final_poly_query_eval::generate_trace(
                    &self.mvk, proof, preflight,
                ))),
                public_values: vec![],
            },
        ]
    }
}

#[derive(Clone, Debug)]
pub struct FoldRecord {
    whir_round: usize,
    query_idx: usize,
    twiddle: F,
    coset_shift: F,
    coset_size: usize,
    coset_idx: usize,
    height: usize,
    left_value: EF,
    right_value: EF,
    value: EF,
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
            records.push(FoldRecord {
                whir_round,
                query_idx,
                twiddle: tw[i << j],
                coset_shift,
                coset_size: m,
                coset_idx: i,
                height: j + 1,
                left_value: lo[i],
                right_value: hi[i],
                value: new_val,
            });
            lo[i] = new_val;
        }
    }
    values[0]
}
