use halo2_base::{utils::biguint_to_fe, AssignedValue};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as RootConfig, Digest as RootDigest,
    },
    openvm_stark_backend::{
        keygen::types::MultiStarkVerifyingKey0,
        p3_field::{
            BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField, PrimeField64,
            TwoAdicField,
        },
        proof::WhirProof,
    },
};

use crate::{
    chip_traits::{
        BabyBearExt4Inst, BabyBearInst, DigestHashInst, GateInst, PopulateInputs, TranscriptInst,
    },
    field::baby_bear::{
        BabyBearExt4Wire, BabyBearExtWire, BabyBearWire, ReducedBabyBearExtWire,
        ReducedBabyBearWire, BABY_BEAR_EXT_DEGREE,
    },
    profiling::CellProfiler,
    stages::{
        batch_constraints::{eval_eq_mle_assigned, eval_eq_mle_ef_f_assigned},
        shared_math::{
            horner_eval_ext_poly_assigned, horner_eval_ext_poly_f_assigned,
            interpolate_quadratic_at_012_assigned,
        },
    },
    transcript::digest_wire_from_root,
    Fr, RootEF, RootF,
};

#[derive(Clone, Debug)]
pub struct MerklePathWire<F = AssignedValue<Fr>> {
    pub leaf_values: Vec<Vec<ReducedBabyBearWire<F>>>,
    pub siblings: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct WhirProofWire<F = AssignedValue<Fr>> {
    pub mu_pow_witness: ReducedBabyBearWire<F>,
    pub folding_pow_witnesses: Vec<ReducedBabyBearWire<F>>,
    pub query_phase_pow_witnesses: Vec<ReducedBabyBearWire<F>>,
    pub whir_sumcheck_polys: Vec<[ReducedBabyBearExtWire<F>; 2]>,
    pub ood_values: Vec<ReducedBabyBearExtWire<F>>,
    pub final_poly: Vec<ReducedBabyBearExtWire<F>>,
    pub codeword_commitment_roots: Vec<F>,
    pub initial_round_merkle_paths: Vec<Vec<MerklePathWire<F>>>,
    pub codeword_merkle_paths: Vec<Vec<MerklePathWire<F>>>,
}

pub(crate) fn load_whir_proof_wire<B: PopulateInputs>(
    b: &mut B,
    whir_proof: &WhirProof<RootConfig>,
) -> WhirProofWire<B::F> {
    let mu_pow_witness = b.bb_load_reduced_witness(whir_proof.mu_pow_witness);
    let folding_pow_witnesses = whir_proof
        .folding_pow_witnesses
        .iter()
        .map(|&witness| b.bb_load_reduced_witness(RootF::from_u64(witness.as_canonical_u64())))
        .collect::<Vec<_>>();
    let query_phase_pow_witnesses = whir_proof
        .query_phase_pow_witnesses
        .iter()
        .map(|&witness| b.bb_load_reduced_witness(RootF::from_u64(witness.as_canonical_u64())))
        .collect::<Vec<_>>();
    let whir_sumcheck_polys = whir_proof
        .whir_sumcheck_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| b.ext_load_reduced_witness(value))
                .collect::<Vec<_>>()
                .try_into()
                .expect("WHIR sumcheck polynomial must have two evaluations")
        })
        .collect::<Vec<_>>();
    let ood_values = whir_proof
        .ood_values
        .iter()
        .map(|&value| b.ext_load_reduced_witness(value))
        .collect::<Vec<_>>();
    let final_poly = whir_proof
        .final_poly
        .iter()
        .map(|&value| b.ext_load_reduced_witness(value))
        .collect::<Vec<_>>();
    let codeword_commitment_roots = whir_proof
        .codeword_commits
        .iter()
        .map(|&digest| b.load_witness(digest_to_fr(digest)))
        .collect::<Vec<_>>();

    let initial_round_merkle_paths = whir_proof
        .initial_round_opened_rows
        .iter()
        .zip(whir_proof.initial_round_merkle_proofs.iter())
        .map(|(rows_per_query, proofs_per_query)| {
            rows_per_query
                .iter()
                .zip(proofs_per_query.iter())
                .map(|(opened_rows, merkle_proof)| {
                    let leaf_values = opened_rows
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|&value| b.bb_load_reduced_witness(value))
                                .collect::<Vec<ReducedBabyBearWire<B::F>>>()
                        })
                        .collect::<Vec<_>>();
                    let siblings = merkle_proof
                        .iter()
                        .map(|&digest| b.load_witness(digest_to_fr(digest)))
                        .collect::<Vec<_>>();
                    MerklePathWire {
                        leaf_values,
                        siblings,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let codeword_merkle_paths = whir_proof
        .codeword_opened_values
        .iter()
        .zip(whir_proof.codeword_merkle_proofs.iter())
        .map(|(values_per_query, proofs_per_query)| {
            values_per_query
                .iter()
                .zip(proofs_per_query.iter())
                .map(|(opened_values, merkle_proof)| {
                    let leaf_values = opened_values
                        .iter()
                        .map(|value| {
                            ext_to_coeffs(*value)
                                .iter()
                                .map(|&coeff| b.bb_load_reduced_witness(RootF::from_u64(coeff)))
                                .collect::<Vec<ReducedBabyBearWire<B::F>>>()
                        })
                        .collect::<Vec<_>>();
                    let siblings = merkle_proof
                        .iter()
                        .map(|&digest| b.load_witness(digest_to_fr(digest)))
                        .collect::<Vec<_>>();
                    MerklePathWire {
                        leaf_values,
                        siblings,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    WhirProofWire {
        mu_pow_witness,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        whir_sumcheck_polys,
        ood_values,
        final_poly,
        codeword_commitment_roots,
        initial_round_merkle_paths,
        codeword_merkle_paths,
    }
}

pub(crate) fn ext_to_coeffs(value: RootEF) -> [u64; BABY_BEAR_EXT_DEGREE] {
    core::array::from_fn(|i| {
        <RootEF as BasedVectorSpace<RootF>>::as_basis_coefficients_slice(&value)[i]
            .as_canonical_u64()
    })
}

fn digest_to_fr(digest: RootDigest) -> Fr {
    biguint_to_fe(&digest[0].as_canonical_biguint())
}

fn eval_mobius_eq_mle_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    u: &[BabyBearExtWire<B::F>],
    x: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert_eq!(u.len(), x.len(), "mobius-eq arity mismatch");
    let one = b.ext_from_base_const(RootF::ONE);
    let two = b.ext_from_base_const(RootF::TWO);
    let three = RootF::from_u64(3);
    let mut acc = one;
    // (1-2u)(1-x) + ux = (1-x) + u(3x-2)
    for (u_i, x_i) in u.iter().zip(x.iter()) {
        let one_minus_x = b.ext_sub(one, *x_i);
        let three_x_minus_two = b.ext_mul_base_const(*x_i, three);
        let three_x_minus_two = b.ext_sub(three_x_minus_two, two);
        let u_term = b.ext_mul(*u_i, three_x_minus_two);
        let factor = b.ext_add(one_minus_x, u_term);
        acc = b.ext_mul(acc, factor);
    }
    acc
}

fn eval_mle_evals_at_point_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    evals: &[BabyBearExtWire<B::F>],
    x: &[BabyBearExtWire<B::F>],
) -> BabyBearExtWire<B::F> {
    assert_eq!(
        evals.len(),
        1usize << x.len(),
        "MLE table length must be 2^arity",
    );
    let mut values = evals.to_vec();
    let mut len = values.len();
    for xj in x.iter().rev() {
        len >>= 1;
        for i in 0..len {
            let lo = values[i];
            let hi = values[i + len];
            let diff = b.ext_sub(hi, lo);
            let weighted = b.ext_mul(diff, *xj);
            values[i] = b.ext_add(lo, weighted);
        }
    }
    values
        .first()
        .copied()
        .expect("MLE reduction must produce one value")
}

fn invert_base_assigned<B: BabyBearInst>(
    b: &mut B,
    value: BabyBearWire<B::F>,
) -> BabyBearWire<B::F> {
    let one = b.bb_one();
    b.bb_div(one, value)
}

fn query_root_from_bits_assigned<B: BabyBearInst>(
    b: &mut B,
    query_bits: &[B::F],
    log_rs_domain_size: usize,
) -> BabyBearWire<B::F> {
    let omega = RootF::two_adic_generator(log_rs_domain_size);
    let mut root = None;
    for (bit_idx, &bit) in query_bits.iter().enumerate() {
        let omega_pow = omega.exp_u64(1u64 << bit_idx).as_canonical_u64();
        let value = b.select_const(Fr::from(omega_pow), Fr::one(), bit);
        let selected = BabyBearWire {
            value,
            max_bits: crate::field::baby_bear::BABYBEAR_MAX_BITS,
        };
        if let Some(prev) = &mut root {
            *prev = b.bb_mul(*prev, selected);
        } else {
            root = Some(selected);
        }
    }
    root.unwrap()
}

fn binary_k_fold_assigned<B: BabyBearExt4Inst>(
    b: &mut B,
    mut values: Vec<BabyBearExtWire<B::F>>,
    alphas: &[BabyBearExtWire<B::F>],
    x: BabyBearWire<B::F>,
) -> BabyBearExtWire<B::F> {
    let n = values.len();
    assert_eq!(
        n,
        1usize << alphas.len(),
        "binary-k fold value count must match 2^k",
    );
    if alphas.is_empty() {
        return values[0];
    }

    let k = alphas.len();
    let omega_k = RootF::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();
    let tw: Vec<RootF> = omega_k.powers().take(1usize << (k - 1)).collect();
    let half = RootF::ONE.halve();
    let inv_tw_half: Vec<RootF> = omega_k_inv
        .powers()
        .take(1usize << (k - 1))
        .map(|p| p * half)
        .collect();

    let mut x_pow = b.bb_reduce_max_bits(x);
    let x_inv = invert_base_assigned(b, x);
    let mut x_inv_pow = b.bb_reduce_max_bits(x_inv);

    for (j, alpha) in alphas.iter().enumerate() {
        let m = n >> (j + 1);
        for i in 0..m {
            let t = b.bb_mul_const(x_pow, tw[i << j]);
            let t_inv_half = b.bb_mul_const(x_inv_pow, inv_tw_half[i << j]);

            let lo = values[i];
            let hi = values[i + m];
            let lo_minus_hi = b.ext_sub(lo, hi);
            let mut alpha_minus_t = *alpha;
            alpha_minus_t.0[0] = b.bb_sub(alpha_minus_t.0[0], t);
            let fold = b.ext_mul(alpha_minus_t, lo_minus_hi);
            values[i] = b.ext_scalar_mul_add(fold, t_inv_half, lo);
        }
        x_pow = b.bb_square(x_pow);
        x_pow = b.bb_reduce_max_bits(x_pow);
        x_inv_pow = b.bb_square(x_inv_pow);
        x_inv_pow = b.bb_reduce_max_bits(x_inv_pow);
    }
    values[0]
}

fn tree_compress_assigned_digests<B: DigestHashInst>(b: &mut B, digests: Vec<B::F>) -> B::F {
    assert!(
        digests.len().is_power_of_two(),
        "tree_compress inputs must be power-of-two length"
    );
    let mut level = digests;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(b.compress_digests(pair[0], pair[1]));
        }
        level = next;
    }
    level
        .pop()
        .expect("tree_compress must output one digest for non-empty inputs")
}

fn constrain_merkle_path<B: GateInst + DigestHashInst>(
    b: &mut B,
    query_bits: &[B::F],
    merkle_path: &MerklePathWire<B::F>,
    root_digest: B::F,
) {
    assert!(
        merkle_path.leaf_values.len().is_power_of_two(),
        "leaf input count must be power of two"
    );

    assert_eq!(
        merkle_path.siblings.len(),
        query_bits.len(),
        "merkle path depth must match query bits",
    );

    let leaf_hashes = merkle_path
        .leaf_values
        .iter()
        .map(|leaf| b.hash_babybear_slice_to_digest(leaf))
        .collect::<Vec<_>>();

    let mut cur = tree_compress_assigned_digests(b, leaf_hashes);
    let query_bits = query_bits.to_vec();

    for (bit, &sibling) in query_bits.iter().zip(merkle_path.siblings.iter()) {
        // Select input order first, then compress once (instead of compressing
        // both orderings and selecting after). Halves Poseidon2 calls per level.
        let left = b.select(sibling, cur, *bit);
        let right = b.select(cur, sibling, *bit);
        cur = b.compress_digests(left, right);
    }

    b.constrain_equal(cur, root_digest);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_whir_verification<B: TranscriptInst + DigestHashInst>(
    b: &mut B,
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    whir_wire: &WhirProofWire<B::F>,
    stacking_openings: &[Vec<ReducedBabyBearExtWire<B::F>>],
    initial_commitment_roots: &[B::F],
    u_cube: &[BabyBearExtWire<B::F>],
    profiler: &mut CellProfiler,
) {
    let params = &mvk0.params;
    let k_whir = params.k_whir();
    let num_whir_rounds = params.num_whir_rounds();

    profiler.push("mu_pows_and_claim", b.cell_count());

    let mu_pow_witness = whir_wire.mu_pow_witness;
    b.check_witness(params.whir.mu_pow_bits, &mu_pow_witness);
    let mu_challenge = b.sample_ext();

    let folding_pow_witnesses = &whir_wire.folding_pow_witnesses;
    let query_phase_pow_witnesses = &whir_wire.query_phase_pow_witnesses;
    let whir_sumcheck_polys = &whir_wire.whir_sumcheck_polys;
    let ood_values = &whir_wire.ood_values;
    let final_poly_reduced = &whir_wire.final_poly;
    let final_poly = final_poly_reduced
        .iter()
        .map(BabyBearExtWire::from)
        .collect::<Vec<_>>();
    let codeword_commitment_roots = &whir_wire.codeword_commitment_roots;
    let codeword_commitment_digests = codeword_commitment_roots
        .iter()
        .copied()
        .map(digest_wire_from_root)
        .collect::<Vec<_>>();

    let total_width = stacking_openings.iter().map(Vec::len).sum::<usize>();
    let one = b.ext_from_base_const(RootF::ONE);
    let mut mu_pows = Vec::with_capacity(total_width);
    let mut mu_pow = one;
    for _ in 0..total_width {
        mu_pows.push(mu_pow);
        mu_pow = b.ext_mul(mu_pow, mu_challenge);
        mu_pow = b.ext_reduce_max_bits(mu_pow);
    }

    let mut final_claim = b.ext_zero();
    let mut mu_idx = 0usize;
    for commit_openings in stacking_openings {
        for opening in commit_openings {
            let weighted = if mu_idx == 0 {
                (*opening).into()
            } else {
                b.ext_mul(opening.into(), mu_pows[mu_idx])
            };
            final_claim = b.ext_add(final_claim, weighted);
            mu_idx += 1;
        }
    }

    profiler.pop(b.cell_count());

    let mut folding_alphas = Vec::new();
    let mut z0_challenges = Vec::new();
    let mut gammas = Vec::with_capacity(num_whir_rounds);
    let mut query_indices = Vec::new();
    let mut folding_counts_per_round = Vec::with_capacity(num_whir_rounds);
    let mut query_counts_per_round = Vec::with_capacity(num_whir_rounds);
    let mut query_index_bits = Vec::new();
    let mut zs_per_round = Vec::with_capacity(num_whir_rounds);

    let mut sumcheck_cursor = 0usize;
    let mut folding_pow_cursor = 0usize;
    let mut log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;

    for (round_idx, round_params) in params.whir.rounds.iter().enumerate() {
        let round_label = format!("round_{round_idx}");
        profiler.push(&round_label, b.cell_count());

        let is_initial_round = round_idx == 0;
        let is_final_round = round_idx + 1 == num_whir_rounds;
        let mut alphas_round = Vec::new();

        profiler.push("sumcheck", b.cell_count());
        for _ in 0..k_whir {
            if let Some(evals) = whir_sumcheck_polys.get(sumcheck_cursor) {
                let ev1 = evals[0];
                let ev2 = evals[1];
                b.observe_ext(&ev1);
                b.observe_ext(&ev2);

                let pow_witness = folding_pow_witnesses[folding_pow_cursor];
                folding_pow_cursor += 1;
                b.check_witness(params.whir.folding_pow_bits, &pow_witness);

                let alpha = b.sample_ext();
                alphas_round.push(alpha);
                folding_alphas.push(alpha);

                let ev1 = ev1.into();
                let ev2 = ev2.into();
                let ev0 = b.ext_sub(final_claim, ev1);
                final_claim = interpolate_quadratic_at_012_assigned(b, [&ev0, &ev1, &ev2], &alpha);
                sumcheck_cursor += 1;
            }
        }
        folding_counts_per_round.push(alphas_round.len());
        profiler.pop(b.cell_count());

        let y0 = if is_final_round {
            for coeff in final_poly_reduced {
                b.observe_ext(coeff);
            }
            None
        } else {
            b.observe_commit(&codeword_commitment_digests[round_idx]);
            let z0 = b.sample_ext();
            z0_challenges.push(z0);

            let y0 = ood_values[round_idx];
            b.observe_ext(&y0);
            Some(y0)
        };

        b.check_witness(
            params.whir.query_phase_pow_bits,
            &query_phase_pow_witnesses[round_idx],
        );

        let query_bits = log_rs_domain_size - k_whir;
        let num_queries = round_params.num_queries;
        query_counts_per_round.push(num_queries);

        let mut ys_round = Vec::with_capacity(num_queries);
        let mut zs_round = Vec::with_capacity(num_queries);

        profiler.push("queries", b.cell_count());
        for query_idx in 0..num_queries {
            let query_index = b.sample_bits(query_bits);
            query_index_bits.push(query_bits);
            query_indices.push(query_index);
            let query_bits_vec = if query_bits == 0 {
                Vec::new()
            } else {
                b.num_to_bits(query_index, query_bits)
            };
            let zi_root = query_root_from_bits_assigned(b, &query_bits_vec, log_rs_domain_size);
            let zi = b.bb_pow_power_of_two(zi_root, k_whir);

            let yi = if is_initial_round {
                let mut codeword_vals = vec![None; 1usize << k_whir];
                let mut mu_power_idx = 0usize;
                for (commit_idx, commit_openings) in stacking_openings.iter().enumerate() {
                    let merkle_path = &whir_wire.initial_round_merkle_paths[commit_idx][query_idx];
                    constrain_merkle_path(
                        b,
                        &query_bits_vec,
                        merkle_path,
                        initial_commitment_roots[commit_idx],
                    );
                    for col_idx in 0..commit_openings.len() {
                        let mu_pow = mu_pows[mu_power_idx];
                        let is_first_mu = mu_power_idx == 0;
                        for (row_idx, row) in merkle_path.leaf_values.iter().enumerate() {
                            let opened_base = row[col_idx].into();
                            codeword_vals[row_idx] = if let Some(prev) = codeword_vals[row_idx] {
                                Some(b.ext_scalar_mul_add(mu_pow, opened_base, prev))
                            } else if is_first_mu {
                                Some(b.ext_from_base_var(opened_base))
                            } else {
                                Some(b.ext_scalar_mul(mu_pow, opened_base))
                            };
                        }
                        mu_power_idx += 1;
                    }
                }

                let codeword_vals = codeword_vals.into_iter().flatten().collect::<Vec<_>>();
                binary_k_fold_assigned(b, codeword_vals, &alphas_round, zi_root)
            } else {
                let merkle_path = &whir_wire.codeword_merkle_paths[round_idx - 1][query_idx];
                constrain_merkle_path(
                    b,
                    &query_bits_vec,
                    merkle_path,
                    codeword_commitment_roots[round_idx - 1],
                );

                let opened_values = merkle_path
                    .leaf_values
                    .iter()
                    .map(|row| BabyBearExt4Wire(core::array::from_fn(|idx| row[idx].into())))
                    .collect::<Vec<_>>();
                binary_k_fold_assigned(b, opened_values, &alphas_round, zi_root)
            };

            zs_round.push(zi);
            ys_round.push(yi);
        }
        profiler.pop(b.cell_count());

        profiler.push("gamma_accumulation", b.cell_count());
        let gamma = b.sample_ext();
        if let Some(y0) = y0 {
            let y0_term = b.ext_mul(y0.into(), gamma);
            final_claim = b.ext_add(final_claim, y0_term);
        }
        let mut gamma_pow = b.ext_mul(gamma, gamma);
        for yi in &ys_round {
            let term = b.ext_mul(*yi, gamma_pow);
            final_claim = b.ext_add(final_claim, term);
            gamma_pow = b.ext_mul(gamma_pow, gamma);
        }

        gammas.push(gamma);
        profiler.pop(b.cell_count());

        zs_per_round.push(zs_round);
        log_rs_domain_size = log_rs_domain_size.saturating_sub(1);
        profiler.pop(b.cell_count());
    }

    profiler.push("final_verification", b.cell_count());
    let rounds = query_counts_per_round.len();
    let t = k_whir * rounds;

    profiler.push("eq_mle_prefix_suffix", b.cell_count());
    let prefix = eval_mobius_eq_mle_assigned(b, &u_cube[..t], &folding_alphas[..t]);
    let suffix = eval_mle_evals_at_point_assigned(b, &final_poly, &u_cube[t..]);
    let mut final_acc = b.ext_mul(prefix, suffix);
    profiler.pop(b.cell_count());

    let mut alpha_offset = k_whir;
    for round_idx in 0..rounds {
        let final_round_label = format!("final_round_{round_idx}");
        profiler.push(&final_round_label, b.cell_count());

        let gamma = &gammas[round_idx];
        let alpha_slc = &folding_alphas[alpha_offset..t];
        let slc_len = (t - alpha_offset) + 1;

        if round_idx + 1 != rounds {
            profiler.push("z0_ood_eval", b.cell_count());
            let z0 = &z0_challenges[round_idx];
            let mut z0_pows = Vec::with_capacity(slc_len);
            z0_pows.push(*z0);
            for _ in 1..slc_len {
                let prev = *z0_pows.last().unwrap();
                let next = b.ext_square(prev);
                z0_pows.push(next);
            }
            let z0_max = *z0_pows.last().unwrap();
            // Pre-reduce z0_pows so eq_mle doesn't redundantly reduce them.
            let z0_pows_reduced: Vec<_> =
                z0_pows.iter().map(|p| b.ext_reduce_max_bits(*p)).collect();
            let eq = eval_eq_mle_assigned(
                b,
                alpha_slc,
                &z0_pows_reduced[..z0_pows_reduced.len().saturating_sub(1)],
            );
            let poly_eval = horner_eval_ext_poly_assigned(b, &final_poly, &z0_max);
            let term = b.ext_mul(*gamma, eq);
            let term = b.ext_mul(term, poly_eval);
            final_acc = b.ext_add(final_acc, term);
            profiler.pop(b.cell_count());
        }

        profiler.push("query_point_evals", b.cell_count());
        let mut gamma_pow = b.ext_mul(*gamma, *gamma);
        for zi in zs_per_round[round_idx].iter() {
            let mut zi_pows = Vec::with_capacity(slc_len);
            zi_pows.push(*zi);
            for _ in 1..slc_len {
                let prev = *zi_pows.last().unwrap();
                let next = b.bb_square(prev);
                zi_pows.push(next);
            }
            // Pre-reduce zi_pows so eq_mle and poly eval don't redundantly reduce.
            let zi_pows_reduced: Vec<_> =
                zi_pows.iter().map(|p| b.bb_reduce_max_bits(*p)).collect();

            let eq = eval_eq_mle_ef_f_assigned(
                b,
                alpha_slc,
                &zi_pows_reduced[..zi_pows_reduced.len().saturating_sub(1)],
            );

            let poly_eval =
                horner_eval_ext_poly_f_assigned(b, &final_poly, zi_pows_reduced.last().unwrap());

            let term = b.ext_mul(gamma_pow, eq);
            let term = b.ext_mul(term, poly_eval);
            final_acc = b.ext_add(final_acc, term);
            gamma_pow = b.ext_mul(gamma_pow, *gamma);
        }
        profiler.pop(b.cell_count());

        profiler.pop(b.cell_count());
        alpha_offset += k_whir;
    }

    b.ext_assert_equal(final_acc, final_claim);
    profiler.pop(b.cell_count());
}
