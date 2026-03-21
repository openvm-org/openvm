use halo2_base::{
    gates::{GateInstructions, RangeInstructions},
    utils::biguint_to_fe,
    AssignedValue, Context, QuantumCell,
};
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
    field::baby_bear::{
        BabyBearChip, BabyBearExt4Wire, BabyBearExtChip, BabyBearExtWire, BabyBearWire,
        BABY_BEAR_EXT_DEGREE,
    },
    hash::poseidon2::{compress_bn254_digests, hash_babybear_slice_to_digest},
    stages::{
        batch_constraints::{eval_eq_mle_assigned, eval_eq_mle_ef_f_assigned},
        shared_math::{
            horner_eval_ext_poly_assigned, horner_eval_ext_poly_f_assigned,
            interpolate_quadratic_at_012_assigned,
        },
    },
    transcript::{digest_wire_from_root, TranscriptGadget},
    Fr, RootEF, RootF,
};

#[derive(Clone, Debug)]
pub struct MerklePathWire {
    pub leaf_values: Vec<Vec<BabyBearWire>>,
    pub siblings: Vec<AssignedValue<Fr>>,
}

#[derive(Clone, Debug)]
pub struct WhirProofWire {
    pub mu_pow_witness: BabyBearWire,
    pub folding_pow_witnesses: Vec<BabyBearWire>,
    pub query_phase_pow_witnesses: Vec<BabyBearWire>,
    pub whir_sumcheck_polys: Vec<Vec<BabyBearExtWire>>,
    pub ood_values: Vec<BabyBearExtWire>,
    pub final_poly: Vec<BabyBearExtWire>,
    pub codeword_commitment_roots: Vec<AssignedValue<Fr>>,
    pub initial_round_merkle_paths: Vec<Vec<MerklePathWire>>,
    pub codeword_merkle_paths: Vec<Vec<MerklePathWire>>,
}

pub(crate) fn load_whir_proof_wire(
    ctx: &mut Context<Fr>,
    base_chip: &crate::field::baby_bear::BabyBearChip,
    ext_chip: &BabyBearExtChip,
    whir_proof: &WhirProof<RootConfig>,
) -> WhirProofWire {
    let mu_pow_witness = base_chip.load_witness(ctx, whir_proof.mu_pow_witness);
    let folding_pow_witnesses = whir_proof
        .folding_pow_witnesses
        .iter()
        .map(|&witness| base_chip.load_witness(ctx, RootF::from_u64(witness.as_canonical_u64())))
        .collect::<Vec<_>>();
    let query_phase_pow_witnesses = whir_proof
        .query_phase_pow_witnesses
        .iter()
        .map(|&witness| base_chip.load_witness(ctx, RootF::from_u64(witness.as_canonical_u64())))
        .collect::<Vec<_>>();
    let whir_sumcheck_polys = whir_proof
        .whir_sumcheck_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| ext_chip.load_witness(ctx, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let ood_values = whir_proof
        .ood_values
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let final_poly = whir_proof
        .final_poly
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let codeword_commitment_roots = whir_proof
        .codeword_commits
        .iter()
        .map(|&digest| ctx.load_witness(digest_to_fr(digest)))
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
                                .map(|&value| base_chip.load_witness(ctx, value))
                                .collect::<Vec<BabyBearWire>>()
                        })
                        .collect::<Vec<_>>();
                    let siblings = merkle_proof
                        .iter()
                        .map(|&digest| ctx.load_witness(digest_to_fr(digest)))
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
                                .map(|&coeff| base_chip.load_witness(ctx, RootF::from_u64(coeff)))
                                .collect::<Vec<BabyBearWire>>()
                        })
                        .collect::<Vec<_>>();
                    let siblings = merkle_proof
                        .iter()
                        .map(|&digest| ctx.load_witness(digest_to_fr(digest)))
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

fn eval_mobius_eq_mle_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    u: &[BabyBearExtWire],
    x: &[BabyBearExtWire],
) -> BabyBearExtWire {
    assert_eq!(u.len(), x.len(), "mobius-eq arity mismatch");
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut acc = one;
    for (u_i, x_i) in u.iter().zip(x.iter()) {
        let two_u = ext_chip.mul_base_const(ctx, *u_i, RootF::TWO);
        let w0 = ext_chip.sub(ctx, one, two_u);
        let one_minus_x = ext_chip.sub(ctx, one, *x_i);
        let left = ext_chip.mul(ctx, w0, one_minus_x);
        let right = ext_chip.mul(ctx, *u_i, *x_i);
        let factor = ext_chip.add(ctx, left, right);
        acc = ext_chip.mul(ctx, acc, factor);
    }
    acc
}

fn eval_mle_evals_at_point_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    evals: &[BabyBearExtWire],
    x: &[BabyBearExtWire],
) -> BabyBearExtWire {
    assert_eq!(
        evals.len(),
        1usize << x.len(),
        "MLE table length must be 2^arity",
    );
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut values = evals.to_vec();
    let mut len = values.len();
    for xj in x.iter().rev() {
        len >>= 1;
        let one_minus_xj = ext_chip.sub(ctx, one, *xj);
        for i in 0..len {
            let lo = values[i];
            let hi = values[i + len];
            let lo_term = ext_chip.mul(ctx, lo, one_minus_xj);
            let hi_term = ext_chip.mul(ctx, hi, *xj);
            values[i] = ext_chip.add(ctx, lo_term, hi_term);
        }
    }
    values
        .first()
        .copied()
        .expect("MLE reduction must produce one value")
}

fn invert_base_assigned(
    ctx: &mut Context<Fr>,
    base_chip: &BabyBearChip,
    value: BabyBearWire,
) -> BabyBearWire {
    let one = base_chip.one(ctx);
    base_chip.div(ctx, one, value)
}

fn query_root_from_bits_assigned(
    ctx: &mut Context<Fr>,
    base_chip: &BabyBearChip,
    query_bits: &[AssignedValue<Fr>],
    log_rs_domain_size: usize,
) -> BabyBearWire {
    let gate = base_chip.range().gate();
    let omega = RootF::two_adic_generator(log_rs_domain_size);
    let mut root = None;
    for (bit_idx, &bit) in query_bits.iter().enumerate() {
        let omega_pow = omega.exp_u64(1u64 << bit_idx).as_canonical_u64();
        let value = gate.select(
            ctx,
            QuantumCell::Constant(Fr::from(omega_pow)),
            QuantumCell::Constant(Fr::one()),
            bit,
        );
        let selected = BabyBearWire {
            value,
            max_bits: crate::field::baby_bear::BABYBEAR_MAX_BITS,
        };
        if let Some(prev) = &mut root {
            *prev = base_chip.mul(ctx, *prev, selected);
        } else {
            root = Some(selected);
        }
    }
    root.unwrap()
}

fn binary_k_fold_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    mut values: Vec<BabyBearExtWire>,
    alphas: &[BabyBearExtWire],
    x: BabyBearWire,
) -> BabyBearExtWire {
    let base_chip = ext_chip.base();

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
    let tw = omega_k.powers().take(1usize << (k - 1)).collect();
    let inv_tw = omega_k_inv.powers().take(1usize << (k - 1)).collect();
    let half = RootF::ONE.halve();

    let mut x_pow = base_chip.reduce_max_bits(ctx, x);
    let x_inv = invert_base_assigned(ctx, base_chip, x);
    let mut x_inv_pow = base_chip.reduce_max_bits(ctx, x_inv);

    for (j, alpha) in alphas.iter().enumerate() {
        let m = n >> (j + 1);
        for i in 0..m {
            let t = base_chip.mul_const(ctx, x_pow, tw[i << j]);
            let t_inv = base_chip.mul_const(ctx, x_inv_pow, inv_tw[i << j]);
            let t_inv_half = base_chip.mul_const(ctx, t_inv, half);

            let lo = values[i];
            let hi = values[i + m];
            let lo_minus_hi = ext_chip.sub(ctx, lo, hi);
            let mut alpha_minus_t = *alpha;
            alpha_minus_t.0[0] = base_chip.sub(ctx, alpha_minus_t.0[0], t);
            let fold = ext_chip.mul(ctx, alpha_minus_t, lo_minus_hi);
            let fold = ext_chip.scalar_mul(ctx, fold, t_inv_half);
            values[i] = ext_chip.add(ctx, lo, fold);
        }
        x_pow = base_chip.square(ctx, x_pow);
        x_pow = base_chip.reduce_max_bits(ctx, x_pow);
        x_inv_pow = base_chip.square(ctx, x_inv_pow);
        x_inv_pow = base_chip.reduce_max_bits(ctx, x_inv_pow);
    }
    values[0]
}

fn tree_compress_assigned_digests(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    digests: Vec<AssignedValue<Fr>>,
) -> AssignedValue<Fr> {
    assert!(
        digests.len().is_power_of_two(),
        "tree_compress inputs must be power-of-two length"
    );
    let mut level = digests;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(compress_bn254_digests(
                ctx,
                ext_chip.range(),
                pair[0],
                pair[1],
            ));
        }
        level = next;
    }
    level
        .pop()
        .expect("tree_compress must output one digest for non-empty inputs")
}

fn constrain_merkle_path(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    query_bits: &[AssignedValue<Fr>],
    merkle_path: &MerklePathWire,
    root_digest: AssignedValue<Fr>,
) {
    assert!(
        merkle_path.leaf_values.len().is_power_of_two(),
        "leaf input count must be power of two"
    );

    let gate = ext_chip.range().gate();

    assert_eq!(
        merkle_path.siblings.len(),
        query_bits.len(),
        "merkle path depth must match query bits",
    );

    let leaf_hashes = merkle_path
        .leaf_values
        .iter()
        .map(|leaf| hash_babybear_slice_to_digest(ctx, ext_chip.range(), leaf))
        .collect::<Vec<_>>();

    let mut cur = tree_compress_assigned_digests(ctx, ext_chip, leaf_hashes);
    let query_bits = query_bits.to_vec();

    for (bit, &sibling) in query_bits.iter().zip(merkle_path.siblings.iter()) {
        // Select input order first, then compress once (instead of compressing
        // both orderings and selecting after). Halves Poseidon2 calls per level.
        let left = gate.select(ctx, sibling, cur, *bit);
        let right = gate.select(ctx, cur, sibling, *bit);
        cur = compress_bn254_digests(ctx, ext_chip.range(), left, right);
    }

    ctx.constrain_equal(&cur, &root_digest);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_whir_verification(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    transcript: &mut TranscriptGadget,
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    whir_wire: &WhirProofWire,
    stacking_openings: &[Vec<BabyBearExtWire>],
    initial_commitment_roots: &[AssignedValue<Fr>],
    u_cube: &[BabyBearExtWire],
) {
    let mut profiler =
        crate::profiling::CellProfiler::new("whir_verification", ctx.advice.len());

    let gate = ext_chip.range().gate();
    let base_chip = ext_chip.base();
    let params = &mvk0.params;
    let k_whir = params.k_whir();
    let num_whir_rounds = params.num_whir_rounds();

    profiler.push("mu_pows_and_claim", ctx.advice.len());

    let mu_pow_witness = whir_wire.mu_pow_witness;
    transcript.check_witness(ctx, base_chip, params.whir.mu_pow_bits, &mu_pow_witness);
    let mu_challenge = transcript.sample_ext(ctx, ext_chip.base());

    let folding_pow_witnesses = &whir_wire.folding_pow_witnesses;
    let query_phase_pow_witnesses = &whir_wire.query_phase_pow_witnesses;
    let whir_sumcheck_polys = &whir_wire.whir_sumcheck_polys;
    let ood_values = &whir_wire.ood_values;
    let final_poly = &whir_wire.final_poly;
    let codeword_commitment_roots = &whir_wire.codeword_commitment_roots;
    let codeword_commitment_digests = codeword_commitment_roots
        .iter()
        .copied()
        .map(digest_wire_from_root)
        .collect::<Vec<_>>();

    let total_width = stacking_openings.iter().map(Vec::len).sum::<usize>();
    let one = ext_chip.from_base_const(ctx, RootF::ONE);
    let mut mu_pows = Vec::with_capacity(total_width);
    let mut mu_pow = one;
    for _ in 0..total_width {
        mu_pows.push(mu_pow);
        mu_pow = ext_chip.mul(ctx, mu_pow, mu_challenge);
    }

    let mut final_claim = ext_chip.zero(ctx);
    let mut mu_idx = 0usize;
    for commit_openings in stacking_openings {
        for opening in commit_openings {
            let weighted = ext_chip.mul(ctx, *opening, mu_pows[mu_idx]);
            final_claim = ext_chip.add(ctx, final_claim, weighted);
            mu_idx += 1;
        }
    }

    profiler.pop(ctx.advice.len());

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
        profiler.push(&round_label, ctx.advice.len());

        let is_initial_round = round_idx == 0;
        let is_final_round = round_idx + 1 == num_whir_rounds;
        let mut alphas_round = Vec::new();

        profiler.push("sumcheck", ctx.advice.len());
        for _ in 0..k_whir {
            if let Some(evals) = whir_sumcheck_polys.get(sumcheck_cursor) {
                let ev1 = evals[0];
                let ev2 = evals[1];
                transcript.observe_ext(ctx, ext_chip.base(), &ev1);
                transcript.observe_ext(ctx, ext_chip.base(), &ev2);

                let pow_witness = folding_pow_witnesses[folding_pow_cursor];
                folding_pow_cursor += 1;
                transcript.check_witness(
                    ctx,
                    base_chip,
                    params.whir.folding_pow_bits,
                    &pow_witness,
                );

                let alpha = transcript.sample_ext(ctx, ext_chip.base());
                alphas_round.push(alpha);
                folding_alphas.push(alpha);

                let ev0 = ext_chip.sub(ctx, final_claim, ev1);
                final_claim = interpolate_quadratic_at_012_assigned(
                    ctx,
                    ext_chip,
                    [&ev0, &ev1, &ev2],
                    &alpha,
                );
                sumcheck_cursor += 1;
            }
        }
        folding_counts_per_round.push(alphas_round.len());
        profiler.pop(ctx.advice.len());

        let y0 = if is_final_round {
            for coeff in final_poly {
                transcript.observe_ext(ctx, ext_chip.base(), coeff);
            }
            None
        } else {
            transcript.observe_commit(
                ctx,
                ext_chip.base(),
                &codeword_commitment_digests[round_idx],
            );
            let z0 = transcript.sample_ext(ctx, ext_chip.base());
            z0_challenges.push(z0);

            let y0 = ood_values[round_idx];
            transcript.observe_ext(ctx, ext_chip.base(), &y0);
            Some(y0)
        };

        transcript.check_witness(
            ctx,
            base_chip,
            params.whir.query_phase_pow_bits,
            &query_phase_pow_witnesses[round_idx],
        );

        let query_bits = log_rs_domain_size - k_whir;
        let num_queries = round_params.num_queries;
        query_counts_per_round.push(num_queries);

        let mut ys_round = Vec::with_capacity(num_queries);
        let mut zs_round = Vec::with_capacity(num_queries);

        profiler.push("queries", ctx.advice.len());
        for query_idx in 0..num_queries {
            let query_index = transcript.sample_bits(ctx, ext_chip.base(), query_bits);
            query_index_bits.push(query_bits);
            query_indices.push(query_index);
            let query_bits_vec = if query_bits == 0 {
                Vec::new()
            } else {
                gate.num_to_bits(ctx, query_index, query_bits)
            };
            let zi_root = query_root_from_bits_assigned(
                ctx,
                ext_chip.base(),
                &query_bits_vec,
                log_rs_domain_size,
            );
            let zi = ext_chip.base().pow_power_of_two(ctx, zi_root, k_whir);

            let yi = if is_initial_round {
                let mut codeword_vals = vec![None; 1usize << k_whir];
                let mut mu_power_idx = 0usize;
                for (commit_idx, commit_openings) in stacking_openings.iter().enumerate() {
                    let merkle_path = &whir_wire.initial_round_merkle_paths[commit_idx][query_idx];
                    constrain_merkle_path(
                        ctx,
                        ext_chip,
                        &query_bits_vec,
                        merkle_path,
                        initial_commitment_roots[commit_idx],
                    );
                    for col_idx in 0..commit_openings.len() {
                        let mu_pow = mu_pows[mu_power_idx];
                        for (row_idx, row) in merkle_path.leaf_values.iter().enumerate() {
                            let opened_base = row[col_idx];
                            let opened_ext = ext_chip.from_base_var(ctx, opened_base);
                            let weighted = ext_chip.mul(ctx, opened_ext, mu_pow);
                            codeword_vals[row_idx] = if let Some(prev) = codeword_vals[row_idx] {
                                Some(ext_chip.add(ctx, prev, weighted))
                            } else {
                                Some(weighted)
                            };
                        }
                        mu_power_idx += 1;
                    }
                }

                let codeword_vals = codeword_vals.into_iter().flatten().collect::<Vec<_>>();
                binary_k_fold_assigned(ctx, ext_chip, codeword_vals, &alphas_round, zi_root)
            } else {
                let merkle_path = &whir_wire.codeword_merkle_paths[round_idx - 1][query_idx];
                constrain_merkle_path(
                    ctx,
                    ext_chip,
                    &query_bits_vec,
                    merkle_path,
                    codeword_commitment_roots[round_idx - 1],
                );

                let opened_values = merkle_path
                    .leaf_values
                    .iter()
                    .map(|row| BabyBearExt4Wire(core::array::from_fn(|idx| row[idx])))
                    .collect::<Vec<_>>();
                binary_k_fold_assigned(ctx, ext_chip, opened_values, &alphas_round, zi_root)
            };

            zs_round.push(zi);
            ys_round.push(yi);
        }
        profiler.pop(ctx.advice.len());

        profiler.push("gamma_accumulation", ctx.advice.len());
        let gamma = transcript.sample_ext(ctx, ext_chip.base());
        if let Some(y0) = y0 {
            let y0_term = ext_chip.mul(ctx, y0, gamma);
            final_claim = ext_chip.add(ctx, final_claim, y0_term);
        }
        let mut gamma_pow = ext_chip.mul(ctx, gamma, gamma);
        for yi in &ys_round {
            let term = ext_chip.mul(ctx, *yi, gamma_pow);
            final_claim = ext_chip.add(ctx, final_claim, term);
            gamma_pow = ext_chip.mul(ctx, gamma_pow, gamma);
        }

        gammas.push(gamma);
        profiler.pop(ctx.advice.len());

        zs_per_round.push(zs_round);
        log_rs_domain_size = log_rs_domain_size.saturating_sub(1);
        profiler.pop(ctx.advice.len());
    }

    profiler.push("final_verification", ctx.advice.len());
    let rounds = query_counts_per_round.len();
    let t = k_whir * rounds;

    profiler.push("eq_mle_prefix_suffix", ctx.advice.len());
    let prefix = eval_mobius_eq_mle_assigned(ctx, ext_chip, &u_cube[..t], &folding_alphas[..t]);
    let suffix = eval_mle_evals_at_point_assigned(ctx, ext_chip, final_poly, &u_cube[t..]);
    let mut final_acc = ext_chip.mul(ctx, prefix, suffix);
    profiler.pop(ctx.advice.len());

    let mut alpha_offset = k_whir;
    for round_idx in 0..rounds {
        let final_round_label = format!("final_round_{round_idx}");
        profiler.push(&final_round_label, ctx.advice.len());

        let gamma = &gammas[round_idx];
        let alpha_slc = &folding_alphas[alpha_offset..t];
        let slc_len = (t - alpha_offset) + 1;

        if round_idx + 1 != rounds {
            profiler.push("z0_ood_eval", ctx.advice.len());
            let z0 = &z0_challenges[round_idx];
            let mut z0_pows = Vec::with_capacity(slc_len);
            z0_pows.push(*z0);
            for _ in 1..slc_len {
                let prev = z0_pows.last().unwrap();
                let next = ext_chip.square(ctx, *prev);
                z0_pows.push(next);
            }
            let z0_max = *z0_pows.last().unwrap();
            // Pre-reduce z0_pows so eq_mle doesn't redundantly reduce them.
            let z0_pows_reduced: Vec<_> = z0_pows
                .iter()
                .map(|p| ext_chip.reduce_max_bits(ctx, *p))
                .collect();
            let eq = eval_eq_mle_assigned(
                ctx,
                ext_chip,
                alpha_slc,
                &z0_pows_reduced[..z0_pows_reduced.len().saturating_sub(1)],
            );
            let poly_eval = horner_eval_ext_poly_assigned(ctx, ext_chip, final_poly, &z0_max);
            let term = ext_chip.mul(ctx, *gamma, eq);
            let term = ext_chip.mul(ctx, term, poly_eval);
            final_acc = ext_chip.add(ctx, final_acc, term);
            profiler.pop(ctx.advice.len());
        }

        profiler.push("query_point_evals", ctx.advice.len());
        let mut gamma_pow = ext_chip.mul(ctx, *gamma, *gamma);
        for zi in zs_per_round[round_idx].iter() {
            let mut zi_pows = Vec::with_capacity(slc_len);
            zi_pows.push(*zi);
            for _ in 1..slc_len {
                let prev = zi_pows.last().unwrap();
                let next = ext_chip.base().square(ctx, *prev);
                zi_pows.push(next);
            }
            // Pre-reduce zi_pows so eq_mle and poly eval don't redundantly reduce.
            let zi_pows_reduced: Vec<_> = zi_pows
                .iter()
                .map(|p| ext_chip.base().reduce_max_bits(ctx, *p))
                .collect();

            let eq = eval_eq_mle_ef_f_assigned(
                ctx,
                ext_chip,
                alpha_slc,
                &zi_pows_reduced[..zi_pows_reduced.len().saturating_sub(1)],
            );

            let poly_eval = horner_eval_ext_poly_f_assigned(
                ctx,
                ext_chip,
                final_poly,
                zi_pows_reduced.last().unwrap(),
            );

            let term = ext_chip.mul(ctx, gamma_pow, eq);
            let term = ext_chip.mul(ctx, term, poly_eval);
            final_acc = ext_chip.add(ctx, final_acc, term);
            gamma_pow = ext_chip.mul(ctx, gamma_pow, *gamma);
        }
        profiler.pop(ctx.advice.len());

        profiler.pop(ctx.advice.len());
        alpha_offset += k_whir;
    }

    let final_residual = ext_chip.sub(ctx, final_acc, final_claim);
    let zero = ext_chip.zero(ctx);
    ext_chip.assert_equal(ctx, final_residual, zero);
    profiler.pop(ctx.advice.len());

    #[cfg(feature = "cell-profiling")]
    if let Ok(dir) = std::env::var("OPENVM_PROFILE_DIR") {
        let _ = std::fs::create_dir_all(&dir);
        profiler.write_flamegraph(
            &format!("{dir}/whir_verification.svg"),
            "WHIR Verification",
            ctx.advice.len(),
        );
        profiler.write_flamegraph_reversed(
            &format!("{dir}/whir_verification_rev.svg"),
            "WHIR Verification (reversed)",
            ctx.advice.len(),
        );
    }
}
