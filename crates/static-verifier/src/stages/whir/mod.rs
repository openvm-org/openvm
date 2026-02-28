use core::iter::zip;

use halo2_base::{
    AssignedValue, Context,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
    utils::biguint_to_fe,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Digest as NativeDigest, EF as NativeEF,
        F as NativeF, Transcript as NativeTranscript, default_transcript,
    },
    openvm_stark_backend::{
        FiatShamirTranscript, StarkProtocolConfig,
        hasher::MerkleHasher,
        keygen::types::MultiStarkVerifyingKey,
        p3_field::{
            BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField, PrimeField64,
            TwoAdicField,
        },
        poly_common::{
            Squarable, eval_eq_mle, eval_mle_evals_at_point, eval_mobius_eq_mle, horner_eval,
            interpolate_quadratic_at_012,
        },
        proof::{Proof, WhirProof},
        verifier::{
            batch_constraints::BatchConstraintError as NativeBatchConstraintError,
            proof_shape::ProofShapeError,
            stacked_reduction::StackedReductionError,
            whir::{VerifyWhirError, binary_k_fold, merkle_verify},
        },
    },
};

use crate::{
    circuit::Fr,
    gadgets::{
        baby_bear::{BABY_BEAR_EXT_DEGREE, BabyBearArithmeticGadgets, BabyBearExtVar, BabyBearVar},
        transcript::{compress_bn254_digests, hash_babybear_slice_to_digest},
    },
    stages::batch_constraints::{
        BatchConstraintError, eval_eq_mle_assigned, ext_from_base_const, ext_mul_base_const,
        ext_pow_power_of_two,
    },
    stages::stacked_reduction::{
        coeffs_to_native_ext as stacked_coeffs_to_native_ext,
        derive_stacked_reduction_intermediates_with_inputs,
    },
    stages::{
        pipeline::{
            collect_trace_commitments, derive_u_cube_from_prism, prepare_pipeline_inputs,
        },
        shared_math::{horner_eval_ext_poly_assigned, interpolate_quadratic_at_012_assigned},
    },
    utils::{assign_and_range_u64, assign_and_range_usize, usize_to_u64},
};

#[derive(Debug, PartialEq, Eq)]
pub enum WhirError {
    SystemParamsMismatch,
    TraceHeightsTooLarge,
    ProofShape(ProofShapeError),
    BatchConstraint(NativeBatchConstraintError<NativeEF>),
    StackedReduction(StackedReductionError<NativeEF>),
    Whir(VerifyWhirError),
    BatchSetup(BatchConstraintError),
}

impl From<ProofShapeError> for WhirError {
    fn from(value: ProofShapeError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<NativeBatchConstraintError<NativeEF>> for WhirError {
    fn from(value: NativeBatchConstraintError<NativeEF>) -> Self {
        Self::BatchConstraint(value)
    }
}

impl From<StackedReductionError<NativeEF>> for WhirError {
    fn from(value: StackedReductionError<NativeEF>) -> Self {
        Self::StackedReduction(value)
    }
}

impl From<VerifyWhirError> for WhirError {
    fn from(value: VerifyWhirError) -> Self {
        Self::Whir(value)
    }
}

impl From<BatchConstraintError> for WhirError {
    fn from(value: BatchConstraintError) -> Self {
        match value {
            BatchConstraintError::SystemParamsMismatch => Self::SystemParamsMismatch,
            BatchConstraintError::TraceHeightsTooLarge => Self::TraceHeightsTooLarge,
            BatchConstraintError::ProofShape(err) => Self::ProofShape(err),
            BatchConstraintError::BatchConstraint(err) => Self::BatchConstraint(err),
            _ => Self::BatchSetup(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhirIntermediates {
    pub k_whir: usize,
    pub initial_log_rs_domain_size: usize,
    pub mu_pow_bits: usize,
    pub mu_pow_witness: u64,
    pub mu_pow_sampled_bits: u64,
    pub mu_pow_witness_ok: bool,
    pub mu_challenge: [u64; BABY_BEAR_EXT_DEGREE],
    pub folding_pow_bits: usize,
    pub folding_pow_witnesses: Vec<u64>,
    pub folding_pow_sampled_bits: Vec<u64>,
    pub folding_pow_witness_ok: Vec<bool>,
    pub folding_alphas: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub z0_challenges: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub query_phase_pow_bits: usize,
    pub query_phase_pow_witnesses: Vec<u64>,
    pub query_phase_pow_sampled_bits: Vec<u64>,
    pub query_phase_pow_witness_ok: Vec<bool>,
    pub gammas: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub folding_counts_per_round: Vec<usize>,
    pub query_counts_per_round: Vec<usize>,
    pub query_index_bits: Vec<usize>,
    pub query_indices: Vec<u64>,
    pub initial_commitment_roots: Vec<Fr>,
    pub codeword_commitment_roots: Vec<Fr>,
    pub merkle_paths: Vec<MerklePathIntermediates>,
    pub final_poly_len: usize,
    pub expected_final_poly_len: usize,
    pub stacking_openings: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub whir_sumcheck_polys: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub ood_values: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub final_poly: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub u_cube: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub final_claim: [u64; BABY_BEAR_EXT_DEGREE],
    pub final_acc: [u64; BABY_BEAR_EXT_DEGREE],
    pub final_residual: [u64; BABY_BEAR_EXT_DEGREE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerklePathIntermediates {
    pub query_position: usize,
    pub query_index_bits: usize,
    pub leaf_inputs: Vec<Vec<u64>>,
    pub root_digest: Fr,
    pub siblings: Vec<Fr>,
}

#[derive(Clone, Debug)]
pub struct AssignedWhirIntermediates {
    pub mu_pow_bits: AssignedValue<Fr>,
    pub mu_pow_sampled_bits: AssignedValue<Fr>,
    pub mu_pow_witness_ok: AssignedValue<Fr>,
    pub mu_challenge: BabyBearExtVar,
    pub folding_pow_bits: AssignedValue<Fr>,
    pub folding_pow_sampled_bits: Vec<AssignedValue<Fr>>,
    pub folding_pow_witness_ok: Vec<AssignedValue<Fr>>,
    pub folding_alphas: Vec<BabyBearExtVar>,
    pub z0_challenges: Vec<BabyBearExtVar>,
    pub query_phase_pow_bits: AssignedValue<Fr>,
    pub query_phase_pow_sampled_bits: Vec<AssignedValue<Fr>>,
    pub query_phase_pow_witness_ok: Vec<AssignedValue<Fr>>,
    pub gammas: Vec<BabyBearExtVar>,
    pub query_indices: Vec<AssignedValue<Fr>>,
    pub whir_sumcheck_polys: Vec<Vec<BabyBearExtVar>>,
    pub stacking_openings: Vec<Vec<BabyBearExtVar>>,
    pub initial_commitment_roots: Vec<AssignedValue<Fr>>,
    pub codeword_commitment_roots: Vec<AssignedValue<Fr>>,
    pub ood_values: Vec<BabyBearExtVar>,
    pub final_poly: Vec<BabyBearExtVar>,
    pub final_poly_len: AssignedValue<Fr>,
    pub u_cube: Vec<BabyBearExtVar>,
    pub final_claim: BabyBearExtVar,
    pub final_acc: BabyBearExtVar,
    pub final_residual: BabyBearExtVar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawWhirWitnessState {
    pub intermediates: WhirIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedWhirState {
    pub final_acc: BabyBearExtVar,
    pub final_claim: BabyBearExtVar,
    pub final_residual: BabyBearExtVar,
}

#[derive(Clone, Debug)]
pub struct CheckedWhirWitnessState {
    pub assigned: AssignedWhirIntermediates,
    pub derived: DerivedWhirState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WhirStrictOwnership {
    pub k_whir: usize,
    pub initial_log_rs_domain_size: usize,
    pub mu_pow_bits: usize,
    pub folding_pow_bits: usize,
    pub query_phase_pow_bits: usize,
    pub folding_counts_per_round: Vec<usize>,
    pub query_counts_per_round: Vec<usize>,
    pub query_index_bits: Vec<usize>,
    pub expected_final_poly_len: usize,
}

#[derive(Clone, Debug)]
struct PreparedWhirInputs {
    transcript: NativeTranscript,
    commits: Vec<NativeDigest>,
    u_cube: Vec<NativeEF>,
}

fn ext_to_coeffs(value: NativeEF) -> [u64; BABY_BEAR_EXT_DEGREE] {
    core::array::from_fn(|i| {
        <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&value)[i]
            .as_canonical_u64()
    })
}

fn coeffs_to_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> NativeEF {
    NativeEF::from_basis_coefficients_fn(|i| NativeF::from_u64(coeffs[i]))
}

fn digest_to_fr(digest: NativeDigest) -> Fr {
    biguint_to_fe(&digest[0].as_canonical_biguint())
}

fn base_slice_to_u64_vec(values: &[NativeF]) -> Vec<u64> {
    values
        .iter()
        .map(|value| value.as_canonical_u64())
        .collect::<Vec<_>>()
}

fn ext_to_u64_vec(value: NativeEF) -> Vec<u64> {
    ext_to_coeffs(value).to_vec()
}

fn check_witness_with_sample_bits(
    transcript: &mut impl FiatShamirTranscript<NativeConfig>,
    bits: usize,
    witness: NativeF,
) -> (bool, u64) {
    if bits == 0 {
        return (true, 0);
    }
    transcript.observe(witness);
    let sampled_bits = transcript.sample_bits(bits) as u64;
    (sampled_bits == 0, sampled_bits)
}

fn prepare_whir_inputs(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<PreparedWhirInputs, WhirError> {
    let mut transcript = default_transcript();
    let prepared = prepare_pipeline_inputs(&mut transcript, config, mvk, proof)?;

    let stacked = derive_stacked_reduction_intermediates_with_inputs(
        &mut transcript,
        &proof.stacking_proof,
        &prepared.layouts,
        &prepared.need_rot_per_commit,
        prepared.l_skip,
        mvk.inner.params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &prepared.r,
        &prepared.omega_skip_pows,
    )?;
    let u_prism = stacked
        .u
        .iter()
        .copied()
        .map(stacked_coeffs_to_native_ext)
        .collect::<Vec<_>>();
    let u_cube = derive_u_cube_from_prism(&u_prism, prepared.l_skip)?;
    let commits = collect_trace_commitments(&mvk.inner, proof, &prepared.trace_id_to_air_id)?;

    Ok(PreparedWhirInputs {
        transcript,
        commits,
        u_cube,
    })
}

pub(crate) fn derive_whir_intermediates_with_inputs(
    transcript: &mut impl FiatShamirTranscript<NativeConfig>,
    config: &NativeConfig,
    whir_proof: &WhirProof<NativeConfig>,
    stacking_openings: &[Vec<NativeEF>],
    commitments: &[NativeDigest],
    u: &[NativeEF],
) -> Result<WhirIntermediates, VerifyWhirError> {
    let params = config.params();
    let widths = stacking_openings
        .iter()
        .map(|v| v.len())
        .collect::<Vec<_>>();

    let mu_pow_bits = params.whir.mu_pow_bits;
    let (mu_pow_witness_ok, mu_pow_sampled_bits) =
        check_witness_with_sample_bits(transcript, mu_pow_bits, whir_proof.mu_pow_witness);
    if !mu_pow_witness_ok {
        return Err(VerifyWhirError::MuPoWInvalid);
    }

    let mu = transcript.sample_ext();

    let WhirProof {
        mu_pow_witness,
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        final_poly,
    } = whir_proof;

    let m = params.l_skip + params.n_stack;
    let k_whir = params.k_whir();
    let num_whir_rounds = params.num_whir_rounds();
    let mut log_rs_domain_size = m + params.log_blowup;
    let initial_log_rs_domain_size = log_rs_domain_size;

    let mut sumcheck_poly_iter = whir_sumcheck_polys.iter();
    let mut folding_pow_iter = folding_pow_witnesses.iter();

    let mu_pows: Vec<_> = mu.powers().take(widths.iter().sum::<usize>()).collect();
    let mut claim = stacking_openings
        .iter()
        .flatten()
        .zip(mu_pows.iter())
        .fold(NativeEF::ZERO, |acc, (&opening, &mu_pow)| {
            acc + mu_pow * opening
        });

    let mut gammas = Vec::with_capacity(num_whir_rounds);
    let mut zs = Vec::with_capacity(num_whir_rounds);
    let mut z0s = Vec::with_capacity(num_whir_rounds.saturating_sub(1));
    let mut alphas = Vec::with_capacity(m);

    let mut folding_pow_witness_ok = Vec::with_capacity(folding_pow_witnesses.len());
    let mut folding_pow_sampled_bits = Vec::with_capacity(folding_pow_witnesses.len());
    let mut folding_alphas = Vec::with_capacity(folding_pow_witnesses.len());
    let mut query_phase_pow_witness_ok = Vec::with_capacity(query_phase_pow_witnesses.len());
    let mut query_phase_pow_sampled_bits = Vec::with_capacity(query_phase_pow_witnesses.len());
    let mut folding_counts_per_round = Vec::with_capacity(num_whir_rounds);
    let mut query_counts_per_round = Vec::with_capacity(num_whir_rounds);
    let mut query_index_bits = Vec::new();
    let mut query_indices = Vec::new();
    let mut merkle_paths = Vec::new();

    for (whir_round, (query_phase_pow_witness, round_params)) in
        zip(query_phase_pow_witnesses, &params.whir.rounds).enumerate()
    {
        let is_initial_round = whir_round == 0;
        let is_final_round = whir_round == num_whir_rounds - 1;

        let mut alphas_round = Vec::with_capacity(k_whir);
        let mut folding_count = 0usize;

        for _ in 0..k_whir {
            if let Some(evals) = sumcheck_poly_iter.next() {
                folding_count += 1;
                let &[ev1, ev2] = evals;

                transcript.observe_ext(ev1);
                transcript.observe_ext(ev2);

                let pow_witness = *folding_pow_iter
                    .next()
                    .expect("proof shape guarantees folding witness length");
                let (pow_ok, sampled_bits) = check_witness_with_sample_bits(
                    transcript,
                    params.whir.folding_pow_bits,
                    pow_witness,
                );
                folding_pow_witness_ok.push(pow_ok);
                folding_pow_sampled_bits.push(sampled_bits);
                if !pow_ok {
                    return Err(VerifyWhirError::FoldingPoWInvalid);
                }

                let alpha = transcript.sample_ext();
                alphas_round.push(alpha);
                folding_alphas.push(ext_to_coeffs(alpha));

                let ev0 = claim - ev1;
                claim = interpolate_quadratic_at_012(&[ev0, ev1, ev2], alpha);
            }
        }
        folding_counts_per_round.push(folding_count);

        let y0 = if is_final_round {
            for coeff in final_poly {
                transcript.observe_ext(*coeff);
            }
            None
        } else {
            let commit = codeword_commits[whir_round];
            transcript.observe_commit(commit);

            let z0 = transcript.sample_ext();
            z0s.push(z0);

            let y0 = ood_values[whir_round];
            transcript.observe_ext(y0);
            Some(y0)
        };

        let (query_pow_ok, query_pow_sampled_bits) = check_witness_with_sample_bits(
            transcript,
            params.whir.query_phase_pow_bits,
            *query_phase_pow_witness,
        );
        query_phase_pow_witness_ok.push(query_pow_ok);
        query_phase_pow_sampled_bits.push(query_pow_sampled_bits);
        if !query_pow_ok {
            return Err(VerifyWhirError::QueryPhasePoWInvalid);
        }

        let query_bits = log_rs_domain_size - k_whir;
        let num_queries = round_params.num_queries;
        query_counts_per_round.push(num_queries);
        let query_indices_iter = (0..num_queries).map(|_| transcript.sample_bits(query_bits));

        let mut zs_round = Vec::with_capacity(num_queries);
        let mut ys_round = Vec::with_capacity(num_queries);

        let hasher = config.hasher();
        let omega = NativeF::two_adic_generator(log_rs_domain_size);

        for (query_idx, index) in query_indices_iter.enumerate() {
            let query_position = query_indices.len();
            query_index_bits.push(query_bits);
            query_indices.push(index as u64);

            let zi_root = omega.exp_u64(index);
            let zi = zi_root.exp_power_of_2(k_whir);

            let yi = if is_initial_round {
                let mut codeword_vals = vec![NativeEF::ZERO; 1 << k_whir];
                let mut mu_pow_iter = mu_pows.iter();

                for commit_idx in 0..commitments.len() {
                    let commit = commitments[commit_idx];
                    let width = widths[commit_idx];
                    let opened_rows = &initial_round_opened_rows[commit_idx][query_idx];
                    let leaf_hashes = opened_rows
                        .iter()
                        .map(|opened_row| hasher.hash_slice(opened_row))
                        .collect::<Vec<_>>();
                    let query_digest = hasher.tree_compress(leaf_hashes);
                    let merkle_proof = &initial_round_merkle_proofs[commit_idx][query_idx];

                    if merkle_verify(hasher, commit, index as u32, query_digest, merkle_proof)
                        .is_err()
                    {
                        return Err(VerifyWhirError::MerkleVerify);
                    }
                    merkle_paths.push(MerklePathIntermediates {
                        query_position,
                        query_index_bits: query_bits,
                        leaf_inputs: opened_rows
                            .iter()
                            .map(|opened_row| base_slice_to_u64_vec(opened_row))
                            .collect::<Vec<_>>(),
                        root_digest: digest_to_fr(commit),
                        siblings: merkle_proof
                            .iter()
                            .map(|digest| digest_to_fr(*digest))
                            .collect::<Vec<_>>(),
                    });

                    for c in 0..width {
                        let mu_pow = *mu_pow_iter.next().expect(
                            "proof shape guarantees total opening width matches mu power count",
                        );
                        for j in 0..(1 << k_whir) {
                            codeword_vals[j] += mu_pow * opened_rows[j][c];
                        }
                    }
                }
                binary_k_fold::<NativeF, NativeEF>(codeword_vals, &alphas_round, zi_root)
            } else {
                let opened_values = codeword_opened_values[whir_round - 1][query_idx].clone();
                let merkle_proof = &codeword_merkle_proofs[whir_round - 1][query_idx];
                let leaf_hashes = opened_values
                    .iter()
                    .map(|opened_value| {
                        hasher.hash_slice(opened_value.as_basis_coefficients_slice())
                    })
                    .collect::<Vec<_>>();
                let query_digest = hasher.tree_compress(leaf_hashes);

                if merkle_verify(
                    hasher,
                    codeword_commits[whir_round - 1],
                    index as u32,
                    query_digest,
                    merkle_proof,
                )
                .is_err()
                {
                    return Err(VerifyWhirError::MerkleVerify);
                }
                merkle_paths.push(MerklePathIntermediates {
                    query_position,
                    query_index_bits: query_bits,
                    leaf_inputs: opened_values
                        .iter()
                        .map(|opened_value| ext_to_u64_vec(*opened_value))
                        .collect::<Vec<_>>(),
                    root_digest: digest_to_fr(codeword_commits[whir_round - 1]),
                    siblings: merkle_proof
                        .iter()
                        .map(|digest| digest_to_fr(*digest))
                        .collect::<Vec<_>>(),
                });

                binary_k_fold::<NativeF, NativeEF>(opened_values, &alphas_round, zi_root)
            };

            zs_round.push(zi);
            ys_round.push(yi);
        }

        let gamma = transcript.sample_ext();
        if let Some(y0) = y0 {
            claim += y0 * gamma;
        }
        for (yi, gamma_pow) in ys_round.iter().zip(gamma.powers().skip(2)) {
            claim += *yi * gamma_pow;
        }

        gammas.push(gamma);
        zs.push(zs_round);
        alphas.extend(alphas_round);

        log_rs_domain_size -= 1;
    }

    let final_poly_len = final_poly.len();
    let expected_final_poly_len = 1 << params.log_final_poly_len();
    if final_poly_len != expected_final_poly_len {
        return Err(VerifyWhirError::FinalPolyDegree);
    }

    let t = k_whir * num_whir_rounds;
    let prefix = eval_mobius_eq_mle(&u[..t], &alphas[..t]);
    let mut final_poly_evals = final_poly.clone();
    let suffix_sum = eval_mle_evals_at_point(&mut final_poly_evals, &u[t..]);
    let mut acc = prefix * suffix_sum;

    let mut j = k_whir;
    for i in 0..num_whir_rounds {
        let zis = &zs[i];
        let gamma = gammas[i];
        let alpha_slc = &alphas[j..t];
        let slc_len = (t - j) + 1;

        if i != num_whir_rounds - 1 {
            let z0_pow = z0s[i].exp_powers_of_2().take(slc_len).collect::<Vec<_>>();
            let (z0_pow_max, z0_pow_left) = z0_pow
                .split_last()
                .expect("slc_len is at least one in non-final rounds");
            let term = gamma
                * eval_eq_mle(alpha_slc, z0_pow_left)
                * horner_eval::<NativeEF, NativeEF, NativeEF>(final_poly, *z0_pow_max);
            acc += term;
        }

        for (zi, gamma_pow) in zip(zis, gamma.powers().skip(2)) {
            let zi_pow = zi.exp_powers_of_2().take(slc_len).collect::<Vec<_>>();
            let (zi_pow_max, zi_pow_left) = zi_pow
                .split_last()
                .expect("slc_len is at least one for query evaluations");
            let term = gamma_pow
                * eval_eq_mle(alpha_slc, zi_pow_left)
                * horner_eval::<NativeEF, NativeF, NativeEF>(final_poly, *zi_pow_max);
            acc += term;
        }

        j += k_whir;
    }

    let final_residual = acc - claim;
    if acc != claim {
        return Err(VerifyWhirError::FinalPolyConstraint);
    }

    let initial_commitment_roots = commitments
        .iter()
        .copied()
        .map(digest_to_fr)
        .collect::<Vec<_>>();
    let codeword_commitment_roots = codeword_commits
        .iter()
        .copied()
        .map(digest_to_fr)
        .collect::<Vec<_>>();

    Ok(WhirIntermediates {
        k_whir,
        initial_log_rs_domain_size,
        mu_pow_bits,
        mu_pow_witness: mu_pow_witness.as_canonical_u64(),
        mu_pow_sampled_bits,
        mu_pow_witness_ok,
        mu_challenge: ext_to_coeffs(mu),
        folding_pow_bits: params.whir.folding_pow_bits,
        folding_pow_witnesses: folding_pow_witnesses
            .iter()
            .map(|w| w.as_canonical_u64())
            .collect::<Vec<_>>(),
        folding_pow_sampled_bits,
        folding_pow_witness_ok,
        folding_alphas,
        z0_challenges: z0s.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>(),
        query_phase_pow_bits: params.whir.query_phase_pow_bits,
        query_phase_pow_witnesses: query_phase_pow_witnesses
            .iter()
            .map(|w| w.as_canonical_u64())
            .collect::<Vec<_>>(),
        query_phase_pow_sampled_bits,
        query_phase_pow_witness_ok,
        gammas: gammas
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        folding_counts_per_round,
        query_counts_per_round,
        query_index_bits,
        query_indices,
        initial_commitment_roots,
        codeword_commitment_roots,
        merkle_paths,
        final_poly_len,
        expected_final_poly_len,
        stacking_openings: stacking_openings
            .iter()
            .map(|row| row.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        whir_sumcheck_polys: whir_sumcheck_polys
            .iter()
            .map(|poly| poly.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        ood_values: ood_values
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        final_poly: final_poly
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        u_cube: u.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>(),
        final_claim: ext_to_coeffs(claim),
        final_acc: ext_to_coeffs(acc),
        final_residual: ext_to_coeffs(final_residual),
    })
}

pub fn derive_whir_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<WhirIntermediates, WhirError> {
    let PreparedWhirInputs {
        mut transcript,
        commits,
        u_cube,
    } = prepare_whir_inputs(config, mvk, proof)?;

    derive_whir_intermediates_with_inputs(
        &mut transcript,
        config,
        &proof.whir_proof,
        &proof.stacking_proof.stacking_openings,
        &commits,
        &u_cube,
    )
    .map_err(Into::into)
}

fn assign_bool(ctx: &mut Context<Fr>, range: &RangeChip<Fr>, value: bool) -> AssignedValue<Fr> {
    let bit = ctx.load_witness(Fr::from(value as u64));
    range.gate().assert_bit(ctx, bit);
    bit
}

fn assign_ext(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    coeffs: [u64; BABY_BEAR_EXT_DEGREE],
) -> BabyBearExtVar {
    baby_bear.load_ext_witness(ctx, range, coeffs)
}

fn ext_from_base_var(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &BabyBearVar,
) -> BabyBearExtVar {
    let zero = baby_bear.zero(ctx, range);
    BabyBearExtVar {
        coeffs: core::array::from_fn(|idx| {
            if idx == 0 {
                value.clone()
            } else {
                zero.clone()
            }
        }),
    }
}

fn eval_mobius_eq_mle_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    u: &[BabyBearExtVar],
    x: &[BabyBearExtVar],
) -> BabyBearExtVar {
    assert_eq!(u.len(), x.len(), "mobius-eq arity mismatch");
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut acc = one.clone();
    for (u_i, x_i) in u.iter().zip(x.iter()) {
        let two_u = ext_mul_base_const(ctx, range, baby_bear, u_i, 2);
        let w0 = baby_bear.ext_sub(ctx, range, &one, &two_u);
        let one_minus_x = baby_bear.ext_sub(ctx, range, &one, x_i);
        let left = baby_bear.ext_mul(ctx, range, &w0, &one_minus_x);
        let right = baby_bear.ext_mul(ctx, range, u_i, x_i);
        let factor = baby_bear.ext_add(ctx, range, &left, &right);
        acc = baby_bear.ext_mul(ctx, range, &acc, &factor);
    }
    acc
}

fn eval_mle_evals_at_point_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    evals: &[BabyBearExtVar],
    x: &[BabyBearExtVar],
) -> BabyBearExtVar {
    assert_eq!(
        evals.len(),
        1usize << x.len(),
        "MLE table length must be 2^arity",
    );
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut values = evals.to_vec();
    let mut len = values.len();
    for xj in x.iter().rev() {
        len >>= 1;
        let one_minus_xj = baby_bear.ext_sub(ctx, range, &one, xj);
        for i in 0..len {
            let lo = values[i].clone();
            let hi = values[i + len].clone();
            let lo_term = baby_bear.ext_mul(ctx, range, &lo, &one_minus_xj);
            let hi_term = baby_bear.ext_mul(ctx, range, &hi, xj);
            values[i] = baby_bear.ext_add(ctx, range, &lo_term, &hi_term);
        }
    }
    values
        .first()
        .cloned()
        .expect("MLE reduction must produce one value")
}

fn invert_base_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &BabyBearVar,
) -> BabyBearVar {
    let value_u64 = value.as_u64();
    assert!(value_u64 != 0, "cannot invert zero BabyBear value");
    let inv_u64 = NativeF::from_u64(value_u64).inverse().as_canonical_u64();
    let inv = baby_bear.load_witness(ctx, range, inv_u64);
    let one = baby_bear.one(ctx, range);
    let check = baby_bear.mul(ctx, range, value, &inv);
    baby_bear.assert_equal(ctx, &check, &one);
    inv
}

fn query_root_from_bits_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    query_bits: &[AssignedValue<Fr>],
    log_rs_domain_size: usize,
) -> BabyBearVar {
    let gate = range.gate();
    let one = ctx.load_constant(Fr::from(1u64));
    let omega = NativeF::two_adic_generator(log_rs_domain_size);
    let mut root = baby_bear.one(ctx, range);
    for (bit_idx, &bit) in query_bits.iter().enumerate() {
        let omega_pow = omega.exp_u64(1u64 << bit_idx).as_canonical_u64();
        let selected_u64 = if *bit.value() == Fr::from(1u64) {
            omega_pow
        } else {
            1u64
        };
        let selected = baby_bear.load_witness(ctx, range, selected_u64);
        let omega_pow_const = ctx.load_constant(Fr::from(omega_pow));
        let bit_times_pow = gate.mul(ctx, bit, omega_pow_const);
        let one_minus_bit = gate.sub(ctx, one, bit);
        let rhs = gate.add(ctx, bit_times_pow, one_minus_bit);
        ctx.constrain_equal(&selected.cell, &rhs);
        root = baby_bear.mul(ctx, range, &root, &selected);
    }
    root
}

fn binary_k_fold_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    mut values: Vec<BabyBearExtVar>,
    alphas: &[BabyBearExtVar],
    x: &BabyBearVar,
) -> BabyBearExtVar {
    if alphas.is_empty() {
        assert_eq!(values.len(), 1, "k=0 fold must have exactly one value");
        return values[0].clone();
    }
    let n = values.len();
    assert_eq!(
        n,
        1usize << alphas.len(),
        "binary-k fold value count must match 2^k",
    );

    let k = alphas.len();
    let omega_k = NativeF::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();
    let tw = omega_k.powers().take(1usize << (k - 1)).collect();
    let inv_tw = omega_k_inv.powers().take(1usize << (k - 1)).collect();
    let half = NativeF::ONE.halve().as_canonical_u64();

    let mut x_pow = x.clone();
    let x_inv = invert_base_assigned(ctx, range, baby_bear, x);
    let mut x_inv_pow = x_inv;

    for (j, alpha) in alphas.iter().enumerate() {
        let m = n >> (j + 1);
        for i in 0..m {
            let t = baby_bear.mul_const(ctx, range, &x_pow, tw[i << j].as_canonical_u64());
            let t_inv =
                baby_bear.mul_const(ctx, range, &x_inv_pow, inv_tw[i << j].as_canonical_u64());
            let t_inv_half = baby_bear.mul_const(ctx, range, &t_inv, half);

            let lo = values[i].clone();
            let hi = values[i + m].clone();
            let lo_minus_hi = baby_bear.ext_sub(ctx, range, &lo, &hi);
            let t_ext = ext_from_base_var(ctx, range, baby_bear, &t);
            let alpha_minus_t = baby_bear.ext_sub(ctx, range, alpha, &t_ext);
            let fold = baby_bear.ext_mul(ctx, range, &alpha_minus_t, &lo_minus_hi);
            let t_inv_half_ext = ext_from_base_var(ctx, range, baby_bear, &t_inv_half);
            let fold = baby_bear.ext_mul(ctx, range, &fold, &t_inv_half_ext);
            values[i] = baby_bear.ext_add(ctx, range, &lo, &fold);
        }
        x_pow = baby_bear.square(ctx, range, &x_pow);
        x_inv_pow = baby_bear.square(ctx, range, &x_inv_pow);
    }
    values[0].clone()
}

fn tree_compress_assigned_digests(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
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
            next.push(compress_bn254_digests(ctx, range, pair[0], pair[1]));
        }
        level = next;
    }
    level
        .pop()
        .expect("tree_compress must output one digest for non-empty inputs")
}

struct AssignedMerklePathPayload {
    leaf_values: Vec<Vec<BabyBearVar>>,
    query_bits: Vec<AssignedValue<Fr>>,
}

fn constrain_merkle_path(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    query_index: AssignedValue<Fr>,
    query_index_bits: usize,
    leaf_inputs: &[Vec<u64>],
    siblings: &[Fr],
    root_digest: AssignedValue<Fr>,
) -> AssignedMerklePathPayload {
    assert!(
        leaf_inputs.len().is_power_of_two(),
        "leaf input count must be power of two"
    );

    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;
    let one = ctx.load_constant(Fr::from(1u64));

    let merkle_depth = assign_and_range_usize(ctx, range, siblings.len());
    gate.assert_is_const(
        ctx,
        &merkle_depth,
        &Fr::from(usize_to_u64(query_index_bits)),
    );

    let leaf_values = leaf_inputs
        .iter()
        .map(|leaf| {
            leaf.iter()
                .map(|&value| baby_bear.load_witness(ctx, range, value))
                .collect::<Vec<BabyBearVar>>()
        })
        .collect::<Vec<_>>();
    let leaf_hashes = leaf_values
        .iter()
        .map(|leaf| hash_babybear_slice_to_digest(ctx, range, leaf))
        .collect::<Vec<_>>();

    let mut cur = tree_compress_assigned_digests(ctx, range, leaf_hashes);
    let query_bits = if siblings.is_empty() {
        Vec::new()
    } else {
        gate.num_to_bits(ctx, query_index, siblings.len())
    };

    for (bit, sibling_digest) in query_bits.iter().zip(siblings.iter()) {
        let sibling = ctx.load_witness(*sibling_digest);
        let left_right = compress_bn254_digests(ctx, range, cur, sibling);
        let right_left = compress_bn254_digests(ctx, range, sibling, cur);
        let one_minus_bit = gate.sub(ctx, one, *bit);
        let pick_right_left = gate.mul(ctx, right_left, *bit);
        let pick_left_right = gate.mul(ctx, left_right, one_minus_bit);
        cur = gate.add(ctx, pick_right_left, pick_left_right);
    }

    ctx.constrain_equal(&cur, &root_digest);
    AssignedMerklePathPayload {
        leaf_values,
        query_bits,
    }
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_whir_intermediates_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &WhirIntermediates,
) -> AssignedWhirIntermediates {
    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;

    let query_index_bits_len = assign_and_range_usize(ctx, range, actual.query_index_bits.len());
    let query_indices_len = assign_and_range_usize(ctx, range, actual.query_indices.len());
    ctx.constrain_equal(&query_index_bits_len, &query_indices_len);

    let folding_pow_sampled_len =
        assign_and_range_usize(ctx, range, actual.folding_pow_sampled_bits.len());
    let folding_pow_ok_len =
        assign_and_range_usize(ctx, range, actual.folding_pow_witness_ok.len());
    ctx.constrain_equal(&folding_pow_sampled_len, &folding_pow_ok_len);

    let query_pow_sampled_len =
        assign_and_range_usize(ctx, range, actual.query_phase_pow_sampled_bits.len());
    let query_pow_ok_len =
        assign_and_range_usize(ctx, range, actual.query_phase_pow_witness_ok.len());
    ctx.constrain_equal(&query_pow_sampled_len, &query_pow_ok_len);

    let query_round_count = assign_and_range_usize(ctx, range, actual.query_counts_per_round.len());
    ctx.constrain_equal(&query_round_count, &query_pow_sampled_len);
    let folding_round_count =
        assign_and_range_usize(ctx, range, actual.folding_counts_per_round.len());
    ctx.constrain_equal(&folding_round_count, &query_pow_sampled_len);

    let folding_pow_witness_len =
        assign_and_range_usize(ctx, range, actual.folding_pow_witnesses.len());
    ctx.constrain_equal(&folding_pow_witness_len, &folding_pow_sampled_len);
    let query_pow_witness_len =
        assign_and_range_usize(ctx, range, actual.query_phase_pow_witnesses.len());
    ctx.constrain_equal(&query_pow_witness_len, &query_pow_sampled_len);
    let whir_sumcheck_poly_len =
        assign_and_range_usize(ctx, range, actual.whir_sumcheck_polys.len());
    ctx.constrain_equal(&whir_sumcheck_poly_len, &folding_pow_sampled_len);

    let codeword_commitment_count =
        assign_and_range_usize(ctx, range, actual.codeword_commitment_roots.len());
    gate.assert_is_const(
        ctx,
        &codeword_commitment_count,
        &Fr::from(usize_to_u64(
            actual.query_counts_per_round.len().saturating_sub(1),
        )),
    );

    let k_whir = assign_and_range_usize(ctx, range, actual.k_whir);
    let k_whir_is_zero = gate.is_zero(ctx, k_whir);
    gate.assert_is_const(ctx, &k_whir_is_zero, &Fr::from(0u64));

    let _initial_log_rs_domain_size =
        assign_and_range_usize(ctx, range, actual.initial_log_rs_domain_size);
    range.check_less_than_safe(
        ctx,
        k_whir,
        usize_to_u64(actual.initial_log_rs_domain_size).saturating_add(1),
    );

    let mu_pow_bits = assign_and_range_usize(ctx, range, actual.mu_pow_bits);
    let mu_pow_sampled_bits = assign_and_range_u64(ctx, range, actual.mu_pow_sampled_bits);
    if actual.mu_pow_bits > 0 {
        range.range_check(ctx, mu_pow_sampled_bits, actual.mu_pow_bits);
    } else {
        gate.assert_is_const(ctx, &mu_pow_sampled_bits, &Fr::from(0u64));
    }
    let mu_pow_witness_ok = assign_bool(ctx, range, actual.mu_pow_witness_ok);
    let mu_pow_is_zero = gate.is_zero(ctx, mu_pow_sampled_bits);
    ctx.constrain_equal(&mu_pow_witness_ok, &mu_pow_is_zero);
    gate.assert_is_const(ctx, &mu_pow_witness_ok, &Fr::from(1u64));
    let mu_challenge = assign_ext(ctx, range, &baby_bear, actual.mu_challenge);

    let folding_pow_bits = assign_and_range_usize(ctx, range, actual.folding_pow_bits);
    let mut folding_pow_sampled_bits = Vec::with_capacity(actual.folding_pow_sampled_bits.len());
    let mut folding_pow_witness_ok = Vec::with_capacity(actual.folding_pow_witness_ok.len());
    for (&sampled_bits, &actual_ok) in actual
        .folding_pow_sampled_bits
        .iter()
        .zip(&actual.folding_pow_witness_ok)
    {
        let sampled_bits = assign_and_range_u64(ctx, range, sampled_bits);
        if actual.folding_pow_bits > 0 {
            range.range_check(ctx, sampled_bits, actual.folding_pow_bits);
        } else {
            gate.assert_is_const(ctx, &sampled_bits, &Fr::from(0u64));
        }

        let bit = assign_bool(ctx, range, actual_ok);
        let is_zero = gate.is_zero(ctx, sampled_bits);
        ctx.constrain_equal(&bit, &is_zero);
        gate.assert_is_const(ctx, &bit, &Fr::from(1u64));
        folding_pow_sampled_bits.push(sampled_bits);
        folding_pow_witness_ok.push(bit);
    }

    let query_phase_pow_bits = assign_and_range_usize(ctx, range, actual.query_phase_pow_bits);
    let mut query_phase_pow_sampled_bits =
        Vec::with_capacity(actual.query_phase_pow_sampled_bits.len());
    let mut query_phase_pow_witness_ok =
        Vec::with_capacity(actual.query_phase_pow_witness_ok.len());
    for (&sampled_bits, &actual_ok) in actual
        .query_phase_pow_sampled_bits
        .iter()
        .zip(&actual.query_phase_pow_witness_ok)
    {
        let sampled_bits = assign_and_range_u64(ctx, range, sampled_bits);
        if actual.query_phase_pow_bits > 0 {
            range.range_check(ctx, sampled_bits, actual.query_phase_pow_bits);
        } else {
            gate.assert_is_const(ctx, &sampled_bits, &Fr::from(0u64));
        }

        let bit = assign_bool(ctx, range, actual_ok);
        let is_zero = gate.is_zero(ctx, sampled_bits);
        ctx.constrain_equal(&bit, &is_zero);
        gate.assert_is_const(ctx, &bit, &Fr::from(1u64));
        query_phase_pow_sampled_bits.push(sampled_bits);
        query_phase_pow_witness_ok.push(bit);
    }

    let query_indices: Vec<AssignedValue<Fr>> = actual
        .query_index_bits
        .iter()
        .enumerate()
        .map(|(query_pos, &actual_bits)| {
            let actual_idx = actual.query_indices.get(query_pos).copied().unwrap_or(0u64);
            let idx = ctx.load_witness(Fr::from(actual_idx));
            if actual_bits > 0 {
                range.range_check(ctx, idx, actual_bits);
            } else {
                gate.assert_is_const(ctx, &idx, &Fr::from(0));
            }
            idx
        })
        .collect();

    let expected_query_count = actual.query_counts_per_round.iter().sum::<usize>();
    let assigned_query_count = assign_and_range_usize(ctx, range, query_indices.len());
    gate.assert_is_const(
        ctx,
        &assigned_query_count,
        &Fr::from(usize_to_u64(expected_query_count)),
    );
    let initial_query_count = actual.query_counts_per_round.first().copied().unwrap_or(0);
    let expected_merkle_paths = actual
        .initial_commitment_roots
        .len()
        .saturating_mul(initial_query_count)
        + actual.query_counts_per_round.iter().skip(1).sum::<usize>();
    let merkle_path_count = assign_and_range_usize(ctx, range, actual.merkle_paths.len());
    gate.assert_is_const(
        ctx,
        &merkle_path_count,
        &Fr::from(usize_to_u64(expected_merkle_paths)),
    );

    let initial_commitment_roots = actual
        .initial_commitment_roots
        .iter()
        .map(|&root| ctx.load_witness(root))
        .collect::<Vec<_>>();
    let codeword_commitment_roots = actual
        .codeword_commitment_roots
        .iter()
        .map(|&root| ctx.load_witness(root))
        .collect::<Vec<_>>();

    let mut path_cursor = 0usize;
    let mut global_query_position = 0usize;
    let mut merkle_payloads = Vec::with_capacity(actual.merkle_paths.len());
    let default_merkle_path = MerklePathIntermediates {
        query_position: 0,
        query_index_bits: 0,
        leaf_inputs: vec![vec![0u64]],
        root_digest: Fr::from(0u64),
        siblings: Vec::new(),
    };
    let zero_query_index = ctx.load_constant(Fr::from(0u64));

    for _ in 0..initial_query_count {
        for &initial_root in &initial_commitment_roots {
            let path = actual
                .merkle_paths
                .get(path_cursor)
                .unwrap_or(&default_merkle_path);
            let query_position = assign_and_range_usize(ctx, range, path.query_position);
            gate.assert_is_const(
                ctx,
                &query_position,
                &Fr::from(usize_to_u64(global_query_position)),
            );

            let expected_bits = actual.query_index_bits[global_query_position];
            let path_query_bits = assign_and_range_usize(ctx, range, path.query_index_bits);
            gate.assert_is_const(
                ctx,
                &path_query_bits,
                &Fr::from(usize_to_u64(expected_bits)),
            );

            let query_index = query_indices
                .get(global_query_position)
                .copied()
                .unwrap_or(zero_query_index);
            let path_root = ctx.load_witness(path.root_digest);
            let expected_root = initial_root;
            ctx.constrain_equal(&path_root, &expected_root);
            let payload = constrain_merkle_path(
                ctx,
                range,
                query_index,
                expected_bits,
                &path.leaf_inputs,
                &path.siblings,
                path_root,
            );
            merkle_payloads.push(payload);

            path_cursor += 1;
        }
        global_query_position += 1;
    }

    for (round_idx, &num_queries) in actual.query_counts_per_round.iter().enumerate().skip(1) {
        let round_root = codeword_commitment_roots[round_idx - 1];
        for _ in 0..num_queries {
            let path = actual
                .merkle_paths
                .get(path_cursor)
                .unwrap_or(&default_merkle_path);
            let query_position = assign_and_range_usize(ctx, range, path.query_position);
            gate.assert_is_const(
                ctx,
                &query_position,
                &Fr::from(usize_to_u64(global_query_position)),
            );

            let expected_bits = actual.query_index_bits[global_query_position];
            let path_query_bits = assign_and_range_usize(ctx, range, path.query_index_bits);
            gate.assert_is_const(
                ctx,
                &path_query_bits,
                &Fr::from(usize_to_u64(expected_bits)),
            );

            let query_index = query_indices
                .get(global_query_position)
                .copied()
                .unwrap_or(zero_query_index);
            let path_root = ctx.load_witness(path.root_digest);
            let expected_root = round_root;
            ctx.constrain_equal(&path_root, &expected_root);
            let payload = constrain_merkle_path(
                ctx,
                range,
                query_index,
                expected_bits,
                &path.leaf_inputs,
                &path.siblings,
                path_root,
            );
            merkle_payloads.push(payload);

            path_cursor += 1;
            global_query_position += 1;
        }
    }
    let path_cursor_cell = assign_and_range_usize(ctx, range, path_cursor);
    gate.assert_is_const(
        ctx,
        &path_cursor_cell,
        &Fr::from(usize_to_u64(actual.merkle_paths.len())),
    );
    let consumed_query_count = assign_and_range_usize(ctx, range, global_query_position);
    gate.assert_is_const(
        ctx,
        &consumed_query_count,
        &Fr::from(usize_to_u64(query_indices.len())),
    );
    let payload_count = assign_and_range_usize(ctx, range, merkle_payloads.len());
    gate.assert_is_const(
        ctx,
        &payload_count,
        &Fr::from(usize_to_u64(actual.merkle_paths.len())),
    );

    let final_poly_len = assign_and_range_usize(ctx, range, actual.final_poly_len);
    gate.assert_is_const(
        ctx,
        &final_poly_len,
        &Fr::from(usize_to_u64(actual.expected_final_poly_len)),
    );
    let whir_sumcheck_polys = actual
        .whir_sumcheck_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let ood_values = actual
        .ood_values
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let final_poly = actual
        .final_poly
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let stacking_openings = actual
        .stacking_openings
        .iter()
        .map(|row| {
            row.iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let stacking_opening_count = assign_and_range_usize(ctx, range, stacking_openings.len());
    gate.assert_is_const(
        ctx,
        &stacking_opening_count,
        &Fr::from(usize_to_u64(initial_commitment_roots.len())),
    );

    let u_cube = actual
        .u_cube
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let folding_alphas = actual
        .folding_alphas
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let z0_challenges = actual
        .z0_challenges
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let gammas = actual
        .gammas
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();

    let final_claim = assign_ext(ctx, range, &baby_bear, actual.final_claim);
    let final_acc = assign_ext(ctx, range, &baby_bear, actual.final_acc);
    let zero = baby_bear.ext_zero(ctx, range);
    let one = ext_from_base_const(ctx, range, &baby_bear, 1);

    let round_count = assign_and_range_usize(ctx, range, actual.query_counts_per_round.len());
    let folding_rounds = assign_and_range_usize(ctx, range, actual.folding_counts_per_round.len());
    ctx.constrain_equal(&round_count, &folding_rounds);
    let gamma_count = assign_and_range_usize(ctx, range, actual.gammas.len());
    ctx.constrain_equal(&round_count, &gamma_count);
    let z0_count = assign_and_range_usize(ctx, range, actual.z0_challenges.len());
    gate.assert_is_const(
        ctx,
        &z0_count,
        &Fr::from(usize_to_u64(
            actual.query_counts_per_round.len().saturating_sub(1),
        )),
    );
    let ood_count = assign_and_range_usize(ctx, range, ood_values.len());
    gate.assert_is_const(
        ctx,
        &ood_count,
        &Fr::from(usize_to_u64(
            actual.query_counts_per_round.len().saturating_sub(1),
        )),
    );
    let folding_alpha_count = assign_and_range_usize(ctx, range, folding_alphas.len());
    gate.assert_is_const(
        ctx,
        &folding_alpha_count,
        &Fr::from(usize_to_u64(
            actual.folding_counts_per_round.iter().sum::<usize>(),
        )),
    );
    if let Some(&first_query_bits) = actual.query_index_bits.first() {
        let first_query_bits_cell = assign_and_range_usize(ctx, range, first_query_bits);
        gate.assert_is_const(
            ctx,
            &first_query_bits_cell,
            &Fr::from(usize_to_u64(
                actual
                    .initial_log_rs_domain_size
                    .saturating_sub(actual.k_whir),
            )),
        );
    }

    let total_width = stacking_openings.iter().map(Vec::len).sum::<usize>();
    let mut mu_pows = Vec::with_capacity(total_width);
    let mut mu_pow = one.clone();
    for _ in 0..total_width {
        mu_pows.push(mu_pow.clone());
        mu_pow = baby_bear.ext_mul(ctx, range, &mu_pow, &mu_challenge);
    }

    let mut derived_claim = baby_bear.ext_zero(ctx, range);
    let mut mu_idx = 0usize;
    for commit_openings in &stacking_openings {
        for opening in commit_openings {
            let weighted = baby_bear.ext_mul(ctx, range, opening, &mu_pows[mu_idx]);
            derived_claim = baby_bear.ext_add(ctx, range, &derived_claim, &weighted);
            mu_idx += 1;
        }
    }
    let consumed_mu_pows = assign_and_range_usize(ctx, range, mu_idx);
    gate.assert_is_const(ctx, &consumed_mu_pows, &Fr::from(usize_to_u64(total_width)));

    let mut alpha_cursor = 0usize;
    let mut sumcheck_cursor = 0usize;
    let mut payload_cursor = 0usize;
    let mut query_cursor = 0usize;
    let mut log_rs_domain_size = actual.initial_log_rs_domain_size;
    let mut zs_per_round = Vec::with_capacity(actual.query_counts_per_round.len());
    let zero_base = baby_bear.zero(ctx, range);
    let default_merkle_payload = AssignedMerklePathPayload {
        leaf_values: vec![vec![zero_base.clone()]],
        query_bits: Vec::new(),
    };

    for (round_idx, &num_queries) in actual.query_counts_per_round.iter().enumerate() {
        let fold_count = actual.folding_counts_per_round[round_idx];
        let alphas_round = &folding_alphas[alpha_cursor..alpha_cursor + fold_count];
        alpha_cursor += fold_count;

        for alpha in alphas_round {
            let evals = whir_sumcheck_polys.get(sumcheck_cursor);
            let eval_count = assign_and_range_usize(ctx, range, evals.map_or(0usize, Vec::len));
            gate.assert_is_const(ctx, &eval_count, &Fr::from(2u64));
            let ev1 = evals
                .and_then(|round| round.first())
                .cloned()
                .unwrap_or_else(|| zero.clone());
            let ev2 = evals
                .and_then(|round| round.get(1))
                .cloned()
                .unwrap_or_else(|| zero.clone());
            let ev0 = baby_bear.ext_sub(ctx, range, &derived_claim, &ev1);
            derived_claim = interpolate_quadratic_at_012_assigned(
                ctx,
                range,
                &baby_bear,
                [&ev0, &ev1, &ev2],
                alpha,
            );
            sumcheck_cursor += 1;
        }

        let mut ys_round = Vec::with_capacity(num_queries);
        let mut zs_round = Vec::with_capacity(num_queries);

        for _ in 0..num_queries {
            let query_bits_len = *actual.query_index_bits.get(query_cursor).unwrap_or(&0usize);
            let query_bits_len_cell = assign_and_range_usize(ctx, range, query_bits_len);
            gate.assert_is_const(
                ctx,
                &query_bits_len_cell,
                &Fr::from(usize_to_u64(
                    log_rs_domain_size.saturating_sub(actual.k_whir),
                )),
            );
            let query_bit_source = merkle_payloads
                .get(payload_cursor)
                .unwrap_or(&default_merkle_payload);
            let query_bit_count =
                assign_and_range_usize(ctx, range, query_bit_source.query_bits.len());
            gate.assert_is_const(
                ctx,
                &query_bit_count,
                &Fr::from(usize_to_u64(query_bits_len)),
            );
            let zi_root_base = query_root_from_bits_assigned(
                ctx,
                range,
                &baby_bear,
                &query_bit_source.query_bits,
                log_rs_domain_size,
            );
            let zi_root_ext = ext_from_base_var(ctx, range, &baby_bear, &zi_root_base);
            let zi = ext_pow_power_of_two(ctx, range, &baby_bear, &zi_root_ext, actual.k_whir);
            zs_round.push(zi);

            let yi = if round_idx == 0 {
                let row_count = 1usize << fold_count;
                let mut codeword_vals = vec![baby_bear.ext_zero(ctx, range); row_count];
                let mut mu_power_idx = 0usize;

                for (commit_idx, commit_openings) in stacking_openings.iter().enumerate() {
                    let payload = merkle_payloads
                        .get(payload_cursor + commit_idx)
                        .unwrap_or(&default_merkle_payload);
                    let opened_row_count =
                        assign_and_range_usize(ctx, range, payload.leaf_values.len());
                    gate.assert_is_const(
                        ctx,
                        &opened_row_count,
                        &Fr::from(usize_to_u64(row_count)),
                    );
                    let payload_query_bits_len =
                        assign_and_range_usize(ctx, range, payload.query_bits.len());
                    gate.assert_is_const(
                        ctx,
                        &payload_query_bits_len,
                        &Fr::from(usize_to_u64(query_bits_len)),
                    );
                    for (&lhs, &rhs) in payload
                        .query_bits
                        .iter()
                        .zip(query_bit_source.query_bits.iter())
                    {
                        ctx.constrain_equal(&lhs, &rhs);
                    }
                    for row in &payload.leaf_values {
                        let row_width = assign_and_range_usize(ctx, range, row.len());
                        gate.assert_is_const(
                            ctx,
                            &row_width,
                            &Fr::from(usize_to_u64(commit_openings.len())),
                        );
                    }

                    for col_idx in 0..commit_openings.len() {
                        let mu_pow = &mu_pows[mu_power_idx];
                        for (row_idx, row) in payload.leaf_values.iter().enumerate() {
                            let opened_base = row
                                .get(col_idx)
                                .cloned()
                                .unwrap_or_else(|| zero_base.clone());
                            let opened_ext =
                                ext_from_base_var(ctx, range, &baby_bear, &opened_base);
                            let weighted = baby_bear.ext_mul(ctx, range, &opened_ext, mu_pow);
                            codeword_vals[row_idx] =
                                baby_bear.ext_add(ctx, range, &codeword_vals[row_idx], &weighted);
                        }
                        mu_power_idx += 1;
                    }
                }
                let consumed_mu_power_idx = assign_and_range_usize(ctx, range, mu_power_idx);
                gate.assert_is_const(
                    ctx,
                    &consumed_mu_power_idx,
                    &Fr::from(usize_to_u64(mu_pows.len())),
                );
                payload_cursor += stacking_openings.len();
                binary_k_fold_assigned(
                    ctx,
                    range,
                    &baby_bear,
                    codeword_vals,
                    alphas_round,
                    &zi_root_base,
                )
            } else {
                let payload = merkle_payloads
                    .get(payload_cursor)
                    .unwrap_or(&default_merkle_payload);
                payload_cursor += 1;
                let non_initial_row_count =
                    assign_and_range_usize(ctx, range, payload.leaf_values.len());
                gate.assert_is_const(
                    ctx,
                    &non_initial_row_count,
                    &Fr::from(usize_to_u64(1usize << fold_count)),
                );
                let opened_values = payload
                    .leaf_values
                    .iter()
                    .map(|row| {
                        let row_width = assign_and_range_usize(ctx, range, row.len());
                        gate.assert_is_const(
                            ctx,
                            &row_width,
                            &Fr::from(usize_to_u64(BABY_BEAR_EXT_DEGREE)),
                        );
                        BabyBearExtVar {
                            coeffs: core::array::from_fn(|idx| {
                                row.get(idx).cloned().unwrap_or_else(|| zero_base.clone())
                            }),
                        }
                    })
                    .collect::<Vec<_>>();
                binary_k_fold_assigned(
                    ctx,
                    range,
                    &baby_bear,
                    opened_values,
                    alphas_round,
                    &zi_root_base,
                )
            };
            ys_round.push(yi);
            query_cursor += 1;
        }

        let gamma = &gammas[round_idx];
        if round_idx + 1 != actual.query_counts_per_round.len() {
            let y0_term = baby_bear.ext_mul(ctx, range, &ood_values[round_idx], gamma);
            derived_claim = baby_bear.ext_add(ctx, range, &derived_claim, &y0_term);
        }
        let mut gamma_pow = baby_bear.ext_mul(ctx, range, gamma, gamma);
        for yi in &ys_round {
            let term = baby_bear.ext_mul(ctx, range, yi, &gamma_pow);
            derived_claim = baby_bear.ext_add(ctx, range, &derived_claim, &term);
            gamma_pow = baby_bear.ext_mul(ctx, range, &gamma_pow, gamma);
        }

        zs_per_round.push(zs_round);
        log_rs_domain_size = log_rs_domain_size.saturating_sub(1);
    }

    let consumed_alpha_count = assign_and_range_usize(ctx, range, alpha_cursor);
    gate.assert_is_const(
        ctx,
        &consumed_alpha_count,
        &Fr::from(usize_to_u64(folding_alphas.len())),
    );
    let consumed_sumcheck_count = assign_and_range_usize(ctx, range, sumcheck_cursor);
    gate.assert_is_const(
        ctx,
        &consumed_sumcheck_count,
        &Fr::from(usize_to_u64(whir_sumcheck_polys.len())),
    );
    let consumed_payload_count = assign_and_range_usize(ctx, range, payload_cursor);
    gate.assert_is_const(
        ctx,
        &consumed_payload_count,
        &Fr::from(usize_to_u64(merkle_payloads.len())),
    );
    let consumed_query_count = assign_and_range_usize(ctx, range, query_cursor);
    gate.assert_is_const(
        ctx,
        &consumed_query_count,
        &Fr::from(usize_to_u64(query_indices.len())),
    );
    baby_bear.assert_ext_equal(ctx, &derived_claim, &final_claim);

    let rounds = actual.query_counts_per_round.len();
    let t = actual.k_whir * rounds;
    assert!(
        folding_alphas.len() >= t,
        "folding alpha vector must cover k_whir * rounds prefix",
    );
    assert!(
        u_cube.len() >= t,
        "u_cube vector must cover k_whir * rounds prefix",
    );
    let prefix =
        eval_mobius_eq_mle_assigned(ctx, range, &baby_bear, &u_cube[..t], &folding_alphas[..t]);
    let suffix =
        eval_mle_evals_at_point_assigned(ctx, range, &baby_bear, &final_poly, &u_cube[t..]);
    let mut derived_final_acc = baby_bear.ext_mul(ctx, range, &prefix, &suffix);

    let mut alpha_offset = actual.k_whir;
    for round_idx in 0..rounds {
        let gamma = &gammas[round_idx];
        let alpha_slc = &folding_alphas[alpha_offset..t];
        let slc_len = (t - alpha_offset) + 1;

        if round_idx + 1 != rounds {
            let z0 = &z0_challenges[round_idx];
            let mut z0_pows = Vec::with_capacity(slc_len);
            z0_pows.push(z0.clone());
            for _ in 1..slc_len {
                let next = baby_bear.ext_mul(
                    ctx,
                    range,
                    z0_pows.last().expect("z0 power sequence is non-empty"),
                    z0_pows.last().expect("z0 power sequence is non-empty"),
                );
                z0_pows.push(next);
            }
            let z0_max = z0_pows
                .last()
                .cloned()
                .expect("z0 power sequence is non-empty");
            let eq = eval_eq_mle_assigned(
                ctx,
                range,
                &baby_bear,
                alpha_slc,
                &z0_pows[..z0_pows.len().saturating_sub(1)],
            );
            let poly_eval =
                horner_eval_ext_poly_assigned(ctx, range, &baby_bear, &final_poly, &z0_max);
            let term = baby_bear.ext_mul(ctx, range, gamma, &eq);
            let term = baby_bear.ext_mul(ctx, range, &term, &poly_eval);
            derived_final_acc = baby_bear.ext_add(ctx, range, &derived_final_acc, &term);
        }

        let mut gamma_pow = baby_bear.ext_mul(ctx, range, gamma, gamma);
        for zi in &zs_per_round[round_idx] {
            let mut zi_pows = Vec::with_capacity(slc_len);
            zi_pows.push(zi.clone());
            for _ in 1..slc_len {
                let next = baby_bear.ext_mul(
                    ctx,
                    range,
                    zi_pows.last().expect("zi power sequence is non-empty"),
                    zi_pows.last().expect("zi power sequence is non-empty"),
                );
                zi_pows.push(next);
            }
            let zi_max = zi_pows
                .last()
                .cloned()
                .expect("zi power sequence is non-empty");
            let eq = eval_eq_mle_assigned(
                ctx,
                range,
                &baby_bear,
                alpha_slc,
                &zi_pows[..zi_pows.len().saturating_sub(1)],
            );
            let poly_eval =
                horner_eval_ext_poly_assigned(ctx, range, &baby_bear, &final_poly, &zi_max);
            let term = baby_bear.ext_mul(ctx, range, &gamma_pow, &eq);
            let term = baby_bear.ext_mul(ctx, range, &term, &poly_eval);
            derived_final_acc = baby_bear.ext_add(ctx, range, &derived_final_acc, &term);
            gamma_pow = baby_bear.ext_mul(ctx, range, &gamma_pow, gamma);
        }

        alpha_offset += actual.k_whir;
    }
    baby_bear.assert_ext_equal(ctx, &derived_final_acc, &final_acc);

    let final_residual = assign_ext(ctx, range, &baby_bear, actual.final_residual);
    let derived_final_residual = baby_bear.ext_sub(ctx, range, &final_acc, &final_claim);
    baby_bear.assert_ext_equal(ctx, &derived_final_residual, &final_residual);

    baby_bear.assert_ext_equal(ctx, &final_residual, &zero);

    AssignedWhirIntermediates {
        mu_pow_bits,
        mu_pow_sampled_bits,
        mu_pow_witness_ok,
        mu_challenge,
        folding_pow_bits,
        folding_pow_sampled_bits,
        folding_pow_witness_ok,
        folding_alphas,
        z0_challenges,
        query_phase_pow_bits,
        query_phase_pow_sampled_bits,
        query_phase_pow_witness_ok,
        gammas,
        query_indices,
        whir_sumcheck_polys,
        stacking_openings,
        initial_commitment_roots,
        codeword_commitment_roots,
        ood_values,
        final_poly,
        final_poly_len,
        u_cube,
        final_claim,
        final_acc,
        final_residual,
    }
}

// Internal standalone whir wrapper for intra-crate composition/tests.
pub(crate) fn derive_and_constrain_whir(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedWhirIntermediates, WhirError> {
    let raw = derive_raw_whir_witness_state(config, mvk, proof)?;
    let ownership = derive_whir_strict_ownership(config, mvk, proof)?;
    Ok(constrain_checked_whir_witness_state_strict(ctx, range, &raw, &ownership).assigned)
}

pub(crate) fn derive_raw_whir_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawWhirWitnessState, WhirError> {
    Ok(RawWhirWitnessState {
        intermediates: derive_whir_intermediates(config, mvk, proof)?,
    })
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_checked_whir_witness_state_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawWhirWitnessState,
) -> CheckedWhirWitnessState {
    let assigned = constrain_whir_intermediates_unchecked(ctx, range, &raw.intermediates);
    let derived = DerivedWhirState {
        final_acc: assigned.final_acc.clone(),
        final_claim: assigned.final_claim.clone(),
        final_residual: assigned.final_residual.clone(),
    };
    CheckedWhirWitnessState { assigned, derived }
}

pub(crate) fn derive_whir_strict_ownership(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<WhirStrictOwnership, WhirError> {
    let expected = derive_whir_intermediates(config, mvk, proof)?;
    Ok(WhirStrictOwnership {
        k_whir: expected.k_whir,
        initial_log_rs_domain_size: expected.initial_log_rs_domain_size,
        mu_pow_bits: expected.mu_pow_bits,
        folding_pow_bits: expected.folding_pow_bits,
        query_phase_pow_bits: expected.query_phase_pow_bits,
        folding_counts_per_round: expected.folding_counts_per_round,
        query_counts_per_round: expected.query_counts_per_round,
        query_index_bits: expected.query_index_bits,
        expected_final_poly_len: expected.expected_final_poly_len,
    })
}

fn constrain_whir_strict_metadata(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawWhirWitnessState,
    assigned: &AssignedWhirIntermediates,
    ownership: &WhirStrictOwnership,
) {
    let gate = range.gate();
    let k_whir = assign_and_range_usize(ctx, range, raw.intermediates.k_whir);
    gate.assert_is_const(ctx, &k_whir, &Fr::from(usize_to_u64(ownership.k_whir)));

    let initial_log_rs_domain_size =
        assign_and_range_usize(ctx, range, raw.intermediates.initial_log_rs_domain_size);
    gate.assert_is_const(
        ctx,
        &initial_log_rs_domain_size,
        &Fr::from(usize_to_u64(ownership.initial_log_rs_domain_size)),
    );

    gate.assert_is_const(
        ctx,
        &assigned.mu_pow_bits,
        &Fr::from(usize_to_u64(ownership.mu_pow_bits)),
    );
    gate.assert_is_const(
        ctx,
        &assigned.folding_pow_bits,
        &Fr::from(usize_to_u64(ownership.folding_pow_bits)),
    );
    gate.assert_is_const(
        ctx,
        &assigned.query_phase_pow_bits,
        &Fr::from(usize_to_u64(ownership.query_phase_pow_bits)),
    );
    gate.assert_is_const(
        ctx,
        &assigned.final_poly_len,
        &Fr::from(usize_to_u64(ownership.expected_final_poly_len)),
    );

    assert_eq!(
        raw.intermediates.folding_counts_per_round.len(),
        ownership.folding_counts_per_round.len(),
        "WHIR strict folding-count schedule length mismatch",
    );
    for (&actual_count, &expected_count) in raw
        .intermediates
        .folding_counts_per_round
        .iter()
        .zip(ownership.folding_counts_per_round.iter())
    {
        let count = assign_and_range_usize(ctx, range, actual_count);
        gate.assert_is_const(ctx, &count, &Fr::from(usize_to_u64(expected_count)));
    }

    assert_eq!(
        raw.intermediates.query_counts_per_round.len(),
        ownership.query_counts_per_round.len(),
        "WHIR strict query-count schedule length mismatch",
    );
    for (&actual_count, &expected_count) in raw
        .intermediates
        .query_counts_per_round
        .iter()
        .zip(ownership.query_counts_per_round.iter())
    {
        let count = assign_and_range_usize(ctx, range, actual_count);
        gate.assert_is_const(ctx, &count, &Fr::from(usize_to_u64(expected_count)));
    }

    assert_eq!(
        raw.intermediates.query_index_bits.len(),
        ownership.query_index_bits.len(),
        "WHIR strict query-index-bit schedule length mismatch",
    );
    for (&actual_bits, &expected_bits) in raw
        .intermediates
        .query_index_bits
        .iter()
        .zip(ownership.query_index_bits.iter())
    {
        let bits = assign_and_range_usize(ctx, range, actual_bits);
        gate.assert_is_const(ctx, &bits, &Fr::from(usize_to_u64(expected_bits)));
    }
}

pub(crate) fn constrain_checked_whir_witness_state_strict(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawWhirWitnessState,
    ownership: &WhirStrictOwnership,
) -> CheckedWhirWitnessState {
    let checked = constrain_checked_whir_witness_state_unchecked(ctx, range, raw);
    constrain_whir_strict_metadata(ctx, range, raw, &checked.assigned, ownership);
    checked
}

pub fn coeffs_to_native_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> NativeEF {
    coeffs_to_ext(coeffs)
}

#[cfg(test)]
mod tests;
