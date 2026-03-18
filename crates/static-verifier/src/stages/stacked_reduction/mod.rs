use std::iter::zip;

use halo2_base::{
    gates::{GateInstructions, RangeInstructions},
    Context,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        default_transcript, BabyBearBn254Poseidon2Config as NativeConfig,
    },
    openvm_stark_backend::{
        keygen::types::MultiStarkVerifyingKey,
        p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
        poly_common::{
            eval_eq_mle, eval_eq_prism, eval_in_uni, eval_rot_kernel_prism, horner_eval,
            interpolate_quadratic_at_012,
        },
        proof::{column_openings_by_rot, Proof, StackingProof},
        prover::stacked_pcs::StackedLayout,
        verifier::{
            batch_constraints::BatchConstraintError as NativeBatchConstraintError,
            proof_shape::ProofShapeError, stacked_reduction::StackedReductionError,
        },
        FiatShamirTranscript,
    },
};

use crate::{
    field::baby_bear::{BabyBearExtChip, BabyBearExtWire, BABY_BEAR_EXT_DEGREE},
    stages::{
        batch_constraints::{
            eval_eq_mle_binary_assigned, eval_eq_prism_assigned, eval_eq_uni_at_one_assigned,
            eval_rot_kernel_prism_assigned, BatchConstraintError,
        },
        full_pipeline::witness::prepare_pipeline_inputs,
        shared_math::{
            column_openings_by_rot_assigned, horner_eval_ext_poly_assigned,
            interpolate_quadratic_at_012_assigned,
        },
    },
    transcript::TranscriptGadget,
    ChildEF, ChildF, Fr,
};

#[derive(Debug, PartialEq, Eq)]
pub enum StackedReductionConstraintError {
    SystemParamsMismatch,
    TraceHeightsTooLarge,
    ProofShape(ProofShapeError),
    BatchConstraint(NativeBatchConstraintError<ChildEF>),
    StackedReduction(StackedReductionError<ChildEF>),
    BatchSetup(BatchConstraintError),
}

impl From<ProofShapeError> for StackedReductionConstraintError {
    fn from(value: ProofShapeError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<NativeBatchConstraintError<ChildEF>> for StackedReductionConstraintError {
    fn from(value: NativeBatchConstraintError<ChildEF>) -> Self {
        Self::BatchConstraint(value)
    }
}

impl From<StackedReductionError<ChildEF>> for StackedReductionConstraintError {
    fn from(value: StackedReductionError<ChildEF>) -> Self {
        Self::StackedReduction(value)
    }
}

impl From<BatchConstraintError> for StackedReductionConstraintError {
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
pub struct QCoeffAccumulationTerm {
    pub commit_idx: usize,
    pub target_col_idx: usize,
    pub lambda_idx: usize,
    pub need_rot: bool,
    pub n: isize,
    pub b_bits: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackedReductionIntermediates {
    pub l_skip: usize,
    pub lambda: ChildEF,
    pub batch_column_openings: Vec<Vec<Vec<ChildEF>>>,
    pub batch_column_openings_need_rot: Vec<Vec<bool>>,
    pub q_coeff_terms: Vec<QCoeffAccumulationTerm>,
    pub r: Vec<ChildEF>,
    pub univariate_round_coeffs: Vec<ChildEF>,
    pub sumcheck_round_polys: Vec<Vec<ChildEF>>,
    pub stacking_openings: Vec<Vec<ChildEF>>,
    pub q_coeffs: Vec<Vec<ChildEF>>,
    pub stacking_matrix_expected_widths: Vec<usize>,
    pub s_0: ChildEF,
    pub s_0_sum_eval: ChildEF,
    pub s_0_residual: ChildEF,
    pub final_claim: ChildEF,
    pub final_sum: ChildEF,
    pub final_residual: ChildEF,
    pub u: Vec<ChildEF>,
}

#[derive(Clone, Debug)]
pub struct AssignedStackedReductionIntermediates {
    pub lambda: BabyBearExtWire,
    pub univariate_round_coeffs: Vec<BabyBearExtWire>,
    pub sumcheck_round_polys: Vec<Vec<BabyBearExtWire>>,
    pub batch_column_openings: Vec<Vec<Vec<BabyBearExtWire>>>,
    pub r: Vec<BabyBearExtWire>,
    pub stacking_openings: Vec<Vec<BabyBearExtWire>>,
    pub s_0: BabyBearExtWire,
    pub s_0_sum_eval: BabyBearExtWire,
    pub s_0_residual: BabyBearExtWire,
    pub final_claim: BabyBearExtWire,
    pub final_sum: BabyBearExtWire,
    pub final_residual: BabyBearExtWire,
    pub u: Vec<BabyBearExtWire>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawStackedWitnessState {
    pub intermediates: StackedReductionIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedStackedState {
    pub s_0_residual: BabyBearExtWire,
    pub final_residual: BabyBearExtWire,
}

#[derive(Clone, Debug)]
pub struct CheckedStackedWitnessState {
    pub assigned: AssignedStackedReductionIntermediates,
    pub derived: DerivedStackedState,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn derive_stacked_reduction_intermediates_with_inputs(
    transcript: &mut impl FiatShamirTranscript<NativeConfig>,
    proof: &StackingProof<NativeConfig>,
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
    column_openings: &Vec<Vec<Vec<ChildEF>>>,
    r: &[ChildEF],
    omega_shift_pows: &[ChildF],
) -> Result<StackedReductionIntermediates, StackedReductionError<ChildEF>> {
    let omega_order = omega_shift_pows.len();
    let omega_order_f = ChildF::from_usize(omega_order);

    let mut lambda_idx = 0usize;
    let lambda_indices_per_layout: Vec<Vec<(usize, bool)>> = layouts
        .iter()
        .enumerate()
        .map(|(commit_idx, layout)| {
            let need_rot_for_commit = &need_rot_per_commit[commit_idx];
            layout
                .sorted_cols
                .iter()
                .map(|&(mat_idx, _col_idx, _slice)| {
                    lambda_idx += 1;
                    (lambda_idx - 1, need_rot_for_commit[mat_idx])
                })
                .collect()
        })
        .collect();
    let t_claims_len = lambda_idx;
    let mut t_claims = Vec::with_capacity(t_claims_len);

    for (trace_idx, parts) in column_openings.iter().enumerate() {
        let need_rot = need_rot_per_commit[0][trace_idx];
        t_claims.extend(column_openings_by_rot(&parts[0], need_rot));
    }

    let mut commit_idx = 1usize;
    for parts in column_openings {
        for cols in parts.iter().skip(1) {
            let need_rot = need_rot_per_commit[commit_idx][0];
            t_claims.extend(column_openings_by_rot(cols, need_rot));
            commit_idx += 1;
        }
    }

    assert_eq!(t_claims.len(), t_claims_len);

    let lambda = transcript.sample_ext();
    let lambda_sqr_powers: Vec<_> = (lambda * lambda).powers().take(t_claims_len).collect();

    let s_0 = zip(&t_claims, &lambda_sqr_powers)
        .map(|(&t_i, &lambda_i)| (t_i.0 + t_i.1 * lambda) * lambda_i)
        .sum::<ChildEF>();
    let s_0_sum_eval = proof
        .univariate_round_coeffs
        .iter()
        .step_by(omega_order)
        .copied()
        .sum::<ChildEF>()
        * omega_order_f;

    let s_0_residual = s_0 - s_0_sum_eval;
    if s_0_residual != ChildEF::ZERO {
        return Err(StackedReductionError::S0Mismatch { s_0, s_0_sum_eval });
    }

    for coeff in &proof.univariate_round_coeffs {
        transcript.observe_ext(*coeff);
    }

    let mut u = vec![ChildEF::ZERO; n_stack + 1];
    u[0] = transcript.sample_ext();

    let mut s_j_0 = s_0;
    let mut claim = horner_eval(&proof.univariate_round_coeffs, u[0]);

    u.iter_mut().enumerate().skip(1).for_each(|(j, u_j)| {
        let s_j_1 = proof.sumcheck_round_polys[j - 1][0];
        let s_j_2 = proof.sumcheck_round_polys[j - 1][1];
        transcript.observe_ext(s_j_1);
        transcript.observe_ext(s_j_2);
        *u_j = transcript.sample_ext();
        s_j_0 = claim - s_j_1;
        claim = interpolate_quadratic_at_012(&[s_j_0, s_j_1, s_j_2], *u_j);
    });

    let mut q_coeffs: Vec<Vec<ChildEF>> = proof
        .stacking_openings
        .iter()
        .map(|vec| vec![ChildEF::ZERO; vec.len()])
        .collect();
    let mut q_coeff_terms = Vec::new();

    layouts
        .iter()
        .enumerate()
        .zip(q_coeffs.iter_mut())
        .for_each(|((commit_idx, layout), coeffs)| {
            let lambda_indices = &lambda_indices_per_layout[commit_idx];
            layout
                .sorted_cols
                .iter()
                .enumerate()
                .for_each(|(col_idx, &(_, _, s))| {
                    let (lambda_idx, need_rot) = lambda_indices[col_idx];
                    let n = s.log_height() as isize - l_skip as isize;
                    let n_lift = n.max(0) as usize;
                    let b: Vec<_> = (l_skip + n_lift..l_skip + n_stack)
                        .map(|j| ChildF::from_bool((s.row_idx >> j) & 1 == 1))
                        .collect();
                    let b_bits = (l_skip + n_lift..l_skip + n_stack)
                        .map(|j| ((s.row_idx >> j) & 1) == 1)
                        .collect::<Vec<_>>();
                    let eq_mle = eval_eq_mle(&u[n_lift + 1..], &b);
                    let ind = eval_in_uni(l_skip, n, u[0]);
                    let (l, rs_n) = if n.is_negative() {
                        (
                            l_skip.wrapping_add_signed(n),
                            &[r[0].exp_power_of_2(-n as usize)] as &[_],
                        )
                    } else {
                        (l_skip, &r[..=n_lift])
                    };
                    let eq_prism = eval_eq_prism(l, &u[..=n_lift], rs_n);
                    let mut batched = lambda_sqr_powers[lambda_idx] * eq_prism;
                    if need_rot {
                        let rot_kernel_prism = eval_rot_kernel_prism(l, &u[..=n_lift], rs_n);
                        batched += lambda_sqr_powers[lambda_idx] * lambda * rot_kernel_prism;
                    }
                    coeffs[s.col_idx] += eq_mle * batched * ind;
                    q_coeff_terms.push(QCoeffAccumulationTerm {
                        commit_idx,
                        target_col_idx: s.col_idx,
                        lambda_idx,
                        need_rot,
                        n,
                        b_bits,
                    });
                });
        });

    let final_sum = q_coeffs.iter().zip(proof.stacking_openings.iter()).fold(
        ChildEF::ZERO,
        |acc, (q_coeff_vec, q_j_vec)| {
            acc + q_coeff_vec.iter().zip(q_j_vec.iter()).fold(
                ChildEF::ZERO,
                |acc, (&q_coeff, &q_j)| {
                    transcript.observe_ext(q_j);
                    acc + (q_coeff * q_j)
                },
            )
        },
    );

    let final_residual = claim - final_sum;
    if final_residual != ChildEF::ZERO {
        return Err(StackedReductionError::FinalSumMismatch { claim, final_sum });
    }

    Ok(StackedReductionIntermediates {
        l_skip,
        lambda,
        batch_column_openings: column_openings.clone(),
        batch_column_openings_need_rot: need_rot_per_commit.to_vec(),
        q_coeff_terms,
        r: r.to_vec(),
        univariate_round_coeffs: proof.univariate_round_coeffs.clone(),
        sumcheck_round_polys: proof
            .sumcheck_round_polys
            .iter()
            .map(|arr| arr.to_vec())
            .collect(),
        stacking_openings: proof.stacking_openings.clone(),
        q_coeffs,
        stacking_matrix_expected_widths: layouts
            .iter()
            .map(|layout| {
                layout
                    .sorted_cols
                    .last()
                    .map(|(_, _, slice)| slice.col_idx + 1)
                    .expect("stacked layout must contain at least one column")
            })
            .collect::<Vec<_>>(),
        s_0,
        s_0_sum_eval,
        s_0_residual,
        final_claim: claim,
        final_sum,
        final_residual,
        u,
    })
}

pub fn derive_stacked_reduction_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<StackedReductionIntermediates, StackedReductionConstraintError> {
    let mut transcript = default_transcript();
    let prepared = prepare_pipeline_inputs(&mut transcript, config, mvk, proof)?;

    derive_stacked_reduction_intermediates_with_inputs(
        &mut transcript,
        &proof.stacking_proof,
        &prepared.layouts,
        &prepared.need_rot_per_commit,
        prepared.l_skip,
        mvk.inner.params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &prepared.r,
        &prepared.omega_skip_pows,
    )
    .map_err(Into::into)
}

fn assign_ext(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    value: ChildEF,
) -> BabyBearExtWire {
    ext_chip.load_witness(ctx, value)
}

fn eval_in_uni_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    l_skip: usize,
    n: isize,
    z: &BabyBearExtWire,
) -> BabyBearExtWire {
    if n.is_negative() {
        let z_pow = ext_chip.pow_power_of_two(ctx, z, l_skip.wrapping_add_signed(n));
        eval_eq_uni_at_one_assigned(ctx, ext_chip, n.unsigned_abs(), &z_pow)
    } else {
        ext_chip.from_base_const(ctx, ChildF::from_u64(1))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_stacked_reduction_from_proof_inputs(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    transcript: &mut TranscriptGadget,
    proof: &StackingProof<NativeConfig>,
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
    batch_column_openings: &[Vec<Vec<BabyBearExtWire>>],
    r: &[BabyBearExtWire],
) -> AssignedStackedReductionIntermediates {
    let omega_order = 1usize << l_skip;
    let one = ext_chip.from_base_const(ctx, ChildF::ONE);

    let mut lambda_idx = 0usize;
    let lambda_indices_per_layout = layouts
        .iter()
        .enumerate()
        .map(|(commit_idx, layout)| {
            let need_rot_for_commit = &need_rot_per_commit[commit_idx];
            layout
                .sorted_cols
                .iter()
                .map(|&(mat_idx, _col_idx, _slice)| {
                    lambda_idx += 1;
                    (lambda_idx - 1, need_rot_for_commit[mat_idx])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut t_claims = Vec::with_capacity(lambda_idx);
    for (trace_idx, parts) in batch_column_openings.iter().enumerate() {
        let need_rot = need_rot_per_commit[0][trace_idx];
        t_claims.extend(column_openings_by_rot_assigned(
            ctx, ext_chip, &parts[0], need_rot,
        ));
    }
    let mut commit_idx = 1usize;
    for parts in batch_column_openings {
        for cols in parts.iter().skip(1) {
            let need_rot = need_rot_per_commit[commit_idx][0];
            t_claims.extend(column_openings_by_rot_assigned(
                ctx, ext_chip, cols, need_rot,
            ));
            commit_idx += 1;
        }
    }

    let lambda = transcript.sample_ext(ctx, ext_chip.range(), ext_chip.base());
    let lambda_sqr = ext_chip.mul(ctx, &lambda, &lambda);
    let mut lambda_sqr_powers = Vec::with_capacity(t_claims.len());
    let mut cur_lambda_sqr = one;
    for _ in 0..t_claims.len() {
        lambda_sqr_powers.push(cur_lambda_sqr);
        cur_lambda_sqr = ext_chip.mul(ctx, &cur_lambda_sqr, &lambda_sqr);
    }

    let mut s_0 = ext_chip.zero(ctx);
    for ((claim, claim_rot), lambda_pow) in t_claims.iter().zip(lambda_sqr_powers.iter()) {
        let claim_rot_lambda = ext_chip.mul(ctx, claim_rot, &lambda);
        let batched_claim = ext_chip.add(ctx, claim, &claim_rot_lambda);
        let term = ext_chip.mul(ctx, &batched_claim, lambda_pow);
        s_0 = ext_chip.add(ctx, &s_0, &term);
    }

    let univariate_round_coeffs = proof
        .univariate_round_coeffs
        .iter()
        .map(|&value| assign_ext(ctx, ext_chip, value))
        .collect::<Vec<_>>();
    let mut s_0_sum_eval = ext_chip.zero(ctx);
    for coeff in univariate_round_coeffs.iter().step_by(omega_order) {
        s_0_sum_eval = ext_chip.add(ctx, &s_0_sum_eval, coeff);
    }
    let s_0_sum_eval =
        ext_chip.mul_base_const(ctx, &s_0_sum_eval, ChildF::from_u64(omega_order as u64));
    let s_0_residual = ext_chip.sub(ctx, &s_0, &s_0_sum_eval);
    let zero = ext_chip.zero(ctx);
    ext_chip.assert_equal(ctx, &s_0_residual, &zero);

    for coeff in &univariate_round_coeffs {
        transcript.observe_ext(ctx, ext_chip.range(), ext_chip.base(), coeff);
    }

    let mut u = Vec::with_capacity(n_stack + 1);
    u.push(transcript.sample_ext(ctx, ext_chip.range(), ext_chip.base()));

    let sumcheck_round_polys = proof
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut final_claim =
        horner_eval_ext_poly_assigned(ctx, ext_chip, &univariate_round_coeffs, &u[0]);
    for round_poly in &sumcheck_round_polys {
        let s_j_1 = round_poly[0];
        let s_j_2 = round_poly[1];
        transcript.observe_ext(ctx, ext_chip.range(), ext_chip.base(), &s_j_1);
        transcript.observe_ext(ctx, ext_chip.range(), ext_chip.base(), &s_j_2);
        let u_j = transcript.sample_ext(ctx, ext_chip.range(), ext_chip.base());
        let s_j_0 = ext_chip.sub(ctx, &final_claim, &s_j_1);
        final_claim =
            interpolate_quadratic_at_012_assigned(ctx, ext_chip, [&s_j_0, &s_j_1, &s_j_2], &u_j);
        u.push(u_j);
    }

    let stacking_matrix_expected_widths = layouts
        .iter()
        .map(|layout| {
            layout
                .sorted_cols
                .last()
                .map(|(_, _, slice)| slice.col_idx + 1)
                .expect("stacked layout must contain at least one column")
        })
        .collect::<Vec<_>>();
    let mut derived_q_coeffs = stacking_matrix_expected_widths
        .iter()
        .map(|&width| {
            core::iter::repeat_with(|| ext_chip.zero(ctx))
                .take(width)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for (commit_idx, layout) in layouts.iter().enumerate() {
        let lambda_indices = &lambda_indices_per_layout[commit_idx];
        for (col_idx, &(_, _, s)) in layout.sorted_cols.iter().enumerate() {
            let (lambda_idx, need_rot) = lambda_indices[col_idx];
            let n = s.log_height() as isize - l_skip as isize;
            let n_lift = n.max(0) as usize;
            let b_bits = (l_skip + n_lift..l_skip + n_stack)
                .map(|j| ((s.row_idx >> j) & 1) == 1)
                .collect::<Vec<_>>();
            let eq_mle = eval_eq_mle_binary_assigned(ctx, ext_chip, &u[n_lift + 1..], &b_bits);
            let ind = eval_in_uni_assigned(ctx, ext_chip, l_skip, n, &u[0]);
            let (l, rs_n) = if n.is_negative() {
                (
                    l_skip.wrapping_add_signed(n),
                    vec![ext_chip.pow_power_of_two(ctx, &r[0], n.unsigned_abs())],
                )
            } else {
                (l_skip, r[..=n_lift].to_vec())
            };
            let eq_prism = eval_eq_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
            let mut batched = ext_chip.mul(ctx, &lambda_sqr_powers[lambda_idx], &eq_prism);
            if need_rot {
                let rot_kernel =
                    eval_rot_kernel_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
                let lambda_rot = ext_chip.mul(ctx, &lambda, &rot_kernel);
                let rot_term = ext_chip.mul(ctx, &lambda_sqr_powers[lambda_idx], &lambda_rot);
                batched = ext_chip.add(ctx, &batched, &rot_term);
            }
            let batched_ind = ext_chip.mul(ctx, &batched, &ind);
            let coeff = ext_chip.mul(ctx, &eq_mle, &batched_ind);
            let updated = ext_chip.add(ctx, &derived_q_coeffs[commit_idx][s.col_idx], &coeff);
            derived_q_coeffs[commit_idx][s.col_idx] = updated;
        }
    }

    let stacking_openings = proof
        .stacking_openings
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut final_sum = ext_chip.zero(ctx);
    for (coeff_row, opening_row) in derived_q_coeffs.iter().zip(stacking_openings.iter()) {
        for (coeff, opening) in coeff_row.iter().zip(opening_row.iter()) {
            transcript.observe_ext(ctx, ext_chip.range(), ext_chip.base(), opening);
            let term = ext_chip.mul(ctx, coeff, opening);
            final_sum = ext_chip.add(ctx, &final_sum, &term);
        }
    }

    let final_residual = ext_chip.sub(ctx, &final_claim, &final_sum);
    ext_chip.assert_equal(ctx, &final_residual, &zero);

    AssignedStackedReductionIntermediates {
        lambda,
        univariate_round_coeffs,
        sumcheck_round_polys,
        batch_column_openings: batch_column_openings.to_vec(),
        r: r.to_vec(),
        stacking_openings,
        s_0,
        s_0_sum_eval,
        s_0_residual,
        final_claim,
        final_sum,
        final_residual,
        u,
    }
}

#[allow(dead_code)]
pub(crate) fn constrain_stacked_reduction_intermediates_with_shared_inputs(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    actual: &StackedReductionIntermediates,
    shared_batch_column_openings: Option<&[Vec<Vec<BabyBearExtWire>>]>,
    shared_r: Option<&[BabyBearExtWire]>,
) -> AssignedStackedReductionIntermediates {
    assert!(!actual.u.is_empty(), "stacked challenges must be non-empty");
    assert_eq!(
        actual.q_coeffs.len(),
        actual.stacking_openings.len(),
        "q-coeff layout must match stacking openings layout",
    );
    assert_eq!(
        actual.stacking_matrix_expected_widths.len(),
        actual.stacking_openings.len(),
        "stacked expected-width schedule must align with stacking openings layout",
    );
    assert_eq!(
        actual.stacking_matrix_expected_widths.len(),
        actual.q_coeffs.len(),
        "stacked expected-width schedule must align with q-coeff layout",
    );

    let lambda = assign_ext(ctx, ext_chip, actual.lambda);
    let u = actual
        .u
        .iter()
        .map(|&actual_u| assign_ext(ctx, ext_chip, actual_u))
        .collect::<Vec<_>>();
    let univariate_round_coeffs = actual
        .univariate_round_coeffs
        .iter()
        .map(|&value| assign_ext(ctx, ext_chip, value))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = actual
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let stacking_openings = actual
        .stacking_openings
        .iter()
        .map(|openings| {
            openings
                .iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let batch_column_openings = shared_batch_column_openings.map_or_else(
        || {
            actual
                .batch_column_openings
                .iter()
                .map(|per_air| {
                    per_air
                        .iter()
                        .map(|part| {
                            part.iter()
                                .map(|&value| assign_ext(ctx, ext_chip, value))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        },
        |shared| {
            assert_eq!(
                shared.len(),
                actual.batch_column_openings.len(),
                "shared batch openings must align with trace count",
            );
            shared.to_vec()
        },
    );
    let r = shared_r.map_or_else(
        || {
            actual
                .r
                .iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        },
        |shared| {
            assert_eq!(
                shared.len(),
                actual.r.len(),
                "shared challenge wires must align with stacked challenge count",
            );
            shared.to_vec()
        },
    );
    let q_coeffs = actual
        .q_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .map(|&value| assign_ext(ctx, ext_chip, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for ((opening_row, q_coeff_row), &expected_width) in actual
        .stacking_openings
        .iter()
        .zip(actual.q_coeffs.iter())
        .zip(actual.stacking_matrix_expected_widths.iter())
    {
        let opening_width = ext_chip
            .base()
            .assign_and_range_usize(ctx, opening_row.len());
        let q_coeff_width = ext_chip
            .base()
            .assign_and_range_usize(ctx, q_coeff_row.len());
        ext_chip.range().gate().assert_is_const(
            ctx,
            &opening_width,
            &Fr::from(expected_width as u64),
        );
        ext_chip.range().gate().assert_is_const(
            ctx,
            &q_coeff_width,
            &Fr::from(expected_width as u64),
        );
    }
    let s_0 = assign_ext(ctx, ext_chip, actual.s_0);

    let mut t_claims = Vec::new();
    let commit_need_rot = &actual.batch_column_openings_need_rot;
    assert!(
        !commit_need_rot.is_empty(),
        "stacked recomputation requires commit rotation metadata",
    );
    for (trace_idx, parts) in batch_column_openings.iter().enumerate() {
        let need_rot = *commit_need_rot[0]
            .get(trace_idx)
            .expect("common-main need_rot metadata must cover all traces");
        t_claims.extend(column_openings_by_rot_assigned(
            ctx, ext_chip, &parts[0], need_rot,
        ));
    }
    let mut commit_idx = 1usize;
    for parts in &batch_column_openings {
        for cols in parts.iter().skip(1) {
            let need_rot = *commit_need_rot[commit_idx]
                .first()
                .expect("non-common commit need_rot metadata must be singleton");
            t_claims.extend(column_openings_by_rot_assigned(
                ctx, ext_chip, cols, need_rot,
            ));
            commit_idx += 1;
        }
    }
    assert_eq!(
        commit_idx,
        commit_need_rot.len(),
        "all non-common commitments must be consumed when deriving t-claims",
    );

    let lambda_sqr = ext_chip.mul(ctx, &lambda, &lambda);
    let mut lambda_sqr_powers = Vec::with_capacity(t_claims.len());
    let mut cur_lambda_sqr = ext_chip.from_base_const(ctx, ChildF::from_u64(1));
    for _ in 0..t_claims.len() {
        lambda_sqr_powers.push(cur_lambda_sqr);
        cur_lambda_sqr = ext_chip.mul(ctx, &cur_lambda_sqr, &lambda_sqr);
    }
    let mut derived_s_0 = ext_chip.zero(ctx);
    for ((claim, claim_rot), lambda_pow) in t_claims.iter().zip(lambda_sqr_powers.iter()) {
        let claim_rot_lambda = ext_chip.mul(ctx, claim_rot, &lambda);
        let batched_claim = ext_chip.add(ctx, claim, &claim_rot_lambda);
        let term = ext_chip.mul(ctx, &batched_claim, lambda_pow);
        derived_s_0 = ext_chip.add(ctx, &derived_s_0, &term);
    }
    ext_chip.assert_equal(ctx, &derived_s_0, &s_0);

    let s_0_sum_eval = assign_ext(ctx, ext_chip, actual.s_0_sum_eval);
    let stride = 1usize << actual.l_skip;
    let mut derived_s_0_sum_eval = ext_chip.zero(ctx);
    for coeff in univariate_round_coeffs.iter().step_by(stride) {
        derived_s_0_sum_eval = ext_chip.add(ctx, &derived_s_0_sum_eval, coeff);
    }
    let derived_s_0_sum_eval =
        ext_chip.mul_base_const(ctx, &derived_s_0_sum_eval, ChildF::from_u64(stride as u64));
    ext_chip.assert_equal(ctx, &derived_s_0_sum_eval, &s_0_sum_eval);

    let s_0_residual = assign_ext(ctx, ext_chip, actual.s_0_residual);
    // Equation-first check: s_0_residual = s_0 - s_0_sum_eval.
    let derived_s_0_residual = ext_chip.sub(ctx, &s_0, &s_0_sum_eval);
    ext_chip.assert_equal(ctx, &derived_s_0_residual, &s_0_residual);

    assert!(
        !univariate_round_coeffs.is_empty(),
        "stacked univariate polynomial must have at least one coefficient",
    );
    assert_eq!(
        sumcheck_round_polys.len() + 1,
        u.len(),
        "stacked sumcheck rounds must align with sampled u challenges",
    );
    let final_claim = assign_ext(ctx, ext_chip, actual.final_claim);
    // Recompute native claim chain from constrained univariate/sumcheck payloads and u challenges.
    let mut derived_claim =
        horner_eval_ext_poly_assigned(ctx, ext_chip, &univariate_round_coeffs, &u[0]);
    for (round_idx, round_poly) in sumcheck_round_polys.iter().enumerate() {
        assert_eq!(
            round_poly.len(),
            2,
            "stacked sumcheck rounds must expose [s(1), s(2)]",
        );
        let s_j_1 = &round_poly[0];
        let s_j_2 = &round_poly[1];
        let s_j_0 = ext_chip.sub(ctx, &derived_claim, s_j_1);
        derived_claim = interpolate_quadratic_at_012_assigned(
            ctx,
            ext_chip,
            [&s_j_0, s_j_1, s_j_2],
            &u[round_idx + 1],
        );
    }
    ext_chip.assert_equal(ctx, &derived_claim, &final_claim);

    let final_sum = assign_ext(ctx, ext_chip, actual.final_sum);
    let mut derived_q_coeffs = actual
        .stacking_matrix_expected_widths
        .iter()
        .map(|&width| {
            core::iter::repeat_with(|| ext_chip.zero(ctx))
                .take(width)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for term in &actual.q_coeff_terms {
        let n_lift = term.n.max(0) as usize;
        assert!(
            term.commit_idx < derived_q_coeffs.len(),
            "q-coeff commit index out of bounds",
        );
        assert!(
            term.target_col_idx < derived_q_coeffs[term.commit_idx].len(),
            "q-coeff target column out of bounds",
        );
        assert!(
            term.lambda_idx < lambda_sqr_powers.len(),
            "q-coeff lambda index out of bounds",
        );
        let eq_mle = eval_eq_mle_binary_assigned(ctx, ext_chip, &u[n_lift + 1..], &term.b_bits);
        let ind = eval_in_uni_assigned(ctx, ext_chip, actual.l_skip, term.n, &u[0]);
        let (l, rs_n) = if term.n.is_negative() {
            (
                actual.l_skip.wrapping_add_signed(term.n),
                vec![ext_chip.pow_power_of_two(ctx, &r[0], term.n.unsigned_abs())],
            )
        } else {
            (actual.l_skip, r[..=n_lift].to_vec())
        };
        let eq_prism = eval_eq_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
        let mut batched = ext_chip.mul(ctx, &lambda_sqr_powers[term.lambda_idx], &eq_prism);
        if term.need_rot {
            let rot_kernel = eval_rot_kernel_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
            let lambda_rot = ext_chip.mul(ctx, &lambda, &rot_kernel);
            let rot_term = ext_chip.mul(ctx, &lambda_sqr_powers[term.lambda_idx], &lambda_rot);
            batched = ext_chip.add(ctx, &batched, &rot_term);
        }
        let batched_ind = ext_chip.mul(ctx, &batched, &ind);
        let coeff = ext_chip.mul(ctx, &eq_mle, &batched_ind);
        let updated = ext_chip.add(
            ctx,
            &derived_q_coeffs[term.commit_idx][term.target_col_idx],
            &coeff,
        );
        derived_q_coeffs[term.commit_idx][term.target_col_idx] = updated;
    }
    for (derived_row, assigned_row) in derived_q_coeffs.iter().zip(q_coeffs.iter()) {
        for (derived_coeff, assigned_coeff) in derived_row.iter().zip(assigned_row.iter()) {
            ext_chip.assert_equal(ctx, derived_coeff, assigned_coeff);
        }
    }
    let mut derived_final_sum = ext_chip.zero(ctx);
    for (coeff_row, opening_row) in derived_q_coeffs.iter().zip(stacking_openings.iter()) {
        for (coeff, opening) in coeff_row.iter().zip(opening_row.iter()) {
            let term = ext_chip.mul(ctx, coeff, opening);
            derived_final_sum = ext_chip.add(ctx, &derived_final_sum, &term);
        }
    }
    ext_chip.assert_equal(ctx, &derived_final_sum, &final_sum);

    let final_residual = assign_ext(ctx, ext_chip, actual.final_residual);
    // Equation-first check: final_residual = final_claim - final_sum.
    let derived_final_residual = ext_chip.sub(ctx, &final_claim, &final_sum);
    ext_chip.assert_equal(ctx, &derived_final_residual, &final_residual);

    let zero = ext_chip.zero(ctx);
    ext_chip.assert_equal(ctx, &s_0_residual, &zero);
    ext_chip.assert_equal(ctx, &final_residual, &zero);

    AssignedStackedReductionIntermediates {
        lambda,
        univariate_round_coeffs,
        sumcheck_round_polys,
        batch_column_openings,
        r,
        stacking_openings,
        s_0,
        s_0_sum_eval,
        s_0_residual,
        final_claim,
        final_sum,
        final_residual,
        u,
    }
}

pub fn coeffs_to_native_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> ChildEF {
    ChildEF::from_basis_coefficients_fn(|i| ChildF::from_u64(coeffs[i]))
}

#[cfg(test)]
mod tests;
