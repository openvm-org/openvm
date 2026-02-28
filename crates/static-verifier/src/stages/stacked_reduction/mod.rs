use std::iter::zip;

use halo2_base::{Context, gates::range::RangeChip};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, EF as NativeEF, F as NativeF,
        default_transcript,
    },
    openvm_stark_backend::{
        FiatShamirTranscript,
        keygen::types::MultiStarkVerifyingKey,
        p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64},
        poly_common::{
            eval_eq_mle, eval_eq_prism, eval_in_uni, eval_rot_kernel_prism, horner_eval,
            interpolate_quadratic_at_012,
        },
        proof::{Proof, StackingProof, column_openings_by_rot},
        prover::stacked_pcs::StackedLayout,
        verifier::{
            batch_constraints::BatchConstraintError as NativeBatchConstraintError,
            proof_shape::ProofShapeError,
            stacked_reduction::StackedReductionError,
        },
    },
};

use crate::{
    circuit::Fr,
    gadgets::baby_bear::{BABY_BEAR_EXT_DEGREE, BabyBearArithmeticGadgets, BabyBearExtVar},
    stages::batch_constraints::{
        BatchConstraintError, eval_eq_mle_binary_assigned, eval_eq_prism_assigned,
        eval_eq_uni_at_one_assigned, eval_rot_kernel_prism_assigned, ext_from_base_const,
        ext_mul_base_const, ext_pow_power_of_two,
    },
    stages::{
        pipeline::prepare_pipeline_inputs,
        shared_math::{
            column_openings_by_rot_assigned, horner_eval_ext_poly_assigned,
            interpolate_quadratic_at_012_assigned,
        },
    },
    utils::assign_and_range_usize,
};

#[derive(Debug, PartialEq, Eq)]
pub enum StackedReductionConstraintError {
    SystemParamsMismatch,
    TraceHeightsTooLarge,
    ProofShape(ProofShapeError),
    BatchConstraint(NativeBatchConstraintError<NativeEF>),
    StackedReduction(StackedReductionError<NativeEF>),
    BatchSetup(BatchConstraintError),
}

impl From<ProofShapeError> for StackedReductionConstraintError {
    fn from(value: ProofShapeError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<NativeBatchConstraintError<NativeEF>> for StackedReductionConstraintError {
    fn from(value: NativeBatchConstraintError<NativeEF>) -> Self {
        Self::BatchConstraint(value)
    }
}

impl From<StackedReductionError<NativeEF>> for StackedReductionConstraintError {
    fn from(value: StackedReductionError<NativeEF>) -> Self {
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
    pub lambda: [u64; BABY_BEAR_EXT_DEGREE],
    pub batch_column_openings: Vec<Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>>,
    pub batch_column_openings_need_rot: Vec<Vec<bool>>,
    pub q_coeff_terms: Vec<QCoeffAccumulationTerm>,
    pub r: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub univariate_round_coeffs: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub sumcheck_round_polys: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub stacking_openings: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub q_coeffs: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub stacking_matrix_expected_widths: Vec<usize>,
    pub s_0: [u64; BABY_BEAR_EXT_DEGREE],
    pub s_0_sum_eval: [u64; BABY_BEAR_EXT_DEGREE],
    pub s_0_residual: [u64; BABY_BEAR_EXT_DEGREE],
    pub final_claim: [u64; BABY_BEAR_EXT_DEGREE],
    pub final_sum: [u64; BABY_BEAR_EXT_DEGREE],
    pub final_residual: [u64; BABY_BEAR_EXT_DEGREE],
    pub u: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
}

#[derive(Clone, Debug)]
pub struct AssignedStackedReductionIntermediates {
    pub lambda: BabyBearExtVar,
    pub univariate_round_coeffs: Vec<BabyBearExtVar>,
    pub sumcheck_round_polys: Vec<Vec<BabyBearExtVar>>,
    pub batch_column_openings: Vec<Vec<Vec<BabyBearExtVar>>>,
    pub r: Vec<BabyBearExtVar>,
    pub stacking_openings: Vec<Vec<BabyBearExtVar>>,
    pub s_0: BabyBearExtVar,
    pub s_0_sum_eval: BabyBearExtVar,
    pub s_0_residual: BabyBearExtVar,
    pub final_claim: BabyBearExtVar,
    pub final_sum: BabyBearExtVar,
    pub final_residual: BabyBearExtVar,
    pub u: Vec<BabyBearExtVar>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawStackedWitnessState {
    pub intermediates: StackedReductionIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedStackedState {
    pub s_0_residual: BabyBearExtVar,
    pub final_residual: BabyBearExtVar,
}

#[derive(Clone, Debug)]
pub struct CheckedStackedWitnessState {
    pub assigned: AssignedStackedReductionIntermediates,
    pub derived: DerivedStackedState,
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

pub(crate) fn derive_stacked_reduction_intermediates_with_inputs(
    transcript: &mut impl FiatShamirTranscript<NativeConfig>,
    proof: &StackingProof<NativeConfig>,
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
    column_openings: &Vec<Vec<Vec<NativeEF>>>,
    r: &[NativeEF],
    omega_shift_pows: &[NativeF],
) -> Result<StackedReductionIntermediates, StackedReductionError<NativeEF>> {
    let omega_order = omega_shift_pows.len();
    let omega_order_f = NativeF::from_usize(omega_order);

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
        .sum::<NativeEF>();
    let s_0_sum_eval = proof
        .univariate_round_coeffs
        .iter()
        .step_by(omega_order)
        .copied()
        .sum::<NativeEF>()
        * omega_order_f;

    let s_0_residual = s_0 - s_0_sum_eval;
    if s_0_residual != NativeEF::ZERO {
        return Err(StackedReductionError::S0Mismatch { s_0, s_0_sum_eval });
    }

    for coeff in &proof.univariate_round_coeffs {
        transcript.observe_ext(*coeff);
    }

    let mut u = vec![NativeEF::ZERO; n_stack + 1];
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

    let mut q_coeffs: Vec<Vec<NativeEF>> = proof
        .stacking_openings
        .iter()
        .map(|vec| vec![NativeEF::ZERO; vec.len()])
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
                        .map(|j| NativeF::from_bool((s.row_idx >> j) & 1 == 1))
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
        NativeEF::ZERO,
        |acc, (q_coeff_vec, q_j_vec)| {
            acc + q_coeff_vec.iter().zip(q_j_vec.iter()).fold(
                NativeEF::ZERO,
                |acc, (&q_coeff, &q_j)| {
                    transcript.observe_ext(q_j);
                    acc + (q_coeff * q_j)
                },
            )
        },
    );

    let final_residual = claim - final_sum;
    if final_residual != NativeEF::ZERO {
        return Err(StackedReductionError::FinalSumMismatch { claim, final_sum });
    }

    Ok(StackedReductionIntermediates {
        l_skip,
        lambda: ext_to_coeffs(lambda),
        batch_column_openings: column_openings
            .iter()
            .map(|per_air| {
                per_air
                    .iter()
                    .map(|part| part.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        batch_column_openings_need_rot: need_rot_per_commit.to_vec(),
        q_coeff_terms,
        r: r.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>(),
        univariate_round_coeffs: proof
            .univariate_round_coeffs
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        sumcheck_round_polys: proof
            .sumcheck_round_polys
            .iter()
            .map(|poly| poly.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        stacking_openings: proof
            .stacking_openings
            .iter()
            .map(|openings| {
                openings
                    .iter()
                    .copied()
                    .map(ext_to_coeffs)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        q_coeffs: q_coeffs
            .iter()
            .map(|coeffs| {
                coeffs
                    .iter()
                    .copied()
                    .map(ext_to_coeffs)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
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
        s_0: ext_to_coeffs(s_0),
        s_0_sum_eval: ext_to_coeffs(s_0_sum_eval),
        s_0_residual: ext_to_coeffs(s_0_residual),
        final_claim: ext_to_coeffs(claim),
        final_sum: ext_to_coeffs(final_sum),
        final_residual: ext_to_coeffs(final_residual),
        u: u.into_iter().map(ext_to_coeffs).collect(),
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
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    coeffs: [u64; BABY_BEAR_EXT_DEGREE],
) -> BabyBearExtVar {
    baby_bear.load_ext_witness(ctx, range, coeffs)
}

fn eval_in_uni_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    l_skip: usize,
    n: isize,
    z: &BabyBearExtVar,
) -> BabyBearExtVar {
    if n.is_negative() {
        let z_pow = ext_pow_power_of_two(ctx, range, baby_bear, z, l_skip.wrapping_add_signed(n));
        eval_eq_uni_at_one_assigned(ctx, range, baby_bear, n.unsigned_abs(), &z_pow)
    } else {
        ext_from_base_const(ctx, range, baby_bear, 1)
    }
}

pub(crate) fn constrain_stacked_reduction_intermediates(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &StackedReductionIntermediates,
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

    let baby_bear = BabyBearArithmeticGadgets;

    let lambda = assign_ext(ctx, range, &baby_bear, actual.lambda);
    let u = actual
        .u
        .iter()
        .map(|&actual_u| assign_ext(ctx, range, &baby_bear, actual_u))
        .collect::<Vec<_>>();
    let univariate_round_coeffs = actual
        .univariate_round_coeffs
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = actual
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let stacking_openings = actual
        .stacking_openings
        .iter()
        .map(|openings| {
            openings
                .iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let batch_column_openings = actual
        .batch_column_openings
        .iter()
        .map(|per_air| {
            per_air
                .iter()
                .map(|part| {
                    part.iter()
                        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let r = actual
        .r
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let q_coeffs = actual
        .q_coeffs
        .iter()
        .map(|coeffs| {
            coeffs
                .iter()
                .map(|&value| assign_ext(ctx, range, &baby_bear, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for ((opening_row, q_coeff_row), &expected_width) in actual
        .stacking_openings
        .iter()
        .zip(actual.q_coeffs.iter())
        .zip(actual.stacking_matrix_expected_widths.iter())
    {
        let opening_width = assign_and_range_usize(ctx, range, opening_row.len());
        let q_coeff_width = assign_and_range_usize(ctx, range, q_coeff_row.len());
        let expected_width_cell = assign_and_range_usize(ctx, range, expected_width);
        ctx.constrain_equal(&opening_width, &expected_width_cell);
        ctx.constrain_equal(&q_coeff_width, &expected_width_cell);
    }
    let s_0 = assign_ext(ctx, range, &baby_bear, actual.s_0);

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
            ctx, range, &baby_bear, &parts[0], need_rot,
        ));
    }
    let mut commit_idx = 1usize;
    for parts in &batch_column_openings {
        for cols in parts.iter().skip(1) {
            let need_rot = *commit_need_rot[commit_idx]
                .first()
                .expect("non-common commit need_rot metadata must be singleton");
            t_claims.extend(column_openings_by_rot_assigned(
                ctx, range, &baby_bear, cols, need_rot,
            ));
            commit_idx += 1;
        }
    }
    assert_eq!(
        commit_idx,
        commit_need_rot.len(),
        "all non-common commitments must be consumed when deriving t-claims",
    );

    let lambda_sqr = baby_bear.ext_mul(ctx, range, &lambda, &lambda);
    let mut lambda_sqr_powers = Vec::with_capacity(t_claims.len());
    let mut cur_lambda_sqr = ext_from_base_const(ctx, range, &baby_bear, 1);
    for _ in 0..t_claims.len() {
        lambda_sqr_powers.push(cur_lambda_sqr.clone());
        cur_lambda_sqr = baby_bear.ext_mul(ctx, range, &cur_lambda_sqr, &lambda_sqr);
    }
    let mut derived_s_0 = baby_bear.ext_zero(ctx, range);
    for ((claim, claim_rot), lambda_pow) in t_claims.iter().zip(lambda_sqr_powers.iter()) {
        let claim_rot_lambda = baby_bear.ext_mul(ctx, range, claim_rot, &lambda);
        let batched_claim = baby_bear.ext_add(ctx, range, claim, &claim_rot_lambda);
        let term = baby_bear.ext_mul(ctx, range, &batched_claim, lambda_pow);
        derived_s_0 = baby_bear.ext_add(ctx, range, &derived_s_0, &term);
    }
    baby_bear.assert_ext_equal(ctx, &derived_s_0, &s_0);

    let s_0_sum_eval = assign_ext(ctx, range, &baby_bear, actual.s_0_sum_eval);
    let stride = 1usize << actual.l_skip;
    let mut derived_s_0_sum_eval = baby_bear.ext_zero(ctx, range);
    for coeff in univariate_round_coeffs.iter().step_by(stride) {
        derived_s_0_sum_eval = baby_bear.ext_add(ctx, range, &derived_s_0_sum_eval, coeff);
    }
    let derived_s_0_sum_eval =
        ext_mul_base_const(ctx, range, &baby_bear, &derived_s_0_sum_eval, stride as u64);
    baby_bear.assert_ext_equal(ctx, &derived_s_0_sum_eval, &s_0_sum_eval);

    let s_0_residual = assign_ext(ctx, range, &baby_bear, actual.s_0_residual);
    // Equation-first check: s_0_residual = s_0 - s_0_sum_eval.
    let derived_s_0_residual = baby_bear.ext_sub(ctx, range, &s_0, &s_0_sum_eval);
    baby_bear.assert_ext_equal(ctx, &derived_s_0_residual, &s_0_residual);

    assert!(
        !univariate_round_coeffs.is_empty(),
        "stacked univariate polynomial must have at least one coefficient",
    );
    assert_eq!(
        sumcheck_round_polys.len() + 1,
        u.len(),
        "stacked sumcheck rounds must align with sampled u challenges",
    );
    let final_claim = assign_ext(ctx, range, &baby_bear, actual.final_claim);
    // Recompute native claim chain from constrained univariate/sumcheck payloads and u challenges.
    let mut derived_claim =
        horner_eval_ext_poly_assigned(ctx, range, &baby_bear, &univariate_round_coeffs, &u[0]);
    for (round_idx, round_poly) in sumcheck_round_polys.iter().enumerate() {
        assert_eq!(
            round_poly.len(),
            2,
            "stacked sumcheck rounds must expose [s(1), s(2)]",
        );
        let s_j_1 = &round_poly[0];
        let s_j_2 = &round_poly[1];
        let s_j_0 = baby_bear.ext_sub(ctx, range, &derived_claim, s_j_1);
        derived_claim = interpolate_quadratic_at_012_assigned(
            ctx,
            range,
            &baby_bear,
            [&s_j_0, s_j_1, s_j_2],
            &u[round_idx + 1],
        );
    }
    baby_bear.assert_ext_equal(ctx, &derived_claim, &final_claim);

    let final_sum = assign_ext(ctx, range, &baby_bear, actual.final_sum);
    let mut derived_q_coeffs = actual
        .stacking_matrix_expected_widths
        .iter()
        .map(|&width| {
            core::iter::repeat_with(|| baby_bear.ext_zero(ctx, range))
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
        let eq_mle =
            eval_eq_mle_binary_assigned(ctx, range, &baby_bear, &u[n_lift + 1..], &term.b_bits);
        let ind = eval_in_uni_assigned(ctx, range, &baby_bear, actual.l_skip, term.n, &u[0]);
        let (l, rs_n) = if term.n.is_negative() {
            (
                actual.l_skip.wrapping_add_signed(term.n),
                vec![ext_pow_power_of_two(
                    ctx,
                    range,
                    &baby_bear,
                    &r[0],
                    term.n.unsigned_abs(),
                )],
            )
        } else {
            (actual.l_skip, r[..=n_lift].to_vec())
        };
        let eq_prism = eval_eq_prism_assigned(ctx, range, &baby_bear, l, &u[..=n_lift], &rs_n);
        let mut batched =
            baby_bear.ext_mul(ctx, range, &lambda_sqr_powers[term.lambda_idx], &eq_prism);
        if term.need_rot {
            let rot_kernel =
                eval_rot_kernel_prism_assigned(ctx, range, &baby_bear, l, &u[..=n_lift], &rs_n);
            let lambda_rot = baby_bear.ext_mul(ctx, range, &lambda, &rot_kernel);
            let rot_term =
                baby_bear.ext_mul(ctx, range, &lambda_sqr_powers[term.lambda_idx], &lambda_rot);
            batched = baby_bear.ext_add(ctx, range, &batched, &rot_term);
        }
        let batched_ind = baby_bear.ext_mul(ctx, range, &batched, &ind);
        let coeff = baby_bear.ext_mul(ctx, range, &eq_mle, &batched_ind);
        let updated = baby_bear.ext_add(
            ctx,
            range,
            &derived_q_coeffs[term.commit_idx][term.target_col_idx],
            &coeff,
        );
        derived_q_coeffs[term.commit_idx][term.target_col_idx] = updated;
    }
    for (derived_row, assigned_row) in derived_q_coeffs.iter().zip(q_coeffs.iter()) {
        for (derived_coeff, assigned_coeff) in derived_row.iter().zip(assigned_row.iter()) {
            baby_bear.assert_ext_equal(ctx, derived_coeff, assigned_coeff);
        }
    }
    let mut derived_final_sum = baby_bear.ext_zero(ctx, range);
    for (coeff_row, opening_row) in derived_q_coeffs.iter().zip(stacking_openings.iter()) {
        for (coeff, opening) in coeff_row.iter().zip(opening_row.iter()) {
            let term = baby_bear.ext_mul(ctx, range, coeff, opening);
            derived_final_sum = baby_bear.ext_add(ctx, range, &derived_final_sum, &term);
        }
    }
    baby_bear.assert_ext_equal(ctx, &derived_final_sum, &final_sum);

    let final_residual = assign_ext(ctx, range, &baby_bear, actual.final_residual);
    // Equation-first check: final_residual = final_claim - final_sum.
    let derived_final_residual = baby_bear.ext_sub(ctx, range, &final_claim, &final_sum);
    baby_bear.assert_ext_equal(ctx, &derived_final_residual, &final_residual);

    let zero = baby_bear.ext_zero(ctx, range);
    baby_bear.assert_ext_equal(ctx, &s_0_residual, &zero);
    baby_bear.assert_ext_equal(ctx, &final_residual, &zero);

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

/// Standalone stacked-reduction derive+constrain wrapper is internal; external callers must use
/// transcript-owned stage composition (`stages::full_pipeline`) as the acceptance boundary.
pub(crate) fn derive_and_constrain_stacked_reduction(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedStackedReductionIntermediates, StackedReductionConstraintError> {
    let raw = derive_raw_stacked_witness_state(config, mvk, proof)?;
    Ok(constrain_checked_stacked_witness_state(ctx, range, &raw).assigned)
}

pub(crate) fn derive_raw_stacked_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawStackedWitnessState, StackedReductionConstraintError> {
    Ok(RawStackedWitnessState {
        intermediates: derive_stacked_reduction_intermediates(config, mvk, proof)?,
    })
}

pub(crate) fn constrain_checked_stacked_witness_state(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawStackedWitnessState,
) -> CheckedStackedWitnessState {
    let assigned = constrain_stacked_reduction_intermediates(ctx, range, &raw.intermediates);
    let derived = DerivedStackedState {
        s_0_residual: assigned.s_0_residual.clone(),
        final_residual: assigned.final_residual.clone(),
    };
    CheckedStackedWitnessState { assigned, derived }
}

pub fn coeffs_to_native_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> NativeEF {
    coeffs_to_ext(coeffs)
}

#[cfg(test)]
mod tests;
