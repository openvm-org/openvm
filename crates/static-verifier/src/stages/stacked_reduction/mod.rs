use core::cmp::Reverse;

use halo2_base::Context;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::{
        keygen::types::MultiStarkVerifyingKey0, p3_field::PrimeCharacteristicRing, proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};

use crate::{
    field::baby_bear::{BabyBearExtChip, BabyBearExtWire},
    stages::{
        batch_constraints::{
            eval_eq_mle_binary_assigned, eval_eq_prism_assigned, eval_eq_uni_at_one_assigned,
            eval_rot_kernel_prism_assigned,
        },
        shared_math::{
            column_openings_by_rot_assigned, horner_eval_ext_poly_assigned,
            interpolate_quadratic_at_012_assigned,
        },
    },
    transcript::TranscriptGadget,
    Fr, RootF,
};

/// Stacked PCS layouts for the trace commitments present in `proof`, derived from the VK widths
/// and per-air trace heights. Caller must ensure `proof` matches the static circuit shape.
pub(crate) fn stacked_reduction_layouts(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    proof: &Proof<RootConfig>,
) -> Vec<StackedLayout> {
    let l_skip = mvk0.params.l_skip;
    let mut per_trace = mvk0
        .per_air
        .iter()
        .zip(&proof.trace_vdata)
        .enumerate()
        .filter_map(|(air_idx, (vk, vdata))| vdata.as_ref().map(|vdata| (air_idx, vk, vdata)))
        .collect::<Vec<_>>();
    per_trace.sort_by_key(|(_, _, vdata)| Reverse(vdata.log_height));

    let common_main_layout = StackedLayout::new(
        l_skip,
        mvk0.params.n_stack + l_skip,
        per_trace
            .iter()
            .map(|(_, vk, vdata)| (vk.params.width.common_main, vdata.log_height))
            .collect::<Vec<_>>(),
    )
    .expect("stacked layout for common main");
    let other_layouts = per_trace
        .iter()
        .flat_map(|(_, vk, vdata)| {
            vk.params
                .width
                .preprocessed
                .iter()
                .chain(&vk.params.width.cached_mains)
                .copied()
                .map(|width| (width, vdata.log_height))
                .collect::<Vec<_>>()
        })
        .map(|sorted| {
            StackedLayout::new(l_skip, mvk0.params.n_stack + l_skip, vec![sorted])
                .expect("stacked layout for auxiliary column")
        })
        .collect::<Vec<_>>();
    core::iter::once(common_main_layout)
        .chain(other_layouts)
        .collect::<Vec<_>>()
}

#[derive(Clone, Debug)]
pub struct StackedReductionWire {
    pub stacking_openings: Vec<Vec<BabyBearExtWire>>,
    pub u: Vec<BabyBearExtWire>,
}

#[derive(Clone, Debug)]
pub(crate) struct StackingProofWire {
    pub univariate_round_coeffs: Vec<BabyBearExtWire>,
    pub sumcheck_round_polys: Vec<Vec<BabyBearExtWire>>,
    pub stacking_openings: Vec<Vec<BabyBearExtWire>>,
}

pub(crate) fn load_stacking_proof_wire(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    stacking_proof: &openvm_stark_sdk::openvm_stark_backend::proof::StackingProof<RootConfig>,
) -> StackingProofWire {
    let univariate_round_coeffs = stacking_proof
        .univariate_round_coeffs
        .iter()
        .map(|&value| ext_chip.load_witness(ctx, value))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = stacking_proof
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&value| ext_chip.load_witness(ctx, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let stacking_openings = stacking_proof
        .stacking_openings
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| ext_chip.load_witness(ctx, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    StackingProofWire {
        univariate_round_coeffs,
        sumcheck_round_polys,
        stacking_openings,
    }
}

fn eval_in_uni_assigned(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    l_skip: usize,
    n: isize,
    z: BabyBearExtWire,
) -> BabyBearExtWire {
    if n.is_negative() {
        let z_pow = ext_chip.pow_power_of_two(ctx, z, l_skip.wrapping_add_signed(n));
        eval_eq_uni_at_one_assigned(ctx, ext_chip, n.unsigned_abs(), &z_pow)
    } else {
        ext_chip.from_base_const(ctx, RootF::from_u64(1))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn constrain_stacked_reduction(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip,
    transcript: &mut TranscriptGadget,
    stacking_wire: &StackingProofWire,
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
    batch_column_openings: &[Vec<Vec<BabyBearExtWire>>],
    r: &[BabyBearExtWire],
) -> StackedReductionWire {
    let omega_order = 1usize << l_skip;
    let one = ext_chip.from_base_const(ctx, RootF::ONE);

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

    let lambda = transcript.sample_ext(ctx, ext_chip.base());
    let lambda_sqr = ext_chip.mul(ctx, lambda, lambda);
    let mut lambda_sqr_powers = Vec::with_capacity(t_claims.len());
    let mut cur_lambda_sqr = one;
    for _ in 0..t_claims.len() {
        lambda_sqr_powers.push(cur_lambda_sqr);
        cur_lambda_sqr = ext_chip.mul(ctx, cur_lambda_sqr, lambda_sqr);
    }

    let mut s_0 = ext_chip.zero(ctx);
    for ((claim, claim_rot), lambda_pow) in t_claims.iter().zip(lambda_sqr_powers.iter()) {
        let claim_rot_lambda = ext_chip.mul(ctx, *claim_rot, lambda);
        let batched_claim = ext_chip.add(ctx, *claim, claim_rot_lambda);
        let term = ext_chip.mul(ctx, batched_claim, *lambda_pow);
        s_0 = ext_chip.add(ctx, s_0, term);
    }

    let univariate_round_coeffs = &stacking_wire.univariate_round_coeffs;
    let mut s_0_sum_eval = ext_chip.zero(ctx);
    for coeff in univariate_round_coeffs.iter().step_by(omega_order) {
        s_0_sum_eval = ext_chip.add(ctx, s_0_sum_eval, *coeff);
    }
    let s_0_sum_eval =
        ext_chip.mul_base_const(ctx, s_0_sum_eval, RootF::from_u64(omega_order as u64));
    let s_0_residual = ext_chip.sub(ctx, s_0, s_0_sum_eval);
    let zero = ext_chip.zero(ctx);
    ext_chip.assert_equal(ctx, s_0_residual, zero);

    for coeff in univariate_round_coeffs {
        transcript.observe_ext(ctx, ext_chip.base(), coeff);
    }

    let mut u = Vec::with_capacity(n_stack + 1);
    u.push(transcript.sample_ext(ctx, ext_chip.base()));

    let sumcheck_round_polys = &stacking_wire.sumcheck_round_polys;

    let mut final_claim =
        horner_eval_ext_poly_assigned(ctx, ext_chip, univariate_round_coeffs, &u[0]);
    for round_poly in sumcheck_round_polys {
        let s_j_1 = round_poly[0];
        let s_j_2 = round_poly[1];
        transcript.observe_ext(ctx, ext_chip.base(), &s_j_1);
        transcript.observe_ext(ctx, ext_chip.base(), &s_j_2);
        let u_j = transcript.sample_ext(ctx, ext_chip.base());
        let s_j_0 = ext_chip.sub(ctx, final_claim, s_j_1);
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
            let ind = eval_in_uni_assigned(ctx, ext_chip, l_skip, n, u[0]);
            let (l, rs_n) = if n.is_negative() {
                (
                    l_skip.wrapping_add_signed(n),
                    vec![ext_chip.pow_power_of_two(ctx, r[0], n.unsigned_abs())],
                )
            } else {
                (l_skip, r[..=n_lift].to_vec())
            };
            let eq_prism = eval_eq_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
            let mut batched = ext_chip.mul(ctx, lambda_sqr_powers[lambda_idx], eq_prism);
            if need_rot {
                let rot_kernel =
                    eval_rot_kernel_prism_assigned(ctx, ext_chip, l, &u[..=n_lift], &rs_n);
                let lambda_rot = ext_chip.mul(ctx, lambda, rot_kernel);
                let rot_term = ext_chip.mul(ctx, lambda_sqr_powers[lambda_idx], lambda_rot);
                batched = ext_chip.add(ctx, batched, rot_term);
            }
            let batched_ind = ext_chip.mul(ctx, batched, ind);
            let coeff = ext_chip.mul(ctx, eq_mle, batched_ind);
            let updated = ext_chip.add(ctx, derived_q_coeffs[commit_idx][s.col_idx], coeff);
            derived_q_coeffs[commit_idx][s.col_idx] = updated;
        }
    }

    let stacking_openings = &stacking_wire.stacking_openings;
    let mut final_sum = ext_chip.zero(ctx);
    for (coeff_row, opening_row) in derived_q_coeffs.iter().zip(stacking_openings.iter()) {
        for (coeff, opening) in coeff_row.iter().zip(opening_row.iter()) {
            transcript.observe_ext(ctx, ext_chip.base(), opening);
            let term = ext_chip.mul(ctx, *coeff, *opening);
            final_sum = ext_chip.add(ctx, final_sum, term);
        }
    }

    let final_residual = ext_chip.sub(ctx, final_claim, final_sum);
    ext_chip.assert_equal(ctx, final_residual, zero);

    StackedReductionWire {
        stacking_openings: stacking_openings.clone(),
        u,
    }
}
