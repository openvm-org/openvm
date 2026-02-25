use std::{borrow::BorrowMut, collections::HashSet};

use itertools::izip;
use openvm_stark_backend::poly_common::{
    eval_eq_uni, eval_eq_uni_at_one, interpolate_quadratic_at_012,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    stacking::{sumcheck::air::SumcheckRoundsCols, utils::get_stacked_slice_data},
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct SumcheckRoundsTraceGenerator;

impl RowMajorChip<F> for SumcheckRoundsTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        debug_assert_eq!(proofs.len(), preflights.len());

        let width = SumcheckRoundsCols::<usize>::width();

        let traces = proofs
            .par_iter()
            .zip(preflights.par_iter())
            .enumerate()
            .map(|(proof_idx, (proof, preflight))| {
                let sumcheck_rounds = &proof.stacking_proof.sumcheck_round_polys;

                let eq_mults = {
                    let mut eq_mults = vec![0usize; vk.inner.params.n_stack];
                    for (sort_idx, (air_idx, vdata)) in
                        preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
                    {
                        if vdata.log_height > vk.inner.params.l_skip {
                            let need_rot = vk.inner.per_air[*air_idx].params.need_rot;
                            let n = vdata.log_height - vk.inner.params.l_skip;
                            eq_mults[n - 1] += proof.batch_constraint_proof.column_openings
                                [sort_idx]
                                .iter()
                                .flatten()
                                .count()
                                / if need_rot { 2 } else { 1 };
                        }
                    }
                    eq_mults
                };

                let u_mults = {
                    let mut u_mults = vec![0usize; vk.inner.params.n_stack];
                    let stacked_slices =
                        get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

                    let mut b_value_set = HashSet::<(usize, usize)>::new();
                    for slice in stacked_slices {
                        let n_lift = slice.n.max(0) as usize;
                        let b_value = slice.row_idx >> (n_lift + vk.inner.params.l_skip);
                        let total_num_bits = vk.inner.params.n_stack - n_lift;

                        for num_bits in (1..=total_num_bits).rev() {
                            let shifted_b_value = b_value >> (total_num_bits - num_bits);
                            if b_value_set.insert((shifted_b_value, num_bits)) {
                                u_mults[vk.inner.params.n_stack - num_bits] += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    u_mults
                };

                let (eq_prism_base, eq_cube_base, rot_cube_base) = {
                    let l_skip = vk.inner.params.l_skip;
                    let omega = F::two_adic_generator(l_skip);
                    let u = preflight.stacking.sumcheck_rnd[0];
                    let r = preflight.batch_constraint.sumcheck_rnd[0];

                    let eq_prism_base = eval_eq_uni(l_skip, u, r);
                    let eq_cube_base = eval_eq_uni(l_skip, u, r * omega);
                    let rot_cube_base =
                        eval_eq_uni_at_one(l_skip, u) * eval_eq_uni_at_one(l_skip, r * omega);
                    (eq_prism_base, eq_cube_base, rot_cube_base)
                };

                let num_rows = sumcheck_rounds.len();
                let proof_idx_value = F::from_usize(proof_idx);

                let mut trace = vec![F::ZERO; num_rows * width];

                let u = &preflight.stacking.sumcheck_rnd[1..];
                let batch_sumcheck_randomness = preflight.batch_constraint_sumcheck_randomness();
                let r = &batch_sumcheck_randomness[1..];

                let initial_tidx = preflight.stacking.intermediate_tidx[1];

                let mut s_eval_at_u = preflight.stacking.univariate_poly_rand_eval;

                let mut eq_cube = EF::ONE;
                let mut r_not_u_prod = EF::ONE;
                let mut rot_cube_minus_prod = EF::ZERO;

                for (round, (sumcheck_round, chunk, &u_round)) in
                    izip!(sumcheck_rounds.iter(), trace.chunks_mut(width), u.iter()).enumerate()
                {
                    let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();

                    let s_eval_at_0 = s_eval_at_u - sumcheck_round[0];
                    s_eval_at_u = interpolate_quadratic_at_012(
                        &[s_eval_at_0, sumcheck_round[0], sumcheck_round[1]],
                        u_round,
                    );

                    cols.proof_idx = proof_idx_value;
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(round == 0);
                    cols.is_last = F::from_bool(round + 1 == num_rows);

                    cols.round = F::from_usize(round + 1);
                    cols.tidx = F::from_usize(initial_tidx + (3 * D_EF * round));

                    cols.s_eval_at_0
                        .copy_from_slice(s_eval_at_0.as_basis_coefficients_slice());
                    cols.s_eval_at_1
                        .copy_from_slice(sumcheck_round[0].as_basis_coefficients_slice());
                    cols.s_eval_at_2
                        .copy_from_slice(sumcheck_round[1].as_basis_coefficients_slice());
                    cols.s_eval_at_u
                        .copy_from_slice(s_eval_at_u.as_basis_coefficients_slice());

                    cols.u_round
                        .copy_from_slice(u_round.as_basis_coefficients_slice());
                    let r_round = if round < r.len() {
                        cols.r_round = r[round].challenge;
                        cols.has_r = F::ONE;
                        EF::from_basis_coefficients_iter(r[round].challenge.into_iter()).unwrap()
                    } else {
                        EF::ZERO
                    };
                    cols.u_mult = F::from_usize(u_mults[round]);

                    cols.eq_prism_base
                        .copy_from_slice(eq_prism_base.as_basis_coefficients_slice());
                    cols.eq_cube_base
                        .copy_from_slice(eq_cube_base.as_basis_coefficients_slice());
                    cols.rot_cube_base
                        .copy_from_slice(rot_cube_base.as_basis_coefficients_slice());

                    let u_not_r = u_round * (EF::ONE - r_round);
                    let r_not_u = r_round * (EF::ONE - u_round);
                    let next_eq_term = EF::ONE - (u_not_r + r_not_u);
                    eq_cube *= next_eq_term;
                    cols.eq_cube
                        .copy_from_slice(eq_cube.as_basis_coefficients_slice());

                    rot_cube_minus_prod =
                        (rot_cube_minus_prod * next_eq_term) + u_not_r * r_not_u_prod;
                    r_not_u_prod *= r_not_u;
                    cols.r_not_u_prod
                        .copy_from_slice(r_not_u_prod.as_basis_coefficients_slice());
                    cols.rot_cube_minus_prod
                        .copy_from_slice(rot_cube_minus_prod.as_basis_coefficients_slice());

                    cols.eq_rot_mult = F::from_usize(eq_mults[round]);
                }

                (trace, num_rows)
            })
            .collect::<Vec<_>>();

        let num_valid_rows = traces.iter().map(|(_trace, num_rows)| *num_rows).sum();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };

        let mut combined_trace = Vec::with_capacity(height * width);
        for (trace, _num_rows) in traces {
            combined_trace.extend(trace);
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        combined_trace.resize(height * width, F::ZERO);
        let mut chunks = combined_trace[num_valid_rows * width..]
            .chunks_mut(width)
            .peekable();

        while let Some(chunk) = chunks.next() {
            let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
            if chunks.peek().is_none() {
                cols.is_last = F::ONE;
            }
        }

        Some(RowMajorMatrix::new(combined_trace, width))
    }
}
