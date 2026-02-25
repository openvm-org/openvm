use std::borrow::BorrowMut;

use itertools::Itertools;
use openvm_stark_backend::poly_common::{eval_eq_uni, eval_rot_kernel_prism, Squarable};
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    stacking::eq_base::air::EqBaseCols,
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct EqBaseTraceGenerator;

impl RowMajorChip<F> for EqBaseTraceGenerator {
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

        let width = EqBaseCols::<usize>::width();

        let num_rows_per_proof = vk.inner.params.l_skip + 1;
        let traces = proofs
            .par_iter()
            .zip(preflights.par_iter())
            .enumerate()
            .map(|(proof_idx, (proof, preflight))| {
                let mut mults = vec![0usize; vk.inner.params.l_skip + 1];
                for (sort_idx, (air_idx, vdata)) in
                    preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
                {
                    let need_rot = vk.inner.per_air[*air_idx].params.need_rot;
                    if vdata.log_height <= vk.inner.params.l_skip {
                        let neg_n = vk.inner.params.l_skip - vdata.log_height;
                        mults[neg_n] += proof.batch_constraint_proof.column_openings[sort_idx]
                            .iter()
                            .flatten()
                            .count()
                            / if need_rot { 2 } else { 1 };
                    }
                }

                let proof_idx_value = F::from_usize(proof_idx);

                let mut trace = vec![F::ZERO; num_rows_per_proof * width];

                let omega = F::two_adic_generator(vk.inner.params.l_skip);
                let mut u = preflight.stacking.sumcheck_rnd[0];
                let mut r = preflight.batch_constraint.sumcheck_rnd[0];
                let mut r_omega = r * omega;

                let mut prod_u_r = u * (u + r);
                let mut prod_u_r_omega = u * (u + r_omega);
                let mut prod_u_1 = u + F::ONE;
                let mut prod_r_omega_1 = r_omega + F::ONE;

                let mut in_prod = EF::ONE;

                let u_pows = u
                    .exp_powers_of_2()
                    .take(vk.inner.params.l_skip + 1)
                    .collect_vec();

                for (row_idx, chunk) in trace.chunks_mut(width).take(num_rows_per_proof).enumerate()
                {
                    let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
                    let is_last = row_idx + 1 == num_rows_per_proof;

                    cols.proof_idx = proof_idx_value;
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(row_idx == 0);
                    cols.is_last = F::from_bool(is_last);

                    cols.row_idx = F::from_usize(row_idx);

                    cols.u_pow.copy_from_slice(u.as_basis_coefficients_slice());
                    cols.r_pow.copy_from_slice(r.as_basis_coefficients_slice());
                    cols.r_omega_pow
                        .copy_from_slice(r_omega.as_basis_coefficients_slice());

                    cols.prod_u_r
                        .copy_from_slice(prod_u_r.as_basis_coefficients_slice());
                    cols.prod_u_r_omega
                        .copy_from_slice(prod_u_r_omega.as_basis_coefficients_slice());
                    cols.prod_u_1
                        .copy_from_slice(prod_u_1.as_basis_coefficients_slice());
                    cols.prod_r_omega_1
                        .copy_from_slice(prod_r_omega_1.as_basis_coefficients_slice());

                    if is_last {
                        cols.mult = F::from_usize(mults[0]);
                    }

                    let l_skip = vk.inner.params.l_skip - row_idx;
                    let u_pow_rev = u_pows[l_skip];

                    if row_idx != 0 {
                        in_prod *= u_pow_rev + F::ONE;
                        cols.eq_neg.copy_from_slice(
                            (eval_eq_uni(l_skip, preflight.stacking.sumcheck_rnd[0], r)
                                * F::from_usize(1 << l_skip))
                            .as_basis_coefficients_slice(),
                        );
                        cols.k_rot_neg.copy_from_slice(
                            (eval_rot_kernel_prism(
                                l_skip,
                                &[preflight.stacking.sumcheck_rnd[0]],
                                &[r],
                            ) * F::from_usize(1 << l_skip))
                            .as_basis_coefficients_slice(),
                        );
                        cols.mult_neg = F::from_usize(mults[row_idx]);
                    }

                    cols.u_pow_rev
                        .copy_from_slice(u_pow_rev.as_basis_coefficients_slice());
                    cols.in_prod
                        .copy_from_slice(in_prod.as_basis_coefficients_slice());

                    u *= u;
                    r *= r;
                    r_omega *= r_omega;

                    prod_u_r *= u + r;
                    prod_u_r_omega *= u + r_omega;
                    prod_u_1 *= u + F::ONE;
                    prod_r_omega_1 *= r_omega + F::ONE;
                }

                (trace, num_rows_per_proof)
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
            let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
            if chunks.peek().is_none() {
                cols.is_last = F::ONE;
            }
        }

        Some(RowMajorMatrix::new(combined_trace, width))
    }
}
