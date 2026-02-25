use std::borrow::BorrowMut;

use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, poly_common::eval_eq_uni_at_one,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    batch_constraint::{eq_airs::eq_neg::air::EqNegCols, SelectorCount},
    system::Preflight,
    tracegen::RowMajorChip,
    utils::MultiVecWithBounds,
};

pub struct EqNegTraceGenerator;

impl RowMajorChip<F> for EqNegTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a [&'a Preflight],
        &'a MultiVecWithBounds<SelectorCount, 1>,
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, preflights, selector_counts) = ctx;
        let l_skip = vk.inner.params.l_skip;
        let width = EqNegCols::<usize>::width();

        if l_skip == 0 {
            let ret = if let Some(height) = required_height {
                // We essentially fill the trace with dummy rows as we do below, but
                // instead of proof_idx being preflights.len() it is 0
                Some(RowMajorMatrix::new(vec![F::ZERO; height * width], width))
            } else {
                Some(RowMajorMatrix::new(vec![], width))
            };
            return ret;
        }

        let total_valid = preflights.len() * (l_skip * (l_skip + 3)) / 2;

        let padded_height = if let Some(height) = required_height {
            if height < total_valid {
                return None;
            }
            height
        } else {
            total_valid.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padded_height * width];
        let mut chunks = trace.chunks_exact_mut(width);

        for (proof_idx, preflight) in preflights.iter().enumerate() {
            let initial_omega = F::two_adic_generator(vk.inner.params.l_skip);
            let initial_u = preflight.stacking.sumcheck_rnd[0];
            let mut initial_r = preflight.batch_constraint.sumcheck_rnd[0];
            let mut initial_r_omega = initial_r * initial_omega;

            for neg_hypercube in 0..l_skip {
                let mut u = initial_u;
                let mut r = initial_r;
                let mut r_omega = initial_r_omega;

                let mut prod_u_r = u * (u + r);
                let mut prod_u_r_omega = u * (u + r_omega);
                let mut prod_1_r = r + F::ONE;
                let mut prod_1_r_omega = r_omega + F::ONE;

                let one_half = F::ONE.halve();
                let mut one_half_pow = one_half;

                for row_idx in 0..=l_skip - neg_hypercube {
                    let chunk = chunks.next().unwrap();
                    let cols: &mut EqNegCols<F> = chunk.borrow_mut();
                    let is_first_hypercube = row_idx == 0;
                    let is_last_hypercube = row_idx == l_skip - neg_hypercube;

                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(neg_hypercube == 0 && is_first_hypercube);
                    cols.is_last = F::from_bool(neg_hypercube + 1 == l_skip && is_last_hypercube);

                    cols.neg_hypercube = F::from_usize(neg_hypercube);
                    cols.neg_hypercube_nz_inv =
                        cols.neg_hypercube.try_inverse().unwrap_or_default();
                    cols.row_index = F::from_usize(row_idx);
                    cols.is_first_hypercube = F::from_bool(is_first_hypercube);
                    cols.is_last_hypercube = F::from_bool(is_last_hypercube);

                    cols.u_pow.copy_from_slice(u.as_basis_coefficients_slice());
                    cols.r_pow.copy_from_slice(r.as_basis_coefficients_slice());
                    cols.r_omega_pow
                        .copy_from_slice(r_omega.as_basis_coefficients_slice());
                    u *= u;
                    r *= r;
                    r_omega *= r_omega;

                    cols.prod_u_r
                        .copy_from_slice(prod_u_r.as_basis_coefficients_slice());
                    cols.prod_u_r_omega
                        .copy_from_slice(prod_u_r_omega.as_basis_coefficients_slice());
                    prod_u_r *= u + r;
                    prod_u_r_omega *= u + r_omega;

                    cols.prod_1_r
                        .copy_from_slice(prod_1_r.as_basis_coefficients_slice());
                    cols.prod_1_r_omega
                        .copy_from_slice(prod_1_r_omega.as_basis_coefficients_slice());
                    cols.one_half_pow = one_half_pow;
                    debug_assert_eq!(
                        prod_1_r * one_half_pow,
                        eval_eq_uni_at_one(row_idx + 1, initial_r)
                    );
                    debug_assert_eq!(
                        prod_1_r_omega * one_half_pow,
                        eval_eq_uni_at_one(row_idx + 1, initial_r_omega)
                    );
                    prod_1_r *= r + F::ONE;
                    prod_1_r_omega *= r_omega + F::ONE;
                    one_half_pow *= one_half;

                    // We only use the count for the last row of `hypercube` or the very first row.
                    // We use the very first row to hard-code the lookup for log_height = 0 since
                    // it's disjoint from the other condition.
                    if neg_hypercube == 0 && row_idx == 0 {
                        let counts = selector_counts[[proof_idx]][0];
                        cols.sel_first_count = F::from_usize(counts.first);
                        cols.sel_last_trans_count = F::from_usize(counts.last + counts.transition);
                    } else if row_idx == l_skip - neg_hypercube {
                        let mut counts = selector_counts[[proof_idx]][row_idx];
                        // On `neg_hypercube`, we collect counts for all heights >= l_skip.
                        if neg_hypercube == 0 {
                            for count in &selector_counts[[proof_idx]][l_skip - neg_hypercube + 1..]
                            {
                                counts.first += count.first;
                                counts.last += count.last;
                                counts.transition += count.transition;
                            }
                        }
                        cols.sel_first_count = F::from_usize(counts.first);
                        cols.sel_last_trans_count = F::from_usize(counts.last + counts.transition);
                    }
                }

                initial_r *= initial_r;
                initial_r_omega *= initial_r_omega;
            }
        }

        for chunk in chunks {
            let cols: &mut EqNegCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_usize(preflights.len());
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
