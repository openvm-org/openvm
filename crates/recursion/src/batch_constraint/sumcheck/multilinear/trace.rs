use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::MultilinearSumcheckCols;
use crate::system::Preflight;

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = MultilinearSumcheckCols::<F>::width();
    let s_deg = vk.inner.max_constraint_degree + 1;
    let rows_per_round = s_deg + 1;

    let rows_per_proof: Vec<usize> = proofs
        .iter()
        .map(|proof| proof.batch_constraint_proof.sumcheck_round_polys.len() * rows_per_round)
        .collect();
    let total_height: usize = rows_per_proof.iter().sum();
    let padded_height = total_height.max(1).next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    let mut factorials = vec![F::ONE; s_deg + 1];
    for i in 1..=s_deg {
        factorials[i] = factorials[i - 1] * F::from_canonical_usize(i);
    }
    let invfact = batch_multiplicative_inverse(&factorials);
    let denom_inv: Vec<F> = (0..=s_deg)
        .map(|i| invfact[i] * invfact[s_deg - i])
        .collect();

    let (data_slice, _) = trace.split_at_mut(total_height * width);
    let mut trace_slices: Vec<&mut [F]> = Vec::with_capacity(rows_per_proof.len());
    let mut remaining = data_slice;

    for &num_rows in &rows_per_proof {
        let (chunk, rest) = remaining.split_at_mut(num_rows * width);
        trace_slices.push(chunk);
        remaining = rest;
    }
    debug_assert_eq!(trace_slices.len(), proofs.len());

    trace_slices
        .par_iter_mut()
        .zip(proofs.par_iter().zip(preflights.par_iter()))
        .enumerate()
        .for_each(|(pidx, (rows, (proof, preflight)))| {
            let polys = &proof.batch_constraint_proof.sumcheck_round_polys;
            let n_rounds = polys.len();
            if rows.is_empty() {
                debug_assert_eq!(n_rounds, 0);
                return;
            }

            debug_assert_eq!(rows.len(), n_rounds * rows_per_round * width);
            let tidx_before_multilinear = preflight.batch_constraint.tidx_before_multilinear;
            let sumcheck_rnd = &preflight.batch_constraint.sumcheck_rnd;

            debug_assert!(sumcheck_rnd.len() > n_rounds);

            let r0 = sumcheck_rnd[0];
            let mut pow = EF::ONE;
            let mut cur_sum_ext = EF::ZERO;
            for coeff in &proof.batch_constraint_proof.univariate_round_coeffs {
                cur_sum_ext += pow * *coeff;
                pow *= r0;
            }

            // Populate per-row metadata that does not depend on prior rows
            rows.par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(row_idx, chunk)| {
                    let round_idx = row_idx / rows_per_round;
                    let eval_idx = row_idx % rows_per_round;
                    let tidx_round = tidx_before_multilinear + round_idx * rows_per_round * D_EF;

                    let cols: &mut MultilinearSumcheckCols<F> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_canonical_usize(pidx);
                    cols.round_idx = F::from_canonical_usize(round_idx);
                    cols.is_round_start = F::from_bool(round_idx == 0 && eval_idx == 0);
                    cols.is_first_eval = F::from_bool(eval_idx == 0);
                    cols.nested_for_loop_aux_cols.is_transition[0] =
                        F::from_bool(round_idx + 1 != n_rounds || eval_idx != s_deg);
                    cols.eval_idx = F::from_canonical_usize(eval_idx);
                    cols.tidx = F::from_canonical_usize(tidx_round + eval_idx * D_EF);
                });

            // Serial portion: compute dependent values round-by-round
            // TODO(ayush): see if some of this can be put into records to make it more parallel
            let mut row_iter = rows.chunks_exact_mut(width);
            for (round_idx, poly) in polys.iter().enumerate() {
                debug_assert_eq!(poly.len(), s_deg);

                let r = sumcheck_rnd[round_idx + 1];
                let s1 = poly[0];
                let s0 = cur_sum_ext - s1;
                let mut evals = Vec::with_capacity(rows_per_round);
                evals.push(s0);
                evals.extend(poly.iter().copied());

                let mut prefix_products = vec![EF::ONE; rows_per_round];
                for i in 0..s_deg {
                    prefix_products[i + 1] = prefix_products[i] * (r - EF::from_canonical_usize(i));
                }

                let mut suffix_products = vec![EF::ONE; rows_per_round];
                for i in (0..s_deg).rev() {
                    suffix_products[i] =
                        suffix_products[i + 1] * (EF::from_canonical_usize(s_deg - i) - r);
                }

                let mut round_sum = EF::ZERO;
                for eval_idx in 0..=s_deg {
                    let eval_ext = evals[eval_idx];
                    let prefix = prefix_products[eval_idx];
                    let suffix = suffix_products[eval_idx];
                    let denom = denom_inv[eval_idx];

                    round_sum += eval_ext * prefix * suffix * denom;

                    let chunk = row_iter.next().unwrap();
                    let cols: &mut MultilinearSumcheckCols<F> = chunk.borrow_mut();
                    cols.eval.copy_from_slice(eval_ext.as_base_slice());
                    cols.prefix_product.copy_from_slice(prefix.as_base_slice());
                    cols.suffix_product.copy_from_slice(suffix.as_base_slice());
                    cols.r.copy_from_slice(r.as_base_slice());
                    cols.denom_inv = denom;
                    cols.cur_sum.copy_from_slice(round_sum.as_base_slice());
                }

                cur_sum_ext = round_sum;
            }
            debug_assert!(row_iter.next().is_none());
        });

    RowMajorMatrix::new(trace, width)
}
