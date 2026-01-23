use std::borrow::BorrowMut;

use openvm_circuit_primitives::{TraceSubRowGenerator, is_equal::IsEqSubAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::UnivariateSumcheckCols;
use crate::system::Preflight;

#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[&Proof],
    preflights: &[&Preflight],
) -> RowMajorMatrix<F> {
    let width = UnivariateSumcheckCols::<F>::width();
    let rows_per_proof: Vec<usize> = proofs
        .iter()
        .map(|proof| {
            let res = proof.batch_constraint_proof.univariate_round_coeffs.len();
            debug_assert!(res > 0);
            res
        })
        .collect();
    let total_height: usize = rows_per_proof.iter().sum();

    let padded_height = total_height.max(1).next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

    let l_skip = vk.inner.params.l_skip;
    let domain_size = 1usize << l_skip;
    let domain_size_ext = EF::from_canonical_usize(domain_size);

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
            let coeffs = &proof.batch_constraint_proof.univariate_round_coeffs;
            let height = coeffs.len();
            if rows.is_empty() {
                debug_assert_eq!(height, 0);
                return;
            }
            debug_assert_eq!(rows.len(), height * width);

            let msgs = preflight
                .batch_constraint_sumcheck_randomness()
                .into_iter()
                .filter(|x| x.idx == F::ZERO)
                .collect::<Vec<_>>();
            assert_eq!(msgs.len(), 1);
            let challenge = msgs[0].challenge;
            let challenge_ext = EF::from_base_slice(&challenge);

            let omega_skip = F::two_adic_generator(l_skip);
            let domain_mask = domain_size - 1;
            let tidx_r = preflight.batch_constraint.tidx_before_multilinear - D_EF;
            rows.par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let coeff_idx = height - 1 - i;
                    let coeff_base_slice = coeffs[coeff_idx].as_base_slice();

                    let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_canonical_usize(pidx);
                    cols.r.copy_from_slice(&challenge);
                    cols.is_first = F::from_bool(i == 0);
                    cols.coeff_idx = F::from_canonical_usize(coeff_idx);

                    let exponent = coeff_idx & domain_mask;
                    cols.omega_skip_power = if exponent == 0 {
                        F::ONE
                    } else {
                        omega_skip.exp_u64(exponent as u64)
                    };
                    IsEqSubAir.generate_subrow(
                        (cols.omega_skip_power, F::ONE),
                        (
                            &mut cols.is_omega_skip_power_equal_to_one_aux.inv,
                            &mut cols.is_omega_skip_power_equal_to_one,
                        ),
                    );
                    cols.coeff.copy_from_slice(coeff_base_slice);
                    cols.tidx = F::from_canonical_usize(tidx_r - (i + 1) * D_EF);
                });

            let mut sum_at_roots = EF::ZERO;
            let mut value_at_r = EF::ZERO;
            let mut row_iter = rows.chunks_exact_mut(width);
            for i in 0..height {
                let coeff_idx = height - 1 - i;
                let chunk = row_iter.next().unwrap();
                let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
                let coeff = EF::from_base_slice(&cols.coeff);

                if coeff_idx & domain_mask == 0 {
                    sum_at_roots += coeff * domain_size_ext;
                }
                value_at_r = coeff + challenge_ext * value_at_r;

                cols.sum_at_roots
                    .copy_from_slice(sum_at_roots.as_base_slice());
                cols.value_at_r.copy_from_slice(value_at_r.as_base_slice());
            }
            debug_assert!(row_iter.next().is_none());
        });

    RowMajorMatrix::new(trace, width)
}
