use core::iter::zip;
use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::MultilinearSumcheckCols;
use crate::system::Preflight;

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = MultilinearSumcheckCols::<F>::width();
    let one_poly_height = vk.inner.max_constraint_degree + 2;
    let height = proofs
        .iter()
        .map(|p| p.batch_constraint_proof.sumcheck_round_polys.len().max(1))
        .sum::<usize>()
        * one_poly_height;
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;

    for (pidx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let polys = &proof.batch_constraint_proof.sumcheck_round_polys;
        let height = polys.len() * one_poly_height;
        let transcript_values = preflight.transcript.values();
        let tidx_before_multilinear = preflight.batch_constraint.tidx_before_multilinear;
        let stride = one_poly_height * D_EF;

        if height > 0 {
            trace[cur_height * width..cur_height * width + height * width]
                .par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(row_idx, chunk)| {
                    let poly_idx = row_idx / one_poly_height;
                    let within_poly = row_idx % one_poly_height;
                    let poly = &polys[poly_idx];

                    let local_tidx_start = tidx_before_multilinear + poly_idx * stride;
                    let r_tidx = local_tidx_start + (one_poly_height - 1) * D_EF;
                    let cols: &mut MultilinearSumcheckCols<F> = chunk.borrow_mut();

                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_canonical_usize(pidx);
                    cols.round = F::from_canonical_usize(poly_idx);
                    cols.r
                        .copy_from_slice(&transcript_values[r_tidx..r_tidx + D_EF]);

                    if within_poly == 0 {
                        cols.tidx = F::from_canonical_usize(r_tidx);
                        cols.i = F::ZERO;
                        cols.i_is_zero = F::ONE;
                        cols.eval_at_i.fill(F::ZERO);
                    } else {
                        let eval_idx = within_poly - 1;
                        cols.tidx = F::from_canonical_usize(local_tidx_start + eval_idx * D_EF);
                        cols.i = F::from_canonical_usize(within_poly);
                        cols.i_is_zero = F::ZERO;
                        cols.eval_at_i
                            .copy_from_slice(poly[eval_idx].as_base_slice());
                    }
                });
        } else {
            let cols: &mut MultilinearSumcheckCols<F> =
                trace[cur_height * width..(cur_height + 1) * width].borrow_mut();
            cols.proof_idx = F::from_canonical_usize(pidx);
            cur_height += 1;
        }
        cur_height += height;
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut MultilinearSumcheckCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
