use std::borrow::BorrowMut;

use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::ExpressionClaimCols;
use crate::system::Preflight;

pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    _proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = ExpressionClaimCols::<F>::width();

    let height = preflights.len();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    for (i, preflight) in preflights.iter().enumerate() {
        let cols: &mut ExpressionClaimCols<_> = trace[i * width..(i + 1) * width].borrow_mut();
        cols.is_first = F::ONE;
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_canonical_usize(i);
        let tidx = preflight.gkr.post_tidx;
        cols.lambda_tidx = F::from_canonical_usize(tidx);
        cols.lambda
            .copy_from_slice(&preflight.transcript.values()[tidx..tidx + D_EF]);
    }
    trace[height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut ExpressionClaimCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });
    RowMajorMatrix::new(trace, width)
}
