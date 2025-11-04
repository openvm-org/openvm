use core::iter::zip;
use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::UnivariateSumcheckCols;
use crate::system::Preflight;

pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = UnivariateSumcheckCols::<F>::width();
    let height = proofs
        .iter()
        .map(|p| {
            let res = p.batch_constraint_proof.univariate_round_coeffs.len();
            debug_assert!(res > 0);
            res
        })
        .sum::<usize>();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;

    for (pidx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let msgs = preflight
            .batch_constraint_sumcheck_randomness()
            .into_iter()
            .filter(|x| x.idx == F::ZERO)
            .collect::<Vec<_>>();
        assert_eq!(msgs.len(), 1);
        let coeffs = &proof.batch_constraint_proof.univariate_round_coeffs;
        let height = coeffs.len();
        let challenge = msgs[0].challenge;
        let coeff_tidx_base = preflight.batch_constraint.tidx_before_univariate;
        let tidx_offset = proof.batch_constraint_proof.univariate_round_coeffs.len() * D_EF;
        let tidx_constant = coeff_tidx_base + tidx_offset;

        trace[cur_height * width..(cur_height + height) * width]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.r.copy_from_slice(&challenge);
                cols.is_first = F::from_bool(i == 0);
                cols.is_last = F::from_bool(i + 1 == height);
                cols.coeff.copy_from_slice(coeffs[i].as_base_slice());
                cols.coeff_tidx = F::from_canonical_usize(coeff_tidx_base + i * D_EF);
                cols.tidx = F::from_canonical_usize(tidx_constant);
            });

        cur_height += height;
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
