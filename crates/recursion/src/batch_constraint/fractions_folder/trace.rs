use core::iter::zip;
use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{batch_constraint::fractions_folder::FractionsFolderCols, system::Preflight};

pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = FractionsFolderCols::<F>::width();

    let height = proofs
        .iter()
        .map(|p| {
            let res = p.batch_constraint_proof.numerator_term_per_air.len();
            debug_assert!(res > 0);
            res
        })
        .sum::<usize>();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;

    for (pidx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let (npa, dpa) = (
            &proof.batch_constraint_proof.numerator_term_per_air,
            &proof.batch_constraint_proof.denominator_term_per_air,
        );
        let height = npa.len();
        let mu_tidx = preflight.batch_constraint.tidx_before_univariate - D_EF;
        let mu_slice = &preflight.transcript.values()[mu_tidx..mu_tidx + D_EF];
        let tidx_alpha_beta = preflight.proof_shape.post_tidx + 2;
        let gkr_post_tidx = preflight.gkr.post_tidx;
        let n_global = preflight.proof_shape.n_global();

        let rows = &mut trace[cur_height * width..(cur_height + height) * width];
        rows.par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(i == 0);
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.air_idx = F::from_canonical_usize(height - 1 - i);
                cols.n_global = F::from_canonical_usize(n_global);
                cols.tidx_alpha_beta = F::from_canonical_usize(tidx_alpha_beta);
                cols.sum_claim_p.copy_from_slice(npa[i].as_base_slice());
                cols.sum_claim_q.copy_from_slice(dpa[i].as_base_slice());
                cols.gkr_post_tidx = F::from_canonical_usize(gkr_post_tidx);
                cols.mu.copy_from_slice(mu_slice);
                cols.tidx = F::from_canonical_usize(gkr_post_tidx + (1 + 2 * i) * D_EF);
            });

        let mut cur_p_sum = [F::ZERO; D_EF];
        let mut cur_q_sum: [_; D_EF] =
            core::array::from_fn(|i| preflight.transcript.values()[tidx_alpha_beta + i]);

        let mu = EF::from_base_slice(mu_slice);
        let mut cur_hash = EF::ZERO;
        for (i, chunk) in rows.chunks_exact_mut(width).enumerate() {
            let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
            for j in 0..D_EF {
                cur_p_sum[j] += cols.sum_claim_p[j];
                cur_q_sum[j] += cols.sum_claim_q[j];
            }
            cols.cur_p_sum.copy_from_slice(&cur_p_sum);
            cols.cur_q_sum.copy_from_slice(&cur_q_sum);

            cur_hash = npa[i] + mu * (dpa[i] + mu * cur_hash);
            cols.cur_hash = cur_hash.as_base_slice().try_into().unwrap();
        }

        cur_height += height;
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
