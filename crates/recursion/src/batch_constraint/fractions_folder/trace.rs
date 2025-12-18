use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{batch_constraint::fractions_folder::FractionsFolderCols, system::Preflight};

#[tracing::instrument(level = "trace", skip_all)]
pub(crate) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = FractionsFolderCols::<F>::width();

    let rows_per_proof: Vec<usize> = proofs
        .iter()
        .map(|proof| {
            let res = proof.batch_constraint_proof.numerator_term_per_air.len();
            debug_assert!(res > 0);
            res
        })
        .collect();

    let total_height: usize = rows_per_proof.iter().sum();
    let padded_height = total_height.max(1).next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];

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
            let (npa, dpa) = (
                &proof.batch_constraint_proof.numerator_term_per_air,
                &proof.batch_constraint_proof.denominator_term_per_air,
            );
            let height = npa.len();
            let mu_tidx = preflight.batch_constraint.tidx_before_univariate - D_EF;
            let mu_slice = &preflight.transcript.values()[mu_tidx..mu_tidx + D_EF];
            let tidx_alpha_beta = preflight.proof_shape.post_tidx + 2;

            debug_assert_eq!(rows.len(), height * width);

            rows.par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let air_idx = height - 1 - i;

                    let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
                    cols.is_valid = F::ONE;
                    cols.is_first = F::from_bool(i == 0);
                    cols.proof_idx = F::from_canonical_usize(pidx);
                    cols.air_idx = F::from_canonical_usize(air_idx);
                    cols.sum_claim_p
                        .copy_from_slice(npa[air_idx].as_base_slice());
                    cols.sum_claim_q
                        .copy_from_slice(dpa[air_idx].as_base_slice());
                    cols.mu.copy_from_slice(mu_slice);
                    cols.tidx = F::from_canonical_usize(mu_tidx - 2 * D_EF - i * 2 * D_EF);
                });

            let mut cur_p_sum = [F::ZERO; D_EF];
            let mut cur_q_sum: [_; D_EF] =
                core::array::from_fn(|i| preflight.transcript.values()[tidx_alpha_beta + i]);

            let mu = EF::from_base_slice(mu_slice);
            let mut cur_hash = EF::ZERO;
            for (i, chunk) in rows.chunks_exact_mut(width).enumerate() {
                let air_idx = height - 1 - i;

                let cols: &mut FractionsFolderCols<F> = chunk.borrow_mut();
                for j in 0..D_EF {
                    cur_p_sum[j] += cols.sum_claim_p[j];
                    cur_q_sum[j] += cols.sum_claim_q[j];
                }
                cols.cur_p_sum.copy_from_slice(&cur_p_sum);
                cols.cur_q_sum.copy_from_slice(&cur_q_sum);

                cur_hash = npa[air_idx] + mu * (dpa[air_idx] + mu * cur_hash);
                cols.cur_hash = cur_hash.as_base_slice().try_into().unwrap();
            }
        });

    RowMajorMatrix::new(trace, width)
}
