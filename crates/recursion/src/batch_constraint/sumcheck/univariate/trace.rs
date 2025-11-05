use core::iter::zip;
use std::borrow::BorrowMut;

use openvm_circuit_primitives::{TraceSubRowGenerator, is_equal::IsEqSubAir};
use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::UnivariateSumcheckCols;
use crate::system::Preflight;

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
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

    let l_skip = vk.inner.params.l_skip;
    let domain_size = 1usize << l_skip;
    let domain_size_ext = EF::from_canonical_usize(domain_size);

    for (pidx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
        let coeffs = &proof.batch_constraint_proof.univariate_round_coeffs;
        let height = coeffs.len();

        let msgs = preflight
            .batch_constraint_sumcheck_randomness()
            .into_iter()
            .filter(|x| x.idx == F::ZERO)
            .collect::<Vec<_>>();
        assert_eq!(msgs.len(), 1);
        let challenge = msgs[0].challenge;
        let challenge_ext = EF::from_base_slice(&challenge);

        let omega_skip = F::two_adic_generator(l_skip);
        let omega_skip_inv = omega_skip.inverse();

        let rows = &mut trace[cur_height * width..(cur_height + height) * width];
        let mut sum_at_roots = EF::ZERO;
        let mut value_at_r = EF::ZERO;
        let mut omega_pow = if height == 0 {
            F::ONE
        } else {
            // (height - 1) % domain_size
            let exponent = (height - 1) & (domain_size - 1);
            omega_skip.exp_u64(exponent as u64)
        };
        let tidx_r = preflight.batch_constraint.tidx_before_multilinear - D_EF;
        for (i, chunk) in rows.chunks_mut(width).enumerate() {
            let coeff_idx = height - 1 - i;
            let coeff = coeffs[coeff_idx];
            // coeff_idx % domain_size == 0
            if coeff_idx & (domain_size - 1) == 0 {
                sum_at_roots += coeff * domain_size_ext;
            }
            value_at_r = coeff + challenge_ext * value_at_r;

            let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
            cols.is_valid = F::ONE;
            cols.proof_idx = F::from_canonical_usize(pidx);
            cols.r.copy_from_slice(&challenge);
            cols.is_first = F::from_bool(i == 0);
            cols.coeff_idx = F::from_canonical_usize(coeff_idx);

            cols.omega_skip_power = omega_pow;
            IsEqSubAir.generate_subrow(
                (cols.omega_skip_power, F::ONE),
                (
                    &mut cols.is_omega_skip_power_equal_to_one_aux.inv,
                    &mut cols.is_omega_skip_power_equal_to_one,
                ),
            );

            cols.coeff.copy_from_slice(coeff.as_base_slice());
            cols.sum_at_roots
                .copy_from_slice(sum_at_roots.as_base_slice());
            cols.value_at_r.copy_from_slice(value_at_r.as_base_slice());

            cols.tidx = F::from_canonical_usize(tidx_r - (i + 1) * D_EF);
            // TODO(ayush): remove dependency on previous iteration
            omega_pow *= omega_skip_inv;
        }

        cur_height += height;
    }

    // TODO(ayush): remove
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(proofs.len() + i);
        });

    RowMajorMatrix::new(trace, width)
}
