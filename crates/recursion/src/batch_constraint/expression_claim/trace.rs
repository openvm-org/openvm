use std::borrow::BorrowMut;

use p3_field::{Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use super::ExpressionClaimCols;
use crate::{
    batch_constraint::expr_eval::{ConstraintsFoldingBlob, InteractionsFoldingBlob},
    primitives::pow::PowerCheckerTraceGenerator,
    system::Preflight,
    utils::MultiProofVecVec,
};

pub struct ExpressionClaimBlob {
    // (n, value), n is before lift, can be negative
    claims: MultiProofVecVec<(isize, EF)>,
}

pub fn generate_expression_claim_blob(
    cf_blob: &ConstraintsFoldingBlob,
    if_blob: &InteractionsFoldingBlob,
) -> ExpressionClaimBlob {
    let mut claims = MultiProofVecVec::new();
    for pidx in 0..cf_blob.folded_claims.num_proofs() {
        claims.extend(if_blob.folded_claims[pidx].iter().cloned());
        claims.extend(cf_blob.folded_claims[pidx].iter().cloned());
        claims.end_proof();
    }
    ExpressionClaimBlob { claims }
}

#[tracing::instrument(name = "generate_trace(ExpressionClaimAir)", skip_all)]
pub(in crate::batch_constraint) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    blob: &ExpressionClaimBlob,
    proofs: &[Proof],
    preflights: &[Preflight],
    pow_checker: &PowerCheckerTraceGenerator<2, 32>,
) -> RowMajorMatrix<F> {
    let width = ExpressionClaimCols::<F>::width();

    let height = blob.claims.len();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let claims = &blob.claims[pidx];

        let num_rounds = proofs[pidx]
            .batch_constraint_proof
            .sumcheck_round_polys
            .len();
        let num_present = preflight.proof_shape.sorted_trace_vdata.len();
        debug_assert_eq!(claims.len(), 3 * num_present);
        let mu_tidx = preflight.batch_constraint.tidx_before_univariate - D_EF;

        trace[cur_height * width..(cur_height + claims.len()) * width]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let n_lift = claims[i].0.max(0) as usize;
                let n_abs = claims[i].0.unsigned_abs() as usize;
                let is_interaction = i < 2 * num_present;
                if is_interaction {
                    pow_checker.add_pow(n_abs);
                }
                let cols: &mut ExpressionClaimCols<_> = chunk.borrow_mut();
                cols.is_first = F::from_bool(i == 0);
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.is_interaction = F::from_bool(is_interaction);
                cols.num_multilinear_sumcheck_rounds = F::from_canonical_usize(num_rounds);
                cols.idx = F::from_canonical_usize(if i < 2 * num_present {
                    i
                } else {
                    i - 2 * num_present
                });
                cols.idx_parity = F::from_bool(is_interaction && i % 2 == 1);
                let trace_idx = if is_interaction {
                    i / 2
                } else {
                    i - 2 * num_present
                };
                cols.trace_idx = F::from_canonical_usize(trace_idx);
                cols.mu
                    .copy_from_slice(&preflight.transcript.values()[mu_tidx..mu_tidx + D_EF]);
                cols.value.copy_from_slice(claims[i].1.as_base_slice());
                cols.eq_sharp_ns.copy_from_slice(
                    preflight.batch_constraint.eq_sharp_ns_frontloaded[n_lift].as_base_slice(),
                );
                cols.multiplier.copy_from_slice(EF::ONE.as_base_slice());
                cols.n_abs = F::from_canonical_usize(n_abs);
                cols.n_sign = F::from_bool(claims[i].0.is_negative());
                cols.n_abs_pow = F::from_canonical_usize(1 << n_abs);
            });

        // Setting `cur_sum`
        let mut cur_sum = EF::ZERO;
        let mu = EF::from_base_slice(&preflight.transcript.values()[mu_tidx..mu_tidx + D_EF]);
        trace[cur_height * width..(cur_height + claims.len()) * width]
            .chunks_exact_mut(width)
            .rev()
            .for_each(|chunk| {
                let cols: &mut ExpressionClaimCols<_> = chunk.borrow_mut();
                // if it's interaction, we need to multiply by eq_sharp_ns and norm_factor
                let multiplier = if cols.is_interaction == F::ONE {
                    let mut mult = EF::from_base_slice(&cols.eq_sharp_ns);
                    if cols.n_sign == F::ONE && cols.idx.as_canonical_u32() % 2 == 0 {
                        mult *=
                            F::from_canonical_usize(1 << cols.n_abs.as_canonical_u32() as usize)
                                .inverse();
                    }
                    mult
                } else {
                    EF::ONE
                };
                cols.multiplier.copy_from_slice(multiplier.as_base_slice());
                cur_sum = cur_sum * mu + EF::from_base_slice(&cols.value) * multiplier;
                cols.cur_sum.copy_from_slice(cur_sum.as_base_slice());
            });

        cur_height += claims.len();
    }
    trace[cur_height * width..]
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut ExpressionClaimCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_canonical_usize(preflights.len() + i);
        });
    RowMajorMatrix::new(trace, width)
}
