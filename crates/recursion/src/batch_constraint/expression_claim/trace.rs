use std::borrow::BorrowMut;

use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, proof::Proof};

use super::ExpressionClaimCols;
use crate::{
    primitives::pow::PowerCheckerTraceGenerator, system::Preflight, tracegen::RowMajorChip,
    utils::MultiProofVecVec,
};

pub struct ExpressionClaimBlob {
    // (n, value), n is before lift, can be negative
    claims: MultiProofVecVec<(isize, EF)>,
}

pub fn generate_expression_claim_blob(
    cf_folded_claims: &MultiProofVecVec<(isize, EF)>,
    if_folded_claims: &MultiProofVecVec<(isize, EF)>,
) -> ExpressionClaimBlob {
    let mut claims = MultiProofVecVec::new();
    for pidx in 0..cf_folded_claims.num_proofs() {
        claims.extend(if_folded_claims[pidx].iter().cloned());
        claims.extend(cf_folded_claims[pidx].iter().cloned());
        claims.end_proof();
    }
    ExpressionClaimBlob { claims }
}

pub struct ExpressionClaimTraceGenerator;

pub(crate) struct ExpressionClaimCtx<'a> {
    pub blob: &'a ExpressionClaimBlob,
    pub proofs: &'a [&'a Proof],
    pub preflights: &'a [&'a Preflight],
    pub pow_checker: &'a PowerCheckerTraceGenerator<2, 32>,
}

impl RowMajorChip<F> for ExpressionClaimTraceGenerator {
    type Ctx<'a> = ExpressionClaimCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let blob = ctx.blob;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        let pow_checker = ctx.pow_checker;
        let width = ExpressionClaimCols::<F>::width();

        let num_valid = blob.claims.len();
        let padded_height = if let Some(height) = required_height {
            if height < num_valid {
                return None;
            }
            height
        } else {
            num_valid.next_power_of_two()
        };
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
                    let n_abs = claims[i].0.unsigned_abs();
                    let is_interaction = i < 2 * num_present;
                    if is_interaction {
                        pow_checker.add_pow(n_abs);
                    }
                    let cols: &mut ExpressionClaimCols<_> = chunk.borrow_mut();
                    cols.is_first = F::from_bool(i == 0);
                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_usize(pidx);
                    cols.is_interaction = F::from_bool(is_interaction);
                    cols.num_multilinear_sumcheck_rounds = F::from_usize(num_rounds);
                    cols.idx = F::from_usize(if i < 2 * num_present {
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
                    cols.trace_idx = F::from_usize(trace_idx);
                    cols.mu
                        .copy_from_slice(&preflight.transcript.values()[mu_tidx..mu_tidx + D_EF]);
                    cols.value
                        .copy_from_slice(claims[i].1.as_basis_coefficients_slice());
                    cols.eq_sharp_ns.copy_from_slice(
                        preflight.batch_constraint.eq_sharp_ns_frontloaded[n_lift]
                            .as_basis_coefficients_slice(),
                    );
                    cols.multiplier
                        .copy_from_slice(EF::ONE.as_basis_coefficients_slice());
                    cols.n_abs = F::from_usize(n_abs);
                    cols.n_sign = F::from_bool(claims[i].0.is_negative());
                    cols.n_abs_pow = F::from_usize(1 << n_abs);
                });

            // Setting `cur_sum`
            let mut cur_sum = EF::ZERO;
            let mu = EF::from_basis_coefficients_slice(
                &preflight.transcript.values()[mu_tidx..mu_tidx + D_EF],
            )
            .unwrap();
            trace[cur_height * width..(cur_height + claims.len()) * width]
                .chunks_exact_mut(width)
                .rev()
                .for_each(|chunk| {
                    let cols: &mut ExpressionClaimCols<_> = chunk.borrow_mut();
                    // if it's interaction, we need to multiply by eq_sharp_ns and norm_factor
                    let multiplier = if cols.is_interaction == F::ONE {
                        let mut mult =
                            EF::from_basis_coefficients_slice(&cols.eq_sharp_ns).unwrap();
                        if cols.n_sign == F::ONE && cols.idx.as_canonical_u32() % 2 == 0 {
                            mult *= F::from_u32(1 << cols.n_abs.as_canonical_u32()).inverse();
                        }
                        mult
                    } else {
                        EF::ONE
                    };
                    cols.multiplier
                        .copy_from_slice(multiplier.as_basis_coefficients_slice());
                    cur_sum = cur_sum * mu
                        + EF::from_basis_coefficients_slice(&cols.value).unwrap() * multiplier;
                    cols.cur_sum
                        .copy_from_slice(cur_sum.as_basis_coefficients_slice());
                });

            cur_height += claims.len();
        }
        trace[cur_height * width..]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut ExpressionClaimCols<F> = chunk.borrow_mut();
                cols.proof_idx = F::from_usize(preflights.len() + i);
            });
        Some(RowMajorMatrix::new(trace, width))
    }
}
