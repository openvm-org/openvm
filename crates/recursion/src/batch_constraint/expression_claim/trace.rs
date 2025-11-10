use std::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2};

use super::ExpressionClaimCols;
use crate::{
    batch_constraint::expr_eval::{ConstraintsFoldingBlob, InteractionsFoldingBlob},
    system::Preflight,
    utils::MultiProofVecVec,
};

struct ExpressionClaimBlob {
    claims: MultiProofVecVec<EF>,
}

fn generate_expression_claim_blob(
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

pub(in crate::batch_constraint) fn generate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    cf_blob: &ConstraintsFoldingBlob,
    if_blob: &InteractionsFoldingBlob,
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
    let width = ExpressionClaimCols::<F>::width();

    let blob = generate_expression_claim_blob(cf_blob, if_blob);

    let height = blob.claims.len();
    let padded_height = height.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_height * width];
    let mut cur_height = 0;
    for (pidx, preflight) in preflights.iter().enumerate() {
        let claims = &blob.claims[pidx];

        let num_present = preflight.proof_shape.sorted_trace_vdata.len();
        debug_assert_eq!(claims.len(), 3 * num_present);
        let mu_tidx = preflight.batch_constraint.tidx_before_univariate - D_EF;

        trace[cur_height * width..(cur_height + claims.len()) * width]
            .par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let cols: &mut ExpressionClaimCols<_> = chunk.borrow_mut();
                cols.is_first = F::from_bool(i == 0);
                cols.is_valid = F::ONE;
                cols.proof_idx = F::from_canonical_usize(pidx);
                cols.num_present = F::from_canonical_usize(num_present);
                cols.is_interaction = F::from_bool(i < 2 * num_present);
                cols.idx = F::from_canonical_usize(if i < 2 * num_present {
                    i
                } else {
                    i - 2 * num_present
                });
                cols.mu
                    .copy_from_slice(&preflight.transcript.values()[mu_tidx..mu_tidx + D_EF]);
                cols.value.copy_from_slice(claims[i].as_base_slice());
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
