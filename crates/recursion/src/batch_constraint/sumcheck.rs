use std::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_maybe_rayon::prelude::*;
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{SumcheckClaimBus, SumcheckClaimMessage},
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, TranscriptBus,
        TranscriptBusMessage,
    },
    system::Preflight,
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct UnivariateSumcheckCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    idx: T,
    idx_mod_domsize: T,
    idx_divisible_by_domsize: T,
    aux_inv_idx: T,
    coeff: [T; D_EF],
    sum_at_roots: [T; D_EF],

    // TODO: reuse all above as a subair?
    coeff_tidx: T, // TODO: coeff_tidx - idx should be a constant, probably derivable from tidx
    tidx: T,
    r: [T; D_EF],
    value_at_r: [T; D_EF],
}

pub struct UnivariateSumcheckAir {
    /// The degree of the univariate polynomial
    pub univariate_deg: usize,
    /// The univariate domain size, aka `2^{l_skip}`
    pub domain_size: usize,

    pub claim_bus: SumcheckClaimBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
}

impl<F> BaseAirWithPublicValues<F> for UnivariateSumcheckAir {}
impl<F> PartitionedBaseAir<F> for UnivariateSumcheckAir {}

impl<F> BaseAir<F> for UnivariateSumcheckAir {
    fn width(&self) -> usize {
        UnivariateSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnivariateSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &UnivariateSumcheckCols<AB::Var> = (*local).borrow();
        let _next: &UnivariateSumcheckCols<AB::Var> = (*next).borrow();

        /*
        // `is_valid` is 0/1 and switches only from 1 to 0
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);
        builder.assert_bool(local.is_valid);

        // while `is_valid`, the indices increase and start with 0
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(next.idx - local.idx);
        builder.when_first_row().assert_zero(local.idx);

        // moreover, `is_valid` is only 1 on the first `univariate_deg + 1` rows, which all exist:
        // - If it ever goes 1 -> 0, it happens on `idx = univariate_deg`
        builder.when(local.is_valid * not(next.is_valid)).assert_eq(
            local.idx,
            AB::Expr::from_canonical_usize(self.univariate_deg),
        );
        // - If it's all ones, then the last row index is this
        builder.when_last_row().when(local.is_valid).assert_eq(
            local.idx,
            AB::Expr::from_canonical_usize(self.univariate_deg),
        );
        // - We forbid it to be all zeroes
        builder.when_first_row().assert_one(local.is_valid);

        // `idx_mod_domsize` starts with 0,
        builder.when_first_row().assert_zero(local.idx_mod_domsize);
        // each time either increases by 1 or drop to 0,
        builder
            .when(next.idx_mod_domsize)
            .assert_one(next.idx_mod_domsize - local.idx_mod_domsize);
        // and `idx_divisible_by_domsize` is the indicator of it being 0.
        builder.assert_bool(local.idx_divisible_by_domsize);
        builder
            .when(local.idx_divisible_by_domsize)
            .assert_zero(local.idx_mod_domsize);
        builder
            .when(next.is_valid)
            .when(not(next.idx_divisible_by_domsize))
            .assert_one(next.idx_mod_domsize - local.idx_mod_domsize);

        // More specifically, it is zero iff (is valid and) the index was going to be divisibly by `domain_size`
        builder
            .when_transition()
            .when(next.idx_divisible_by_domsize)
            .assert_eq(
                local.idx_mod_domsize + AB::Expr::ONE,
                AB::Expr::from_canonical_usize(self.domain_size),
            );
        builder
            .when(local.is_valid)
            .assert_one(local.aux_inv_idx * local.idx_mod_domsize);

        // `sum_at_roots` needs to change this way:
        builder.when_transition().assert_eq(
            local.sum_at_roots,
            next.sum_at_roots + local.idx_divisible_by_domsize * local.coeff,
        );
        // and it must _always_ equal coeff on the last row -- this holds
        // even when not `is_valid`
        builder
            .when_last_row()
            .assert_eq(local.sum_at_roots, local.coeff);
        */

        self.claim_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.value_at_r.map(|x| x.into()),
            },
            local.is_first,
        );
        for i in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.r[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.is_first,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.coeff_tidx + AB::Expr::from_canonical_usize(i),
                    value: local.coeff[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid,
            );
        }
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: AB::Expr::ZERO,
                challenge: local.r.map(|x| x.into()),
            },
            local.is_first,
        );
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MultilinearSumcheckCols<T> {
    is_valid: T,
    is_first: T,
    is_last: T,
    proof_idx: T,

    tidx: T,
    round: T,
    r: [T; D_EF],
    prefix_invfact: T,
    suffix_invfact: T,
    forw_invfact: T,
    back_invfact: T,
    i: T,
    eval_at_i: [T; D_EF],
    cur_sum: [T; D_EF],
    i_is_zero: T,
}

pub struct MultilinearSumcheckAir {
    pub claim_bus: SumcheckClaimBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
}

impl<F> BaseAirWithPublicValues<F> for MultilinearSumcheckAir {}
impl<F> PartitionedBaseAir<F> for MultilinearSumcheckAir {}

impl<F> BaseAir<F> for MultilinearSumcheckAir {
    fn width(&self) -> usize {
        MultilinearSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for MultilinearSumcheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MultilinearSumcheckCols<AB::Var> = (*local).borrow();
        let _next: &MultilinearSumcheckCols<AB::Var> = (*next).borrow();

        // ...

        for i in 0..D_EF {
            self.claim_bus.receive(
                builder,
                local.proof_idx,
                SumcheckClaimMessage {
                    round: local.round.into(),
                    value: local.eval_at_i.map(|x| x.into()),
                },
                local.is_first.into(),
            );
            self.claim_bus.send(
                builder,
                local.proof_idx,
                SumcheckClaimMessage {
                    round: local.round.into() + AB::Expr::ONE,
                    value: local.cur_sum.map(|x| x.into()),
                },
                local.is_first.into(),
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.r[i].into(),
                    is_sample: AB::Expr::ONE,
                },
                local.i_is_zero,
            );
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                    value: local.eval_at_i[i].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.is_valid - local.i_is_zero,
            );
        }
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round + AB::Expr::ONE,
                challenge: local.r.map(|x| x.into()),
            },
            local.i_is_zero,
        );
    }
}

pub(crate) fn generate_univariate_trace(
    _vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let width = UnivariateSumcheckCols::<F>::width();
    let msgs = preflight
        .batch_constraint_sumcheck_randomness()
        .into_iter()
        .filter(|x| x.idx == F::ZERO)
        .collect::<Vec<_>>();
    assert_eq!(msgs.len(), 1);
    let coeffs = &proof.batch_constraint_proof.univariate_round_coeffs;
    let height = coeffs.len();
    let mut trace = vec![F::ZERO; width * height];
    let challenge = msgs[0].challenge;
    let coeff_tidx_base = preflight.batch_constraint.tidx_before_univariate;
    let tidx_offset = proof.batch_constraint_proof.univariate_round_coeffs.len() * D_EF;
    let tidx_constant = coeff_tidx_base + tidx_offset;

    trace
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, chunk)| {
            let cols: &mut UnivariateSumcheckCols<F> = chunk.borrow_mut();
            cols.is_valid = F::ONE;
            cols.r.copy_from_slice(&challenge);
            cols.is_first = F::from_bool(i == 0);
            cols.is_last = F::from_bool(i + 1 == height);
            cols.coeff.copy_from_slice(coeffs[i].as_base_slice());
            cols.coeff_tidx = F::from_canonical_usize(coeff_tidx_base + i * D_EF);
            cols.tidx = F::from_canonical_usize(tidx_constant);
        });

    RowMajorMatrix::new(trace, width)
}

pub(crate) fn generate_multilinear_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let width = MultilinearSumcheckCols::<F>::width();
    let one_poly_height = vk.inner.max_constraint_degree + 2;
    let polys = &proof.batch_constraint_proof.sumcheck_round_polys;
    let height = polys.len() * one_poly_height;
    let mut trace = vec![F::ZERO; width * height];
    let transcript_values = preflight.transcript.values();
    let tidx_before_multilinear = preflight.batch_constraint.tidx_before_multilinear;
    let stride = one_poly_height * D_EF;

    trace
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

    RowMajorMatrix::new(trace, width)
}
