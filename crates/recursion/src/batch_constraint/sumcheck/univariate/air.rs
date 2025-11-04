use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::{
        BatchConstraintConductorBus,
        bus::{
            BatchConstraintConductorMessage, BatchConstraintInnerMessageType, SumcheckClaimBus,
            SumcheckClaimMessage,
        },
    },
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, TranscriptBus,
        TranscriptBusMessage,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct UnivariateSumcheckCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_idx: T,

    pub idx: T,
    pub idx_mod_domsize: T,
    pub idx_divisible_by_domsize: T,
    pub aux_inv_idx: T,
    pub coeff: [T; D_EF],
    pub sum_at_roots: [T; D_EF],

    // TODO: reuse all above as a subair?
    pub coeff_tidx: T, // TODO: coeff_tidx - idx should be a constant, probably derivable from tidx
    pub tidx: T,
    pub r: [T; D_EF],
    pub value_at_r: [T; D_EF],
}

pub struct UnivariateSumcheckAir {
    /// The degree of the univariate polynomial
    pub univariate_deg: usize,
    /// The univariate domain size, aka `2^{l_skip}`
    pub domain_size: usize,

    pub claim_bus: SumcheckClaimBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
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
        self.batch_constraint_conductor_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: AB::Expr::ZERO,
                value: local.r.map(|x| x.into()),
            },
            local.is_first * AB::Expr::TWO,
        );
    }
}
