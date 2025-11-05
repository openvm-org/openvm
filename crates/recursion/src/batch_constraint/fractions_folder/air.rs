use std::borrow::Borrow;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, extension::BinomiallyExtendable};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{SumcheckClaimBus, SumcheckClaimMessage},
    bus::{BatchConstraintModuleBus, BatchConstraintModuleMessage, TranscriptBus},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct FractionsFolderCols<T> {
    pub is_valid: T,
    pub proof_idx: T,
    pub is_first: T,

    pub air_idx: T,

    // TODO(ayush): probably don't need all 3
    pub tidx: T,
    pub tidx_alpha_beta: T,
    pub gkr_post_tidx: T, // TODO: this number of tids is really annoying

    pub sum_claim_p: [T; D_EF],
    pub sum_claim_q: [T; D_EF],
    pub cur_p_sum: [T; D_EF],
    pub cur_q_sum: [T; D_EF],
    pub mu: [T; D_EF],
    pub cur_hash: [T; D_EF],
}

pub struct FractionsFolderAir {
    pub transcript_bus: TranscriptBus,
    pub sumcheck_bus: SumcheckClaimBus,
    pub gkr_claim_bus: BatchConstraintModuleBus,
}

impl<F> BaseAirWithPublicValues<F> for FractionsFolderAir {}
impl<F> PartitionedBaseAir<F> for FractionsFolderAir {}

impl<F> BaseAir<F> for FractionsFolderAir {
    fn width(&self) -> usize {
        FractionsFolderCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for FractionsFolderAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &FractionsFolderCols<AB::Var> = (*local).borrow();
        let next: &FractionsFolderCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<1, 0>;

        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols::default(),
            ),
        );

        let is_last = LoopSubAir::local_is_last(next.is_valid, next.is_first);
        let is_transition = AB::Expr::ONE - is_last.clone();

        // TODO(ayush): do i need to constrain that air_idx starts at a particular value?
        // builder
        //     .when(local.is_first)
        //     .assert_zero(local.air_idx);

        // Air index decrements by 1
        builder
            .when(local.is_valid * is_transition.clone())
            .assert_eq(next.air_idx, local.air_idx - AB::Expr::ONE);

        builder.when(is_last.clone()).assert_zero(local.air_idx);

        ///////////////////////////////////////////////////////////////////////
        // Transition Constraints
        ///////////////////////////////////////////////////////////////////////

        let is_transition = AB::Expr::ONE - is_last.clone();

        assert_array_eq(&mut builder.when(is_transition.clone()), local.mu, next.mu);

        builder.when(is_transition.clone()).assert_eq(
            next.tidx,
            local.tidx - AB::Expr::from_canonical_usize(2 * D_EF),
        );

        ///////////////////////////////////////////////////////////////////////
        // Running Sum Constraints
        ///////////////////////////////////////////////////////////////////////

        // Initialize running sum
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.cur_p_sum,
            local.sum_claim_p,
        );

        // Add air sum to running sum
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            next.cur_p_sum,
            ext_field_add(local.cur_p_sum, next.sum_claim_p),
        );

        ///////////////////////////////////////////////////////////////////////
        // Polynomial Hash Evaluation (a la Horner's Method)
        ///////////////////////////////////////////////////////////////////////

        // Initialize hash
        // h = p + mu * q
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.cur_hash,
            ext_field_add(
                local.sum_claim_p,
                ext_field_multiply(local.mu, local.sum_claim_q),
            ),
        );

        // Update hash
        // h' = p + mu * (q + mu * h)
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            next.cur_hash,
            ext_field_add(
                next.sum_claim_p,
                ext_field_multiply(
                    next.mu,
                    ext_field_add(
                        next.sum_claim_q,
                        ext_field_multiply(next.mu, local.cur_hash),
                    ),
                ),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // Sample mu
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_canonical_usize(D_EF),
            local.mu,
            local.is_valid * local.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.sum_claim_q,
            local.is_valid,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx - AB::Expr::from_canonical_usize(D_EF),
            local.sum_claim_p,
            local.is_valid,
        );
        // Sample alpha
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx_alpha_beta,
            ext_field_subtract(local.cur_q_sum, local.sum_claim_q),
            local.is_valid * local.is_first,
        );

        self.sumcheck_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: AB::Expr::ZERO,
                value: local.cur_hash.map(Into::into),
            },
            local.is_valid * is_last.clone(),
        );

        self.gkr_claim_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintModuleMessage {
                tidx_alpha_beta: local.tidx_alpha_beta.into(),
                tidx: local.gkr_post_tidx.into(),
                gkr_input_layer_claim: [
                    local.cur_p_sum.map(Into::into),
                    local.cur_q_sum.map(Into::into),
                ],
            },
            local.is_valid * is_last,
        );
    }
}
