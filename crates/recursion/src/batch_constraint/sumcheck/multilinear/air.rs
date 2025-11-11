use std::borrow::Borrow;

use openvm_circuit_primitives::{SubAir, utils::assert_array_eq};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra, extension::BinomiallyExtendable};
use p3_matrix::Matrix;
use stark_backend_v2::D_EF;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, SumcheckClaimBus, SumcheckClaimMessage,
    },
    bus::{
        ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus, StackingModuleBus,
        StackingModuleMessage, TranscriptBus,
    },
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        assert_one_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_subtract_scalar, scalar_subtract_ext_field,
    },
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct MultilinearSumcheckCols<T> {
    pub is_valid: T,
    pub proof_idx: T,
    pub round_idx: T,
    pub is_round_start: T,
    pub is_first_eval: T,

    pub nested_for_loop_aux_cols: NestedForLoopAuxCols<T, 1>,

    pub eval_idx: T,

    pub cur_sum: [T; D_EF],
    pub eval: [T; D_EF],

    // TODO(ayush): can be preprocessed cols
    // 1 / i!(d - i)!
    pub denom_inv: T,

    pub prefix_product: [T; D_EF],
    pub suffix_product: [T; D_EF],

    pub r: [T; D_EF],

    pub tidx: T,
}

pub struct MultilinearSumcheckAir {
    pub max_constraint_degree: usize,
    pub claim_bus: SumcheckClaimBus,
    pub transcript_bus: TranscriptBus,
    pub randomness_bus: ConstraintSumcheckRandomnessBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,
    pub stacking_module_bus: StackingModuleBus,
}

impl<F> BaseAirWithPublicValues<F> for MultilinearSumcheckAir {}
impl<F> PartitionedBaseAir<F> for MultilinearSumcheckAir {}

impl<F> BaseAir<F> for MultilinearSumcheckAir {
    fn width(&self) -> usize {
        MultilinearSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for MultilinearSumcheckAir
where
    <AB::Expr as FieldAlgebra>::F: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &MultilinearSumcheckCols<AB::Var> = (*local).borrow();
        let next: &MultilinearSumcheckCols<AB::Var> = (*next).borrow();

        let s_deg = self.max_constraint_degree + 1;

        ///////////////////////////////////////////////////////////////////////
        // Loop Constraints
        ///////////////////////////////////////////////////////////////////////

        type LoopSubAir = NestedForLoopSubAir<2, 1>;
        LoopSubAir {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx, local.round_idx],
                        is_first: [local.is_round_start, local.is_first_eval],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx, next.round_idx],
                        is_first: [next.is_round_start, next.is_first_eval],
                    }
                    .map_into(),
                ),
                local.nested_for_loop_aux_cols.map_into(),
            ),
        );

        let is_last_round = LoopSubAir::local_is_last(next.is_valid, next.is_round_start);
        let is_last_eval = LoopSubAir::local_is_last(next.is_valid, next.is_first_eval);

        // Eval idx starts at 0
        builder
            .when(local.is_first_eval)
            .assert_zero(local.eval_idx);
        // Eval idx increments by 1
        builder
            .when(AB::Expr::ONE - is_last_eval.clone())
            .assert_eq(next.eval_idx, local.eval_idx + AB::Expr::ONE);
        // Eval idx ends at s_deg
        builder
            .when(local.is_valid * is_last_eval.clone())
            .assert_eq(local.eval_idx, AB::Expr::from_canonical_usize(s_deg));

        ///////////////////////////////////////////////////////////////////////
        // Factorials Constraints
        ///////////////////////////////////////////////////////////////////////

        // TODO(ayush): cache
        let d_factorial_inv = AB::Expr::from_f(
            <AB::Expr as FieldAlgebra>::F::from_canonical_usize((1..=s_deg).product()).inverse(),
        );
        // Starts at d!
        builder
            .when(local.is_valid * local.is_first_eval)
            .assert_eq(local.denom_inv, d_factorial_inv);
        // 1 / (i + 1)!(d - i - 1)!  (i + 1) = 1 / i!(d - i)! * (d - i)
        builder
            .when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone()))
            .assert_eq(
                next.denom_inv * next.eval_idx,
                local.denom_inv * (AB::Expr::from_canonical_usize(s_deg) - local.eval_idx),
            );

        ///////////////////////////////////////////////////////////////////////
        // Prefix/Suffix Product Constraints
        ///////////////////////////////////////////////////////////////////////

        assert_array_eq(
            &mut builder.when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone())),
            next.r,
            local.r,
        );

        // Prefix Product
        // Starts at 1
        assert_one_ext(
            &mut builder.when(local.is_valid * local.is_first_eval),
            local.prefix_product,
        );
        // p' = p * (r - i)
        assert_array_eq(
            &mut builder.when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone())),
            next.prefix_product,
            ext_field_multiply(
                local.prefix_product,
                ext_field_subtract_scalar(local.r, local.eval_idx),
            ),
        );

        // Suffix Product
        // Ends at 1
        assert_one_ext(
            &mut builder.when(local.is_valid * is_last_eval.clone()),
            local.suffix_product,
        );
        // s = s' * ((d - i) - r)
        assert_array_eq(
            &mut builder.when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone())),
            local.suffix_product,
            ext_field_multiply(
                next.suffix_product,
                scalar_subtract_ext_field(
                    AB::Expr::from_canonical_usize(s_deg) - local.eval_idx,
                    local.r,
                ),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Sumcheck evaluation constraints
        ///////////////////////////////////////////////////////////////////////

        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first_eval),
            local.cur_sum,
            ext_field_multiply(
                local.eval,
                ext_field_multiply_scalar(
                    ext_field_multiply(local.prefix_product, local.suffix_product),
                    local.denom_inv,
                ),
            ),
        );

        assert_array_eq(
            &mut builder.when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone())),
            next.cur_sum,
            ext_field_add(
                local.cur_sum,
                ext_field_multiply(
                    next.eval,
                    ext_field_multiply_scalar(
                        ext_field_multiply(next.prefix_product, next.suffix_product),
                        next.denom_inv,
                    ),
                ),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Transition constraints
        ///////////////////////////////////////////////////////////////////////

        // TODO(ayush): tidx initial value should be constrained using some interaction
        builder
            .when(local.is_valid * (AB::Expr::ONE - is_last_eval.clone()))
            .assert_eq(next.tidx, local.tidx + AB::Expr::from_canonical_usize(D_EF));

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // Observe evaluations s(1), s(2) etc.
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            next.eval,
            local.is_valid * (AB::Expr::ONE - is_last_eval.clone()),
        );
        // Sample challenge r
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.r,
            local.is_valid * is_last_eval.clone(),
        );

        self.stacking_module_bus.send(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(D_EF),
            },
            local.is_valid * is_last_round.clone(),
        );

        self.claim_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: local.round_idx.into(),
                value: ext_field_add(local.eval, next.eval),
            },
            local.is_valid * local.is_first_eval,
        );
        self.claim_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: local.round_idx + AB::Expr::ONE,
                value: local.cur_sum.map(Into::into),
            },
            local.is_valid * is_last_eval.clone()
            // TODO(ayush): remove this when there's an air that can receive the final sumcheck claim
            * local.nested_for_loop_aux_cols.is_transition[0],
        );
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round_idx + AB::Expr::ONE,
                challenge: local.r.map(|x| x.into()),
            },
            local.is_valid * local.is_first_eval,
        );
        self.batch_constraint_conductor_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.round_idx + AB::Expr::ONE,
                value: local.r.map(|x| x.into()),
            },
            local.is_valid * local.is_first_eval,
        );
    }
}
