use std::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
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
    pub is_proof_start: T,
    pub is_first_eval: T,

    pub nested_for_loop_aux_cols: NestedForLoopAuxCols<T, 1>,

    /// A valid row which is not involved in any interactions
    /// but should satisfy air constraints
    pub is_dummy: T,

    pub eval_idx: T,

    pub cur_sum: [T; D_EF],
    pub eval: [T; D_EF],

    pub prefix_product: [T; D_EF],
    pub suffix_product: [T; D_EF],
    // 1 / i!(d - i)!
    pub denom_inv: T,

    // Lagrange coefficients for the interpolated polynomial
    // prefix_product * suffix_product * denom_inv
    pub lagrange_coeff: [T; D_EF],

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
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &MultilinearSumcheckCols<AB::Var> = (*local).borrow();
        let next: &MultilinearSumcheckCols<AB::Var> = (*next).borrow();

        let s_deg = self.max_constraint_degree + 1;

        ///////////////////////////////////////////////////////////////////////
        // Boolean Constraints
        ///////////////////////////////////////////////////////////////////////

        builder.assert_bool(local.is_dummy);

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
                        is_first: [local.is_proof_start, local.is_first_eval],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx, next.round_idx],
                        is_first: [next.is_proof_start, next.is_first_eval],
                    }
                    .map_into(),
                ),
                local.nested_for_loop_aux_cols.map_into(),
            ),
        );

        // TODO(ayush): move to NestedForLoopSubAir
        builder.when(local.is_first_eval).assert_one(local.is_valid);
        builder
            .when(local.is_proof_start)
            .assert_one(local.is_valid);

        let is_transition_eval = next.is_valid - next.is_first_eval;
        let is_last_eval = local.is_valid - is_transition_eval.clone();

        let is_proof_transition = next.is_valid - next.is_proof_start;
        let is_proof_end = local.is_valid - is_proof_transition.clone();

        let is_not_dummy = AB::Expr::ONE - local.is_dummy;

        // Round idx starts at 0
        builder
            .when(local.is_proof_start)
            .assert_zero(local.round_idx);

        // Eval idx starts at 0
        builder
            .when(local.is_first_eval)
            .assert_zero(local.eval_idx);
        // Eval idx increments by 1
        builder
            .when(is_transition_eval.clone())
            .assert_eq(next.eval_idx, local.eval_idx + AB::Expr::ONE);
        // Eval idx ends at s_deg if not dummy
        builder
            .when(is_last_eval.clone() * is_not_dummy.clone())
            .assert_eq(local.eval_idx, AB::Expr::from_usize(s_deg));

        ///////////////////////////////////////////////////////////////////////
        // Factorials Constraints
        ///////////////////////////////////////////////////////////////////////

        let d_factorial_inv = AB::Expr::from_prime_subfield(
            <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield::from_usize((1..=s_deg).product())
                .inverse(),
        );
        // Starts at d!
        builder
            .when(local.is_first_eval)
            .assert_eq(local.denom_inv, d_factorial_inv);
        // 1 / (i + 1)!(d - i - 1)!  (i + 1) = 1 / i!(d - i)! * (d - i)
        builder.when(is_transition_eval.clone()).assert_eq(
            next.denom_inv * next.eval_idx,
            local.denom_inv * (AB::Expr::from_usize(s_deg) - local.eval_idx),
        );

        ///////////////////////////////////////////////////////////////////////
        // Prefix/Suffix Product Constraints
        ///////////////////////////////////////////////////////////////////////

        assert_array_eq(
            &mut builder.when(is_transition_eval.clone()),
            next.r,
            local.r,
        );

        // Prefix Product
        // Starts at 1
        assert_one_ext(&mut builder.when(local.is_first_eval), local.prefix_product);
        // p' = p * (r - i)
        assert_array_eq(
            &mut builder.when(is_transition_eval.clone()),
            next.prefix_product,
            ext_field_multiply(
                local.prefix_product,
                ext_field_subtract_scalar(local.r, local.eval_idx),
            ),
        );

        // Suffix Product
        // Ends at 1
        assert_one_ext(
            &mut builder.when(is_last_eval.clone()),
            local.suffix_product,
        );
        // s = s' * (i + 1 - r)
        assert_array_eq(
            &mut builder.when(is_transition_eval.clone()),
            local.suffix_product,
            ext_field_multiply(
                next.suffix_product,
                scalar_subtract_ext_field(local.eval_idx + AB::Expr::ONE, local.r),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Sumcheck evaluation constraints
        ///////////////////////////////////////////////////////////////////////

        // Lagrange coefficient
        assert_array_eq(
            &mut builder.when(local.is_valid),
            local.lagrange_coeff,
            ext_field_multiply_scalar(
                ext_field_multiply(local.prefix_product, local.suffix_product),
                local.denom_inv,
            ),
        );

        // Initialize at first evaluation
        assert_array_eq(
            &mut builder.when(local.is_first_eval),
            local.cur_sum,
            ext_field_multiply(local.eval, local.lagrange_coeff),
        );

        // Cumulative sum
        assert_array_eq(
            &mut builder.when(is_transition_eval.clone()),
            next.cur_sum,
            ext_field_add(
                local.cur_sum,
                ext_field_multiply(next.eval, next.lagrange_coeff),
            ),
        );

        ///////////////////////////////////////////////////////////////////////
        // Transition constraints
        ///////////////////////////////////////////////////////////////////////

        builder
            .when(is_transition_eval.clone())
            .assert_eq(next.tidx, local.tidx + AB::Expr::from_usize(D_EF));

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////

        // Observe evaluations s(1), s(2) etc.
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            next.eval,
            is_transition_eval.clone() * is_not_dummy.clone(),
        );
        // Sample challenge r
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.r,
            is_last_eval.clone() * is_not_dummy.clone(),
        );

        // Receive tidx from univariate sumcheck and send it to stacking module
        self.stacking_module_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleMessage { tidx: local.tidx },
            local.is_proof_start * is_not_dummy.clone(),
        );
        self.stacking_module_bus.send(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx + AB::Expr::from_usize(D_EF),
            },
            is_proof_end.clone() * is_not_dummy.clone(),
        );

        self.claim_bus.receive(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: local.round_idx.into(),
                value: ext_field_add(local.eval, next.eval),
            },
            local.is_first_eval * is_not_dummy.clone(),
        );
        self.claim_bus.send(
            builder,
            local.proof_idx,
            SumcheckClaimMessage {
                round: local.round_idx + AB::Expr::ONE,
                value: local.cur_sum.map(Into::into),
            },
            is_last_eval.clone() * is_not_dummy.clone(),
        );
        self.randomness_bus.send(
            builder,
            local.proof_idx,
            ConstraintSumcheckRandomness {
                idx: local.round_idx + AB::Expr::ONE,
                challenge: local.r.map(|x| x.into()),
            },
            local.is_first_eval * is_not_dummy.clone(),
        );
        // Here idx > 0 and all idx are distinct within one proof_idx
        self.batch_constraint_conductor_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.round_idx + AB::Expr::ONE,
                value: local.r.map(|x| x.into()),
            },
            local.is_first_eval * is_not_dummy,
        );
    }
}
