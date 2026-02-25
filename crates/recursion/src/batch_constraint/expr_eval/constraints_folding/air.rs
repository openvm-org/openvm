use std::borrow::Borrow;

use openvm_circuit_primitives::{utils::assert_array_eq, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    batch_constraint::bus::{
        ConstraintsFoldingBus, ConstraintsFoldingMessage, EqNOuterBus, EqNOuterMessage,
        ExpressionClaimBus, ExpressionClaimMessage,
    },
    bus::{NLiftBus, NLiftMessage, TranscriptBus},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct ConstraintsFoldingCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub air_idx: T,
    pub sort_idx: T,
    pub constraint_idx: T,
    pub n_lift: T,

    pub lambda_tidx: T,
    pub lambda: [T; D_EF],

    pub value: [T; D_EF],
    pub cur_sum: [T; D_EF],
    pub eq_n: [T; D_EF],

    pub is_first_in_air: T,
}

pub struct ConstraintsFoldingAir {
    pub transcript_bus: TranscriptBus,
    pub constraint_bus: ConstraintsFoldingBus,
    pub expression_claim_bus: ExpressionClaimBus,
    pub eq_n_outer_bus: EqNOuterBus,
    pub n_lift_bus: NLiftBus,
}

impl<F> BaseAirWithPublicValues<F> for ConstraintsFoldingAir {}
impl<F> PartitionedBaseAir<F> for ConstraintsFoldingAir {}

impl<F> BaseAir<F> for ConstraintsFoldingAir {
    fn width(&self) -> usize {
        ConstraintsFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ConstraintsFoldingAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &ConstraintsFoldingCols<AB::Var> = (*local).borrow();
        let next: &ConstraintsFoldingCols<AB::Var> = (*next).borrow();

        type LoopSubAir = NestedForLoopSubAir<2>;
        LoopSubAir {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid,
                    counter: [local.proof_idx, local.sort_idx],
                    is_first: [local.is_first, local.is_first_in_air],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx, next.sort_idx],
                    is_first: [next.is_first, next.is_first_in_air],
                }
                .map_into(),
            ),
        );

        let is_same_proof = next.is_valid - next.is_first;
        let is_same_air = next.is_valid - next.is_first_in_air;

        // =========================== indices consistency ===============================
        // When we are within one air, constraint_idx increases by 0/1
        builder
            .when(is_same_air.clone())
            .assert_bool(next.constraint_idx - local.constraint_idx);
        // First constraint_idx within an air is zero
        builder
            .when(local.is_first_in_air)
            .assert_zero(local.constraint_idx);
        builder
            .when(is_same_air.clone())
            .assert_eq(local.n_lift, next.n_lift);

        // ======================== lambda and cur sum consistency ============================
        assert_array_eq(&mut builder.when(is_same_proof), local.lambda, next.lambda);
        assert_array_eq(
            &mut builder.when(is_same_air.clone()),
            local.cur_sum,
            ext_field_add(
                local.value,
                ext_field_multiply::<AB::Expr>(local.lambda, next.cur_sum),
            ),
        );
        assert_array_eq(
            &mut builder.when(is_same_air.clone()),
            local.eq_n,
            next.eq_n,
        );
        // numerator and the last element of the message are just the corresponding values
        assert_array_eq(
            &mut builder.when(AB::Expr::ONE - is_same_air.clone()),
            local.cur_sum,
            local.value,
        );

        self.n_lift_bus.receive(
            builder,
            local.proof_idx,
            NLiftMessage {
                air_idx: local.air_idx,
                n_lift: local.n_lift,
            },
            local.is_first_in_air * local.is_valid,
        );
        self.constraint_bus.receive(
            builder,
            local.proof_idx,
            ConstraintsFoldingMessage {
                air_idx: local.air_idx.into(),
                constraint_idx: local.constraint_idx - AB::Expr::ONE,
                value: local.value.map(Into::into),
            },
            local.is_valid * (AB::Expr::ONE - local.is_first_in_air),
        );
        let folded_sum: [AB::Expr; D_EF] = ext_field_add(
            ext_field_multiply_scalar::<AB::Expr>(next.cur_sum, is_same_air.clone()),
            ext_field_multiply_scalar::<AB::Expr>(local.cur_sum, AB::Expr::ONE - is_same_air),
        );
        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ZERO,
                idx: local.sort_idx.into(),
                value: ext_field_multiply(folded_sum, local.eq_n),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.lambda_tidx,
            local.lambda,
            local.is_valid * local.is_first,
        );

        self.eq_n_outer_bus.lookup_key(
            builder,
            local.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ZERO,
                n: local.n_lift.into(),
                value: local.eq_n.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
    }
}
