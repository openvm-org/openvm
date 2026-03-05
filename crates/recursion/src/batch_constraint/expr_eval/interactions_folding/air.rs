use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    SubAir,
};
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
        Eq3bBus, Eq3bMessage, ExpressionClaimBus, ExpressionClaimMessage, InteractionsFoldingBus,
        InteractionsFoldingMessage,
    },
    bus::{AirShapeBus, AirShapeBusMessage, AirShapeProperty, TranscriptBus},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{assert_zeros, ext_field_add, ext_field_multiply},
};

#[derive(AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct InteractionsFoldingCols<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub beta_tidx: T,

    pub air_idx: T,
    pub sort_idx: T,
    pub interaction_idx: T,
    pub node_idx: T,

    pub has_interactions: T,

    pub is_first_in_air: T,
    /// It's true for the num row, which doesn't need to be beta folded.
    pub is_first_in_message: T, // aka "is_mult"
    // the second in message is the first denom, and it's cur_sum is the folded denom
    pub is_second_in_message: T,
    pub is_bus_index: T,

    pub idx_in_message: T,
    pub value: [T; D_EF],
    /// Current sum for doing beta folding. This is the value for one interaction.
    /// When local.is_first_in_message, next.cur_sum should be the folded denom.
    /// (because local row is for the num row, which doesn't need to be beta folded)
    /// It doesn't multiply with eq_3b yet.
    pub cur_sum: [T; D_EF],
    pub beta: [T; D_EF],
    pub eq_3b: [T; D_EF],

    /// The summed num and denom for all interactions.
    /// It's summed over all the interactions in the AIR: cur_sum * eq_3b when is_first_in_message
    pub final_acc_num: [T; D_EF],
    pub final_acc_denom: [T; D_EF],
}

pub struct InteractionsFoldingAir {
    pub interaction_bus: InteractionsFoldingBus,
    pub air_shape_bus: AirShapeBus,
    pub transcript_bus: TranscriptBus,
    pub expression_claim_bus: ExpressionClaimBus,
    pub eq_3b_bus: Eq3bBus,
}

impl<F> BaseAirWithPublicValues<F> for InteractionsFoldingAir {}
impl<F> PartitionedBaseAir<F> for InteractionsFoldingAir {}

impl<F> BaseAir<F> for InteractionsFoldingAir {
    fn width(&self) -> usize {
        InteractionsFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for InteractionsFoldingAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &InteractionsFoldingCols<AB::Var> = (*local).borrow();
        let next: &InteractionsFoldingCols<AB::Var> = (*next).borrow();

        type LoopSubAir = NestedForLoopSubAir<3>;
        LoopSubAir {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid,
                    counter: [local.proof_idx, local.sort_idx, local.interaction_idx],
                    is_first: [
                        local.is_first,
                        local.is_first_in_air,
                        local.is_first_in_message,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx, next.sort_idx, next.interaction_idx],
                    is_first: [
                        next.is_first,
                        next.is_first_in_air,
                        next.is_first_in_message,
                    ],
                }
                .map_into(),
            ),
        );

        builder.assert_bool(local.has_interactions);
        builder.assert_bool(local.is_bus_index);
        builder
            .when(local.is_bus_index)
            .assert_one(local.has_interactions);
        builder
            .when(local.has_interactions + local.is_bus_index)
            .assert_one(local.is_valid);
        let is_same_proof = next.is_valid - next.is_first;
        let is_same_air = next.is_valid - next.is_first_in_air;
        let is_same_message = next.is_valid - next.is_first_in_message;
        let next_is_first_in_air_or_invalid =
            next.is_first_in_air + (AB::Expr::ONE - next.is_valid);
        let next_is_first_in_message_or_invalid =
            next.is_first_in_message + (AB::Expr::ONE - next.is_valid);

        // =========================== indices consistency ===============================
        // When we are within one proof, sort_idx increases by 0/1
        builder
            .when(is_same_proof.clone())
            .assert_bool(next.sort_idx - local.sort_idx);
        // When we are within one AIR, interaction_idx increases by 0/1 as well
        builder
            .when(is_same_air.clone())
            .assert_bool(next.interaction_idx - local.interaction_idx);
        // First AIR within a proof is zero, and first interaction within an AIR is also zero
        builder.when(local.is_first).assert_zero(local.sort_idx);
        builder
            .when(not::<AB::Expr>(is_same_air.clone()))
            .assert_zero(next.interaction_idx);

        builder
            .when(local.is_first_in_message + local.is_second_in_message)
            .assert_zero(local.idx_in_message);
        builder
            .when(is_same_message.clone())
            .assert_one(next.idx_in_message - local.idx_in_message + local.is_first_in_message);

        // // =========================== general consistency ================================
        // The row describes an AIR without interactions iff it's first and last in the message,
        // unless the row is invalid
        builder.when(local.is_valid).assert_eq(
            local.is_first_in_message * next_is_first_in_message_or_invalid.clone(),
            not(local.has_interactions),
        );
        // If we have interactions, then the row is valid
        builder
            .when(local.has_interactions)
            .assert_one(local.is_valid);
        // If we don't have interactions and the row is valid, then it's first and last _within AIR_
        builder
            .when(not(local.has_interactions))
            .when(local.is_valid)
            .assert_one(local.is_first_in_air);
        builder
            .when(not(local.has_interactions))
            .when(local.is_valid)
            .assert_one(next_is_first_in_air_or_invalid.clone());
        // If the row is valid, then this is the bus index iff the next one is first in message
        // or invalid
        builder.when(local.has_interactions).assert_eq(
            local.is_bus_index,
            next_is_first_in_message_or_invalid.clone(),
        );
        // An interaction has at least two fields (mult and bus index)
        builder
            .when(local.has_interactions)
            .assert_bool(local.is_bus_index + local.is_first_in_message);
        // final_acc_num only changes when it's first in message
        assert_array_eq(
            &mut builder
                .when(not(local.is_first_in_message) * local.is_valid * is_same_air.clone()),
            local.final_acc_num,
            next.final_acc_num,
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_message * local.has_interactions),
            local.final_acc_num,
            ext_field_add(
                next.final_acc_num,
                ext_field_multiply(local.cur_sum, local.eq_3b),
            ),
        );
        assert_zeros(
            &mut builder
                .when(local.is_first_in_message * (local.is_valid - local.has_interactions)),
            local.final_acc_num,
        );
        // final_acc_denom only changes when it's second in message
        assert_array_eq(
            &mut builder.when(
                (not(local.is_second_in_message) + not(local.has_interactions))
                    * local.is_valid
                    * is_same_air.clone(),
            ),
            local.final_acc_denom,
            next.final_acc_denom,
        );
        assert_array_eq(
            &mut builder.when(local.is_second_in_message * local.is_valid),
            local.final_acc_denom,
            ext_field_add(
                next.final_acc_denom,
                ext_field_multiply(local.cur_sum, local.eq_3b),
            ),
        );
        assert_array_eq(
            &mut builder.when(is_same_message.clone()),
            local.eq_3b,
            next.eq_3b,
        );
        // the running sums are zero on the last row of the proof
        assert_zeros(
            &mut builder.when(LoopSubAir::local_is_last(
                local.is_valid,
                next.is_valid,
                next.is_first_in_air,
            )),
            local.final_acc_num,
        );
        assert_zeros(
            &mut builder.when(LoopSubAir::local_is_last(
                local.is_valid,
                next.is_valid,
                next.is_first_in_air,
            )),
            local.final_acc_denom,
        );
        // Constraint is_second_in_message
        builder.assert_bool(local.is_second_in_message);
        builder
            .when(local.is_first_in_message * local.has_interactions)
            .assert_one(next.is_second_in_message);
        builder
            .when(next.is_second_in_message)
            .assert_one(local.is_first_in_message);

        // ======================== beta and cur sum consistency ============================
        assert_array_eq(&mut builder.when(is_same_proof), local.beta, next.beta);
        assert_array_eq(
            &mut builder.when(is_same_message * not(local.is_first_in_message)),
            local.cur_sum,
            ext_field_add(
                local.value,
                ext_field_multiply::<AB::Expr>(local.beta, next.cur_sum),
            ),
        );
        // numerator and the last element of the message are just the corresponding values
        assert_array_eq(
            &mut builder.when(next_is_first_in_message_or_invalid + local.is_first_in_message),
            local.cur_sum,
            local.value,
        );

        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ONE,
                idx: local.sort_idx * AB::Expr::TWO,
                // value: ext_field_multiply(local.cur_sum, local.eq_3b),
                value: local.final_acc_num.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.expression_claim_bus.send(
            builder,
            local.proof_idx,
            ExpressionClaimMessage {
                is_interaction: AB::Expr::ONE,
                idx: local.sort_idx * AB::Expr::TWO + AB::Expr::ONE,
                // value: ext_field_multiply(next.cur_sum, next.eq_3b),
                value: local.final_acc_denom.map(Into::into),
            },
            local.is_first_in_air * local.is_valid,
        );
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ZERO,
                idx_in_message: local.idx_in_message.into(),
                value: local.value.map(Into::into),
            },
            local.has_interactions
                * (AB::Expr::ONE - local.is_first_in_message - local.is_bus_index),
        );
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ONE,
                idx_in_message: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            local.is_first_in_message * local.has_interactions,
        );
        self.interaction_bus.receive(
            builder,
            local.proof_idx,
            InteractionsFoldingMessage {
                air_idx: local.air_idx.into(),
                interaction_idx: local.interaction_idx.into(),
                is_mult: AB::Expr::ZERO,
                idx_in_message: AB::Expr::NEG_ONE,
                value: local.value.map(Into::into),
            },
            local.is_bus_index,
        );

        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.beta_tidx,
            local.beta,
            local.is_valid * local.is_first,
        );

        self.air_shape_bus.lookup_key(
            builder,
            local.proof_idx,
            AirShapeBusMessage {
                sort_idx: local.sort_idx.into(),
                property_idx: AirShapeProperty::NumInteractions.to_field(),
                value: (local.interaction_idx + AB::Expr::ONE) * local.has_interactions,
            },
            next_is_first_in_air_or_invalid * local.is_valid,
        );

        self.eq_3b_bus.receive(
            builder,
            local.proof_idx,
            Eq3bMessage {
                sort_idx: local.sort_idx,
                interaction_idx: local.interaction_idx,
                eq_3b: local.eq_3b,
            },
            local.has_interactions * local.is_first_in_message,
        );
    }
}
