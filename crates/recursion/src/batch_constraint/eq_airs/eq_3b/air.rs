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
        BatchConstraintConductorBus, BatchConstraintConductorMessage,
        BatchConstraintInnerMessageType, Eq3bBus, Eq3bMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Eq3bColumns<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub sort_idx: T,
    pub interaction_idx: T,

    pub n_lift: T,
    pub two_to_the_n_lift: T,
    pub n: T,
    pub hypercube_volume: T, // 2^n
    pub n_at_least_n_lift: T,

    pub has_no_interactions: T,

    pub is_first_in_air: T,
    pub is_first_in_interaction: T,

    pub idx: T,         // stacked_idx >> l_skip, restored bit by bit
    pub running_idx: T, // the current stacked_idx >> l_skip
    pub nth_bit: T,

    pub xi: [T; D_EF],
    pub eq: [T; D_EF],
}

pub struct Eq3bAir {
    pub eq_3b_bus: Eq3bBus,
    pub batch_constraint_conductor_bus: BatchConstraintConductorBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for Eq3bAir {}
impl<F> PartitionedBaseAir<F> for Eq3bAir {}

impl<F> BaseAir<F> for Eq3bAir {
    fn width(&self) -> usize {
        Eq3bColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Eq3bAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &Eq3bColumns<AB::Var> = (*local).borrow();
        let next: &Eq3bColumns<AB::Var> = (*next).borrow();

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
                        local.is_first_in_interaction,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx, next.sort_idx, next.interaction_idx],
                    is_first: [
                        next.is_first,
                        next.is_first_in_air,
                        next.is_first_in_interaction,
                    ],
                }
                .map_into(),
            ),
        );

        builder.assert_bool(local.n_at_least_n_lift);
        builder.assert_bool(local.nth_bit);
        builder.assert_bool(local.has_no_interactions);

        let within_one_air = next.is_valid - next.is_first_in_air;
        let within_one_interaction = next.is_valid - next.is_first_in_interaction;
        let is_last_in_interaction = local.is_valid - within_one_interaction.clone();

        // =============================== n consistency ==================================
        builder
            .when(local.is_first_in_interaction)
            .assert_zero(local.n);
        builder
            .when(local.is_first_in_interaction)
            .when(local.is_valid)
            .assert_one(local.hypercube_volume);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.n_lift, local.n_lift);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.two_to_the_n_lift, local.two_to_the_n_lift);
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.n, local.n + AB::Expr::ONE);
        builder.when(within_one_interaction.clone()).assert_eq(
            next.hypercube_volume,
            local.hypercube_volume * AB::Expr::TWO,
        );
        // n_at_least_n_lift is nondecreasing
        builder
            .when(within_one_interaction.clone())
            .when(local.n_at_least_n_lift)
            .assert_one(next.n_at_least_n_lift);
        // it's always 1 in the end
        builder
            .when(not(local.has_no_interactions))
            .when(is_last_in_interaction.clone())
            .when(local.is_valid)
            .assert_one(local.n_at_least_n_lift);

        // Either there is a moment where it switches from 0 to 1, then it's when n = n_lift
        builder
            .when(not(local.n_at_least_n_lift))
            .when(next.n_at_least_n_lift)
            .assert_eq(next.n, next.n_lift);
        builder
            .when(not(local.n_at_least_n_lift))
            .when(next.n_at_least_n_lift)
            .assert_eq(next.hypercube_volume, next.two_to_the_n_lift);
        // Or it's 1 from the beginning, in which case n_lift = 0
        builder
            .when(local.is_first_in_interaction)
            .when(local.n_at_least_n_lift)
            .assert_zero(local.n_lift);
        builder
            .when(local.is_first_in_interaction)
            .when(local.n_at_least_n_lift)
            .assert_one(local.two_to_the_n_lift);

        builder.when(within_one_air.clone()).assert_eq(
            next.running_idx,
            local.running_idx + next.is_first_in_interaction * local.two_to_the_n_lift,
        );

        // =========================== Xi and product consistency =============================
        // Boundary conditions
        assert_array_eq(
            &mut builder.when(local.is_valid * local.is_first),
            local.eq,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        builder
            .when(local.is_first_in_interaction)
            .assert_zero(local.idx);
        builder.when(local.is_first).assert_zero(local.running_idx);
        builder
            .when(is_last_in_interaction.clone())
            .when(not(local.has_no_interactions))
            .assert_eq(local.idx, local.running_idx);
        builder
            .when(next.is_valid)
            .when(local.has_no_interactions)
            .assert_eq(local.running_idx, next.running_idx);

        // If n is less than n_lift, assert that eq doesn't change
        assert_array_eq(
            &mut builder
                .when(local.is_valid - local.has_no_interactions)
                .when(not(local.n_at_least_n_lift)),
            local.eq,
            next.eq,
        );
        // Within transition, idx increases by nth_bit * hypercube_volume
        builder
            .when(within_one_interaction.clone())
            .assert_eq(next.idx, local.idx + local.nth_bit * local.hypercube_volume);
        // It can't increase if n < n_lift
        builder
            .when(not(local.n_at_least_n_lift))
            .assert_zero(local.nth_bit);
        // When transition, eq multiplies correspondingly
        assert_array_eq(
            &mut builder.when(within_one_interaction.clone()),
            next.eq,
            ext_field_multiply(
                local.eq,
                ext_field_add::<AB::Expr>(
                    ext_field_multiply_scalar(local.xi, local.nth_bit),
                    ext_field_multiply_scalar::<AB::Expr>(
                        ext_field_one_minus(local.xi),
                        AB::Expr::ONE - local.nth_bit,
                    ),
                ),
            ),
        );

        // ==================== no interactions consistency ==========================
        builder
            .when(local.has_no_interactions)
            .assert_one(local.is_valid);
        builder
            .when(local.has_no_interactions)
            .assert_one(local.is_first_in_air);
        builder
            .when(local.has_no_interactions)
            .assert_zero(within_one_air);

        self.batch_constraint_conductor_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_usize(self.l_skip),
                value: local.xi.map(|x| x.into()),
            },
            local.n_at_least_n_lift * within_one_interaction,
        );

        // The air with this sort_idx has that n_lift, because it's constrained by some
        // HyperdimBusMessages between some other AIRs

        // This pidx has that n_logup because of some GkrModuleMessage between some other AIRs

        self.eq_3b_bus.send(
            builder,
            local.proof_idx,
            Eq3bMessage {
                sort_idx: local.sort_idx,
                interaction_idx: local.interaction_idx,
                eq_3b: local.eq,
            },
            is_last_in_interaction * (local.is_valid - local.has_no_interactions),
        );
    }
}
