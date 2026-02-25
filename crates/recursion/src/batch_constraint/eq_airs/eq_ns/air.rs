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
        BatchConstraintInnerMessageType, EqNOuterBus, EqNOuterMessage, EqZeroNBus, EqZeroNMessage,
    },
    bus::{SelHypercubeBus, SelHypercubeBusMessage, XiRandomnessBus, XiRandomnessMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        base_to_ext, ext_field_add, ext_field_multiply, ext_field_multiply_scalar,
        ext_field_one_minus, ext_field_subtract,
    },
};

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct EqNsColumns<T> {
    pub is_valid: T,
    pub is_first: T,
    pub proof_idx: T,

    pub n: T,
    pub n_less_than_n_logup: T,
    pub n_less_than_n_max: T,
    pub is_transition_and_n_less_than_n_max: T,
    pub xi_n: [T; D_EF],
    pub r_n: [T; D_EF],
    pub r_product: [T; D_EF],
    pub r_pref_product: [T; D_EF],
    pub one_minus_r_pref_prod: [T; D_EF],
    pub eq: [T; D_EF],
    pub eq_sharp: [T; D_EF],

    /// The number of traces whose `n_lift` equals `local.n`.
    /// Note that it cannot be derived from `xi_mult` because
    /// `xi_mult` counts interactions, not AIRs with interactions.
    pub num_traces: T,
    pub xi_mult: T,
    pub sel_first_count: T,
    pub sel_last_and_trans_count: T,
}

pub struct EqNsAir {
    pub zero_n_bus: EqZeroNBus,
    pub xi_bus: XiRandomnessBus,
    pub r_xi_bus: BatchConstraintConductorBus,
    pub sel_hypercube_bus: SelHypercubeBus,
    pub eq_n_outer_bus: EqNOuterBus,

    pub l_skip: usize,
}

impl<F> BaseAirWithPublicValues<F> for EqNsAir {}
impl<F> PartitionedBaseAir<F> for EqNsAir {}

impl<F> BaseAir<F> for EqNsAir {
    fn width(&self) -> usize {
        EqNsColumns::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqNsAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqNsColumns<AB::Var> = (*local).borrow();
        let next: &EqNsColumns<AB::Var> = (*next).borrow();

        // Summary:
        // - n consistency: treat `n_less_than_n_logup` as boolean, set `n = 0` on the first row,
        //   increment `n` while continuing within the proof, propagate the “less than n_logup” flag
        //   forward, and clear it whenever a row is invalid.
        // - r consistency: initialize `r_product` to one when the next row begins the proof and,
        //   otherwise, multiply by the current `r_n` values to update the running product.
        // - eq consistency: update both `eq` and `eq_sharp` by multiplying with the shared factor
        //   `1 - (xi + r - 2 * xi * r)` whenever advancing beyond the first row.

        type LoopSubAir = NestedForLoopSubAir<1>;
        LoopSubAir {}.eval(
            builder,
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
        );

        let is_transition = next.is_valid - next.is_first;
        let is_last = local.is_valid - is_transition.clone();

        // ========================= n consistency ==============================
        builder.assert_bool(local.n_less_than_n_max);
        builder.assert_bool(local.n_less_than_n_logup);
        builder.when(local.is_first).assert_zero(local.n);
        builder
            .when(is_transition.clone())
            .assert_one(next.n - local.n);
        builder
            .when(is_transition.clone())
            .when(next.n_less_than_n_logup)
            .assert_one(local.n_less_than_n_logup);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.n_less_than_n_logup);
        builder
            .when(is_transition.clone())
            .when(next.n_less_than_n_max)
            .assert_one(local.n_less_than_n_max);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.n_less_than_n_max);
        // ========================= r consistency ==============================
        assert_array_eq(
            &mut builder.when(local.is_valid - local.n_less_than_n_max),
            local.r_n,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_last.clone()),
            local.r_product,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone()),
            local.r_product,
            ext_field_multiply(next.r_product, local.r_n),
        );
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.r_pref_product,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone() * local.n_less_than_n_max),
            next.r_pref_product,
            ext_field_multiply(local.r_pref_product, local.r_n),
        );
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.one_minus_r_pref_prod,
            base_to_ext::<AB::Expr>(AB::Expr::ONE),
        );
        assert_array_eq(
            &mut builder.when(is_transition.clone() * local.n_less_than_n_max),
            next.one_minus_r_pref_prod,
            ext_field_multiply(
                local.one_minus_r_pref_prod,
                ext_field_subtract(base_to_ext::<AB::Expr>(AB::F::ONE), local.r_n),
            ),
        );
        self.sel_hypercube_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelHypercubeBusMessage {
                n: local.n.into(),
                is_first: AB::Expr::ZERO,
                value: local.r_pref_product.map(Into::into),
            },
            local.is_valid * local.sel_last_and_trans_count,
        );
        self.sel_hypercube_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            SelHypercubeBusMessage {
                n: local.n.into(),
                is_first: AB::Expr::ONE,
                value: local.one_minus_r_pref_prod.map(Into::into),
            },
            local.is_valid * local.sel_first_count,
        );
        // ========================= eq consistency ===============================
        let mult = ext_field_one_minus::<AB::Expr>(ext_field_subtract::<AB::Expr>(
            ext_field_add(local.xi_n, local.r_n),
            ext_field_multiply_scalar::<AB::Expr>(
                ext_field_multiply(local.xi_n, local.r_n),
                AB::Expr::TWO,
            ),
        ));
        builder.assert_eq(
            local.is_transition_and_n_less_than_n_max,
            is_transition.clone() * local.n_less_than_n_max,
        );
        assert_array_eq(
            &mut builder.when(local.is_transition_and_n_less_than_n_max),
            next.eq,
            ext_field_multiply(local.eq, mult.clone()),
        );
        assert_array_eq(
            &mut builder.when(local.is_transition_and_n_less_than_n_max),
            next.eq_sharp,
            ext_field_multiply(local.eq_sharp, mult.clone()),
        );

        self.xi_bus.receive(
            builder,
            local.proof_idx,
            XiRandomnessMessage {
                idx: local.n + AB::Expr::from_usize(self.l_skip),
                xi: local.xi_n.map(|x| x.into()),
            },
            is_transition.clone(),
        );
        // Here idx >= l_skip and all idx are different within one proof_idx
        self.r_xi_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::Xi.to_field(),
                idx: local.n + AB::Expr::from_usize(self.l_skip),
                value: local.xi_n.map(|x| x.into()),
            },
            local.n_less_than_n_logup * local.xi_mult,
        );

        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ZERO,
                value: local.eq.map(|x| x.into()),
            },
            local.is_first,
        );
        self.zero_n_bus.receive(
            builder,
            local.proof_idx,
            EqZeroNMessage {
                is_sharp: AB::Expr::ONE,
                value: local.eq_sharp.map(|x| x.into()),
            },
            local.is_first,
        );

        self.r_xi_bus.lookup_key(
            builder,
            local.proof_idx,
            BatchConstraintConductorMessage {
                msg_type: BatchConstraintInnerMessageType::R.to_field(),
                idx: local.n + AB::Expr::ONE,
                value: local.r_n.map(|x| x.into()),
            },
            local.n_less_than_n_max,
        );

        self.eq_n_outer_bus.add_key_with_lookups(
            builder,
            next.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ZERO,
                n: next.n.into(),
                value: ext_field_multiply(next.eq, next.r_product),
            },
            next.is_valid * next.num_traces,
        );
        self.eq_n_outer_bus.add_key_with_lookups(
            builder,
            next.proof_idx,
            EqNOuterMessage {
                is_sharp: AB::Expr::ONE,
                n: next.n.into(),
                value: ext_field_multiply(next.eq_sharp, next.r_product),
            },
            next.is_valid * next.num_traces * AB::Expr::TWO, // two because num+denom per trace
        );
    }
}
