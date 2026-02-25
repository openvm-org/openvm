use std::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{and, not},
    SubAir,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField,
};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    stacking::bus::{
        EqBitsInternalBus, EqBitsInternalMessage, EqBitsLookupBus, EqBitsLookupMessage,
        EqRandValuesLookupBus, EqRandValuesLookupMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        assert_one_ext, assert_zeros, ext_field_add_scalar, ext_field_multiply,
        ext_field_multiply_scalar, ext_field_subtract, ext_field_subtract_scalar,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct EqBitsCols<F> {
    // Proof index columns for continuations
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,

    // Multiplicities of internal and external lookups
    pub internal_child_flag: F,
    pub external_mult: F,

    // Parent row's b_value and evaluation
    pub sub_b_value: F,
    pub sub_eval: [F; D_EF],

    // This row's b_value's LSB, number of bits, and u_{n_stack - num_bits + 1}
    pub b_lsb: F,
    pub num_bits: F,
    pub u_val: [F; D_EF],
}

pub struct EqBitsAir {
    // Internal buses
    pub eq_bits_internal_bus: EqBitsInternalBus,
    pub eq_bits_lookup_bus: EqBitsLookupBus,
    pub eq_rand_values_bus: EqRandValuesLookupBus,

    // Other fields
    pub n_stack: usize,
    pub l_skip: usize,
}

impl BaseAirWithPublicValues<F> for EqBitsAir {}
impl PartitionedBaseAir<F> for EqBitsAir {}

impl<F> BaseAir<F> for EqBitsAir {
    fn width(&self) -> usize {
        EqBitsCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EqBitsAir
where
    AB::F: PrimeField32 + TwoAdicField,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let local: &EqBitsCols<AB::Var> = (*local).borrow();
        let next: &EqBitsCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1> {}.eval(
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
        /*
         * Constrain that the root evaluation is correct, i.e. eq_bits([], []) = 1.
         */
        let mut when_first = builder.when(local.is_first);

        when_first.assert_zero(local.sub_b_value);
        when_first.assert_zero(local.num_bits);
        when_first.assert_zero(local.b_lsb);

        assert_zeros(&mut when_first, local.u_val);
        assert_one_ext(&mut when_first, local.sub_eval);

        /*
         * Receive the parent b_value and eq_bits(u[0..k - 1], b[0..k - 1]) eval, as
         * well as the value of u_{n_stack - num_bits + 1}.
         */
        self.eq_rand_values_bus.lookup_key(
            builder,
            local.proof_idx,
            EqRandValuesLookupMessage {
                idx: AB::Expr::from_usize(self.n_stack + 1) - local.num_bits,
                u: local.u_val.map(Into::into),
            },
            and(not(local.is_first), local.is_valid),
        );

        self.eq_bits_internal_bus.receive(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: local.sub_b_value.into(),
                num_bits: local.num_bits.into() - AB::Expr::ONE,
                eval: local.sub_eval.map(Into::into),
                child_lsb: local.b_lsb.into(),
            },
            and(not(local.is_first), local.is_valid),
        );

        let b_value = AB::Expr::TWO * local.sub_b_value + local.b_lsb;
        builder.assert_bool(local.b_lsb);

        /*
         * Compute eq_bits(u, b) and send it to the appropriate internal and external
         * buses. Field internal_child_flag is odd iff it has a child with lsb 0, and
         * is >= 2 iff it has a child with new bit 1. Because internal message field
         * num_bits must be strictly increasing, each (b_value, num_bits) pair must be
         * unique by induction.
         */
        let eval = ext_field_multiply(
            local.sub_eval,
            ext_field_subtract_scalar::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(
                    ext_field_add_scalar::<AB::Expr>(
                        ext_field_multiply_scalar(local.u_val, AB::Expr::TWO * local.b_lsb),
                        AB::Expr::ONE,
                    ),
                    local.u_val,
                ),
                local.b_lsb,
            ),
        );

        let three = AB::F::from_u8(3);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.internal_child_flag);
        builder
            .when_ne(local.internal_child_flag, three)
            .when_ne(local.internal_child_flag, AB::Expr::TWO)
            .assert_bool(local.internal_child_flag);

        // Multiplicity is f(x) = 1/3 * x * (x - 2) * (2x - 5), which is such that
        // f(0), f(2) = 0 and f(1), f(3) = 1.
        self.eq_bits_internal_bus.send(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: b_value.clone(),
                num_bits: local.num_bits.into(),
                eval: eval.clone(),
                child_lsb: AB::Expr::ZERO,
            },
            local.internal_child_flag
                * (local.internal_child_flag - AB::Expr::TWO)
                * (local.internal_child_flag * AB::Expr::TWO - AB::Expr::from_u8(5))
                * three.inverse(),
        );

        // Multiplicity is f(x) = -1/6 * x * (x - 1) * (2x - 7), which is such that
        // f(0), f(1) = 0 and f(2), f(3) = 1.
        self.eq_bits_internal_bus.send(
            builder,
            local.proof_idx,
            EqBitsInternalMessage {
                b_value: b_value.clone(),
                num_bits: local.num_bits.into(),
                eval: eval.clone(),
                child_lsb: AB::Expr::ONE,
            },
            local.internal_child_flag
                * (AB::Expr::ONE - local.internal_child_flag)
                * (local.internal_child_flag * AB::Expr::TWO - AB::Expr::from_u8(7))
                * AB::F::from_u8(6).inverse(),
        );

        self.eq_bits_lookup_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            EqBitsLookupMessage {
                b_value: b_value * AB::Expr::from_usize(1 << self.l_skip),
                num_bits: local.num_bits.into(),
                eval,
            },
            local.is_valid * local.external_mult,
        );
    }
}
