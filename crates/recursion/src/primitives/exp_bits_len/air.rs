use core::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::primitives::{
    bus::{ExpBitsLenBus, ExpBitsLenMessage},
    exp_bits_len::trace::{LOW_BITS_COUNT, NUM_BITS_MAX_PLUS_ONE},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ExpBitsLenCols<T> {
    /// Marks rows that belong to an `ExpBitsLen` request rather than trailing padding.
    pub is_valid: T,
    /// Marks the first row of a 32-row request block. Only these rows publish on the lookup bus.
    pub is_first: T,
    /// Bit position carried by this row. Rows `0..30` decompose bits, row `31` is terminal.
    pub bit_idx: T,
    /// `base^(2^bit_idx)`.
    pub base: T,
    /// Remaining suffix of the canonical 31-bit decomposition after shifting by `bit_idx`.
    pub bit_src: T,
    /// Remaining number of low bits that still affect `result`.
    pub num_bits: T,
    /// Boolean witness for `num_bits != 0`.
    pub apply_bit: T,
    /// Countdown for the low 27 bits used in the BabyBear `< p` canonicality check.
    pub low_bits_left: T,
    /// Boolean witness for `low_bits_left != 0`.
    pub in_low_region: T,
    /// Running product for the requested low-bit exponentiation.
    pub result: T,
    /// Multiplies `result` by either `1` or `base` on the next transition.
    pub result_multiplier: T,
    /// Current decomposition bit.
    pub bit_src_mod_2: T,
    /// Running flag: all low bits `b0..b26` seen so far are zero.
    pub low_bits_are_zero: T,
    /// Running flag: all high bits `b27..b30` seen so far are one.
    pub high_bits_all_one: T,
}

#[derive(Debug)]
pub struct ExpBitsLenAir {
    pub exp_bits_len_bus: ExpBitsLenBus,
}

impl ExpBitsLenAir {
    pub fn new(exp_bits_len_bus: ExpBitsLenBus) -> Self {
        Self { exp_bits_len_bus }
    }
}

fn assert_babybear_field<F: PrimeField32>() {
    assert_eq!(
        F::ORDER_U32,
        BabyBear::ORDER_U32,
        "ExpBitsLenAir is hard-coded for the BabyBear modulus; canonicality constraints assume p = 15 * 2^27 + 1",
    );
}

impl<F: PrimeField32> BaseAirWithPublicValues<F> for ExpBitsLenAir {}
impl<F: PrimeField32> PartitionedBaseAir<F> for ExpBitsLenAir {}

impl<F: PrimeField32> BaseAir<F> for ExpBitsLenAir {
    fn width(&self) -> usize {
        assert_babybear_field::<F>();
        ExpBitsLenCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        assert_babybear_field::<AB::F>();
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &ExpBitsLenCols<AB::Var> = (*local).borrow();
        let next: &ExpBitsLenCols<AB::Var> = (*next).borrow();

        let is_transition = next.is_valid.into() - next.is_first.into();
        let local_is_last = local.is_valid.into() - is_transition.clone();
        let local_in_high_region =
            local.is_valid.into() - local.in_low_region.into() - local_is_last.clone();
        let next_is_not_low_region = next.is_valid.into() - next.in_low_region.into();

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);
        builder.assert_bool(is_transition.clone());
        builder
            .when(is_transition.clone())
            .assert_one(local.is_valid);
        builder.assert_bool(local.apply_bit);
        builder.assert_bool(local.in_low_region);
        builder.assert_bool(local.bit_src_mod_2);
        builder.assert_bool(local.low_bits_are_zero);
        builder.assert_bool(local.high_bits_all_one);
        builder.assert_bool(local_in_high_region.clone());
        builder.when(local.is_first).assert_one(local.is_valid);
        builder.when(local.num_bits).assert_one(local.apply_bit);
        builder
            .when(local.low_bits_left)
            .assert_one(local.in_low_region);

        builder.assert_eq(
            local.result_multiplier - AB::Expr::ONE,
            local.apply_bit * local.bit_src_mod_2 * (local.base - AB::Expr::ONE),
        );

        builder
            .when_first_row()
            .assert_eq(local.is_valid, local.is_first);
        builder.when(local.is_first).assert_one(local.in_low_region);
        builder
            .when(local.is_first)
            .assert_zero(local_is_last.clone());
        builder.when(local.is_first).assert_zero(local.bit_idx);
        builder
            .when(local.is_first)
            .assert_eq(local.low_bits_left, AB::Expr::from_usize(LOW_BITS_COUNT));
        builder
            .when(local.is_first)
            .assert_one(local.low_bits_are_zero);
        builder
            .when(local.is_first)
            .assert_zero(local.high_bits_all_one);

        builder
            .when(local_is_last.clone())
            .assert_zero(local.in_low_region);
        builder.when(local_is_last.clone()).assert_eq(
            local.bit_idx,
            AB::Expr::from_usize(NUM_BITS_MAX_PLUS_ONE - 1),
        );
        builder
            .when(local_is_last.clone())
            .assert_zero(local.bit_src);
        builder
            .when(local_is_last.clone())
            .assert_zero(local.num_bits);
        builder
            .when(local_is_last.clone())
            .assert_zero(local.apply_bit);
        builder
            .when(local_is_last.clone())
            .assert_zero(local.low_bits_left);
        builder.when(local_is_last.clone()).assert_one(local.result);
        builder
            .when(local_is_last.clone())
            .assert_one(local.result_multiplier);
        builder
            .when(local_is_last.clone())
            .assert_zero(local.high_bits_all_one * (AB::Expr::ONE - local.low_bits_are_zero));

        builder
            .when(is_transition.clone())
            .assert_eq(next.bit_idx, local.bit_idx + AB::Expr::ONE);
        builder
            .when(is_transition.clone())
            .assert_eq(next.base, local.base * local.base);
        builder.when(is_transition.clone()).assert_eq(
            local.bit_src,
            next.bit_src * AB::Expr::TWO + local.bit_src_mod_2,
        );
        builder
            .when(is_transition.clone())
            .assert_eq(next.num_bits, local.num_bits - local.apply_bit);
        builder.when(is_transition.clone()).assert_eq(
            next.low_bits_left,
            local.low_bits_left - local.in_low_region,
        );
        builder
            .when(is_transition.clone())
            .assert_eq(local.result, next.result * local.result_multiplier);
        builder.when(local.in_low_region).assert_eq(
            next.low_bits_are_zero,
            local.low_bits_are_zero * (AB::Expr::ONE - local.bit_src_mod_2),
        );
        builder
            .when(local_in_high_region.clone())
            .assert_eq(next.low_bits_are_zero, local.low_bits_are_zero);
        builder
            .when(local.in_low_region * next.in_low_region)
            .assert_zero(next.high_bits_all_one);
        builder
            .when(local.in_low_region * next_is_not_low_region.clone())
            .assert_one(next.high_bits_all_one);
        builder.when(local_in_high_region).assert_eq(
            next.high_bits_all_one,
            local.high_bits_all_one * local.bit_src_mod_2,
        );

        self.exp_bits_len_bus.add_key_with_lookups(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.is_first,
        );
    }
}
