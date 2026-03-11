use core::borrow::Borrow;

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::primitives::bus::{RangeCheckerBus, RangeCheckerBusMessage};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RangeCheckerCols<T> {
    pub value: T,
    pub mult: T,
}

#[derive(Debug)]
pub struct RangeCheckerAir<const NUM_BITS: usize> {
    pub bus: RangeCheckerBus,
}

impl<F, const NUM_BITS: usize> BaseAir<F> for RangeCheckerAir<NUM_BITS> {
    fn width(&self) -> usize {
        RangeCheckerCols::<F>::width()
    }
}
impl<F, const NUM_BITS: usize> BaseAirWithPublicValues<F> for RangeCheckerAir<NUM_BITS> {}
impl<F, const NUM_BITS: usize> PartitionedBaseAir<F> for RangeCheckerAir<NUM_BITS> {}

impl<AB: AirBuilder + InteractionBuilder, const NUM_BITS: usize> Air<AB>
    for RangeCheckerAir<NUM_BITS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &RangeCheckerCols<AB::Var> = (*local).borrow();
        let next: &RangeCheckerCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_zero(local.value);
        builder
            .when_transition()
            .assert_eq(local.value + AB::F::ONE, next.value);
        builder
            .when_last_row()
            .assert_eq(local.value, AB::F::from_usize((1 << NUM_BITS) - 1));

        self.bus.add_key_with_lookups(
            builder,
            RangeCheckerBusMessage {
                value: local.value.into(),
                max_bits: AB::Expr::from_usize(NUM_BITS),
            },
            local.mult,
        );
    }
}
