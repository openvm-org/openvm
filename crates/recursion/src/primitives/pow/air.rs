use core::borrow::Borrow;

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_util::log2_strict_usize, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::primitives::bus::{
    PowerCheckerBus, PowerCheckerBusMessage, RangeCheckerBus, RangeCheckerBusMessage,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct PowerCheckerCols<T> {
    pub log: T,
    pub pow: T,
    pub mult_pow: T,
    pub mult_range: T,
}

#[derive(Debug)]
pub struct PowerCheckerAir<const BASE: usize, const N: usize> {
    pub pow_bus: PowerCheckerBus,
    pub range_bus: RangeCheckerBus,
}

impl<F, const BASE: usize, const N: usize> BaseAir<F> for PowerCheckerAir<BASE, N> {
    fn width(&self) -> usize {
        PowerCheckerCols::<F>::width()
    }
}
impl<F, const B: usize, const N: usize> BaseAirWithPublicValues<F> for PowerCheckerAir<B, N> {}
impl<F, const B: usize, const N: usize> PartitionedBaseAir<F> for PowerCheckerAir<B, N> {}

impl<AB: AirBuilder + InteractionBuilder, const BASE: usize, const N: usize> Air<AB>
    for PowerCheckerAir<BASE, N>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &PowerCheckerCols<AB::Var> = (*local).borrow();
        let next: &PowerCheckerCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_zero(local.log);
        builder.when_first_row().assert_eq(local.pow, AB::F::ONE);

        builder
            .when_transition()
            .assert_eq(local.log + AB::F::ONE, next.log);
        builder
            .when_transition()
            .assert_eq(local.pow * AB::F::from_usize(BASE), next.pow);

        builder
            .when_last_row()
            .assert_eq(local.log, AB::F::from_usize(N - 1));

        self.pow_bus.add_key_with_lookups(
            builder,
            PowerCheckerBusMessage {
                log: local.log,
                exp: local.pow,
            },
            local.mult_pow,
        );
        self.range_bus.add_key_with_lookups(
            builder,
            RangeCheckerBusMessage {
                value: local.log.into(),
                max_bits: AB::Expr::from_usize(log2_strict_usize(N)),
            },
            local.mult_range,
        );
    }
}
