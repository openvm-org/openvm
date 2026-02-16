use core::borrow::Borrow;
use std::sync::atomic::{AtomicU32, Ordering};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::primitives::bus::{RangeCheckerBus, RangeCheckerBusMessage};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct RangeCheckerCols<T> {
    value: T,
    mult: T,
}

#[derive(Debug)]
pub struct RangeCheckerCpuTraceGenerator<const NUM_BITS: usize> {
    count: Vec<AtomicU32>,
}

impl<const NUM_BITS: usize> Default for RangeCheckerCpuTraceGenerator<NUM_BITS> {
    fn default() -> Self {
        let mut count = Vec::with_capacity(1 << NUM_BITS);
        for _ in 0..(1 << NUM_BITS) {
            count.push(AtomicU32::new(0));
        }
        Self { count }
    }
}

impl<const NUM_BITS: usize> RangeCheckerCpuTraceGenerator<NUM_BITS> {
    pub fn add_count(&self, value: usize) {
        self.add_count_mult(value, 1);
    }

    pub fn add_count_mult(&self, value: usize, mult: u32) {
        self.count[value].fetch_add(mult, Ordering::Relaxed);
    }

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let trace = self
            .count
            .iter()
            .enumerate()
            .flat_map(|(value, mult)| {
                [
                    F::from_usize(value),
                    F::from_u32(mult.load(Ordering::Relaxed)),
                ]
            })
            .collect_vec();
        RowMajorMatrix::new(trace, RangeCheckerCols::<u8>::width())
    }
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
