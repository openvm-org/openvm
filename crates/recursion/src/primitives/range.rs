use core::borrow::Borrow;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    prover::types::AirProofRawInput,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::F;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{RangeCheckerBus, RangeCheckerBusMessage};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct RangeCheckerCols<T> {
    value: T,
    mult: T,
}

#[derive(Debug)]
pub struct RangeCheckerAir<const NUM_BITS: usize> {
    pub bus: RangeCheckerBus,
    count: Vec<AtomicU32>,
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

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &RangeCheckerCols<AB::Var> = (*local).borrow();
        let next: &RangeCheckerCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_zero(local.value);
        builder
            .when_transition()
            .assert_eq(local.value + AB::F::ONE, next.value);
        builder.when_last_row().assert_eq(
            local.value,
            AB::F::from_canonical_usize((1 << NUM_BITS) - 1),
        );

        self.bus.add_key_with_lookups(
            builder,
            RangeCheckerBusMessage {
                value: local.value.into(),
                max_bits: AB::Expr::from_canonical_usize(self.max_bits()),
            },
            local.mult,
        );
    }
}

impl<const NUM_BITS: usize> RangeCheckerAir<NUM_BITS> {
    pub fn new(bus: RangeCheckerBus) -> Self {
        let mut count = Vec::with_capacity(1 << NUM_BITS);
        for _ in 0..(1 << NUM_BITS) {
            count.push(AtomicU32::new(0));
        }
        Self { bus, count }
    }

    pub fn add_count(&self, value: usize) {
        self.count[value].fetch_add(1, Ordering::Relaxed);
    }

    pub fn max_bits(&self) -> usize {
        NUM_BITS
    }

    pub fn generate_proof_input(&self) -> AirProofRawInput<F> {
        let trace = self
            .count
            .iter()
            .enumerate()
            .flat_map(|(value, mult)| {
                [
                    F::from_canonical_usize(value),
                    F::from_canonical_u32(mult.load(Ordering::Relaxed)),
                ]
            })
            .collect_vec();
        AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                trace,
                BaseAir::<F>::width(self),
            ))),
            public_values: vec![],
        }
    }
}
