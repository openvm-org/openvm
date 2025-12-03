use core::borrow::Borrow;
use std::sync::atomic::{AtomicU32, Ordering};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_util::log2_strict_usize,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::F;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::primitives::bus::{
    PowerCheckerBus, PowerCheckerBusMessage, RangeCheckerBus, RangeCheckerBusMessage,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct PowerCheckerCols<T> {
    log: T,
    pow: T,
    mult_pow: T,
    mult_range: T,
}

#[derive(Debug)]
pub struct PowerCheckerCpuTraceGenerator<const BASE: usize, const N: usize> {
    count_pow: Vec<AtomicU32>,
    count_range: Vec<AtomicU32>,
}

impl<const BASE: usize, const N: usize> Default for PowerCheckerCpuTraceGenerator<BASE, N> {
    fn default() -> Self {
        assert!(N.is_power_of_two());
        let mut count_pow = Vec::with_capacity(N);
        let mut count_range = Vec::with_capacity(N);
        for _ in 0..N {
            count_pow.push(AtomicU32::new(0));
            count_range.push(AtomicU32::new(0));
        }
        Self {
            count_pow,
            count_range,
        }
    }
}

impl<const BASE: usize, const N: usize> PowerCheckerCpuTraceGenerator<BASE, N> {
    pub fn add_pow(&self, log: usize) -> usize {
        self.count_pow[log].fetch_add(1, Ordering::Relaxed);
        1 << log
    }

    pub fn add_range(&self, value: usize) {
        self.count_range[value].fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_pow_count(&self, log: usize, count: u32) {
        debug_assert!(log < self.count_pow.len());
        if count != 0 {
            self.count_pow[log].fetch_add(count, Ordering::Relaxed);
        }
    }

    pub fn add_range_count(&self, value: usize, count: u32) {
        debug_assert!(value < self.count_range.len());
        if count != 0 {
            self.count_range[value].fetch_add(count, Ordering::Relaxed);
        }
    }

    pub fn reset(&self) {
        for counter in &self.count_pow {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in &self.count_range {
            counter.store(0, Ordering::Relaxed);
        }
    }

    #[tracing::instrument(name = "generate_trace", skip_all)]
    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let mut current_pow = F::ONE;
        let trace = self
            .count_pow
            .iter()
            .zip(self.count_range.iter())
            .enumerate()
            .flat_map(|(log, (mult_pow, mult_range))| {
                let ret = [
                    F::from_canonical_usize(log),
                    current_pow,
                    F::from_canonical_u32(mult_pow.load(Ordering::Relaxed)),
                    F::from_canonical_u32(mult_range.load(Ordering::Relaxed)),
                ];
                current_pow *= F::from_canonical_usize(BASE);
                ret
            })
            .collect_vec();
        RowMajorMatrix::new(trace, PowerCheckerCols::<u8>::width())
    }
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

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &PowerCheckerCols<AB::Var> = (*local).borrow();
        let next: &PowerCheckerCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_zero(local.log);
        builder.when_first_row().assert_eq(local.pow, AB::F::ONE);

        builder
            .when_transition()
            .assert_eq(local.log + AB::F::ONE, next.log);
        builder
            .when_transition()
            .assert_eq(local.pow * AB::F::from_canonical_usize(BASE), next.pow);

        builder
            .when_last_row()
            .assert_eq(local.log, AB::F::from_canonical_usize(N - 1));

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
                max_bits: AB::Expr::from_canonical_usize(log2_strict_usize(N)),
            },
            local.mult_range,
        );
    }
}
