use std::sync::atomic::{AtomicU32, Ordering};

use itertools::Itertools;
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::primitives::range::air::RangeCheckerCols;

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
