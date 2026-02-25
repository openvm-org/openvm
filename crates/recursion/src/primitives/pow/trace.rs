use std::sync::atomic::{AtomicU32, Ordering};

use itertools::Itertools;
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::primitives::pow::air::PowerCheckerCols;

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

    pub fn take_counts(&self) -> (Vec<u32>, Vec<u32>) {
        let pow = self
            .count_pow
            .iter()
            .map(|counter| counter.swap(0, Ordering::Relaxed))
            .collect();
        let range = self
            .count_range
            .iter()
            .map(|counter| counter.swap(0, Ordering::Relaxed))
            .collect();
        (pow, range)
    }

    pub fn reset(&self) {
        for counter in &self.count_pow {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in &self.count_range {
            counter.store(0, Ordering::Relaxed);
        }
    }

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let mut current_pow = F::ONE;
        let trace = self
            .count_pow
            .iter()
            .zip(self.count_range.iter())
            .enumerate()
            .flat_map(|(log, (mult_pow, mult_range))| {
                let ret = [
                    F::from_usize(log),
                    current_pow,
                    F::from_u32(mult_pow.load(Ordering::Relaxed)),
                    F::from_u32(mult_range.load(Ordering::Relaxed)),
                ];
                current_pow *= F::from_usize(BASE);
                ret
            })
            .collect_vec();
        RowMajorMatrix::new(trace, PowerCheckerCols::<u8>::width())
    }
}
