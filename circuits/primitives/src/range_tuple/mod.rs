//! Range check a tuple simultaneously.
//! When you know you want to range check `(x, y)` to `x_bits, y_bits` respectively
//! and `2^{x_bits + y_bits} < ~2^20`, then you can use this chip to do the range check in one interaction
//! versus the two interactions necessary if you were to use [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead.

use std::{
    mem::size_of,
    sync::{atomic::AtomicU32, Arc},
};

use ax_stark_backend::{
    config::StarkGenericConfig,
    interaction::InteractionBuilder,
    p3_uni_stark::Val,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

mod bus;

#[cfg(test)]
pub mod tests;

pub use bus::*;

#[derive(Default, Copy, Clone)]
pub struct RangeTupleCols<T> {
    pub mult: T,
}

#[derive(Default, Clone)]
pub struct RangeTuplePreprocessedCols<T> {
    pub tuple: Vec<T>,
}

pub const NUM_RANGE_TUPLE_COLS: usize = size_of::<RangeTupleCols<u8>>();

#[derive(Clone, Copy, Debug)]
pub struct RangeTupleCheckerAir<const N: usize> {
    pub bus: RangeTupleCheckerBus<N>,
}

impl<const N: usize> RangeTupleCheckerAir<N> {
    pub fn height(&self) -> u32 {
        self.bus.sizes.iter().product()
    }
}
impl<F: Field, const N: usize> BaseAirWithPublicValues<F> for RangeTupleCheckerAir<N> {}
impl<F: Field, const N: usize> PartitionedBaseAir<F> for RangeTupleCheckerAir<N> {}

impl<F: Field, const N: usize> BaseAir<F> for RangeTupleCheckerAir<N> {
    fn width(&self) -> usize {
        NUM_RANGE_TUPLE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut unrolled_matrix = Vec::with_capacity((self.height() as usize) * N);
        let mut row = [0u32; N];
        for _ in 0..self.height() {
            unrolled_matrix.extend(row);
            for i in (0..N).rev() {
                if row[i] < self.bus.sizes[i] - 1 {
                    row[i] += 1;
                    break;
                }
                row[i] = 0;
            }
        }
        Some(RowMajorMatrix::new(
            unrolled_matrix
                .iter()
                .map(|&v| F::from_canonical_u32(v))
                .collect(),
            N,
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder, const N: usize> Air<AB> for RangeTupleCheckerAir<N> {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local = RangeTuplePreprocessedCols {
            tuple: (*prep_local).to_vec(),
        };
        let main = builder.main();
        let local = main.row_slice(0);
        let local = RangeTupleCols { mult: (*local)[0] };

        self.bus.receive(prep_local.tuple).eval(builder, local.mult);
    }
}

#[derive(Debug)]
pub struct RangeTupleCheckerChip<const N: usize> {
    pub air: RangeTupleCheckerAir<N>,
    count: Vec<Arc<AtomicU32>>,
}

impl<const N: usize> RangeTupleCheckerChip<N> {
    pub fn new(bus: RangeTupleCheckerBus<N>) -> Self {
        let range_max = bus.sizes.iter().product();
        let count = (0..range_max)
            .map(|_| Arc::new(AtomicU32::new(0)))
            .collect();

        Self {
            air: RangeTupleCheckerAir { bus },
            count,
        }
    }

    pub fn bus(&self) -> &RangeTupleCheckerBus<N> {
        &self.air.bus
    }

    pub fn sizes(&self) -> &[u32; N] {
        &self.air.bus.sizes
    }

    pub fn add_count(&self, ids: &[u32]) {
        let index = ids
            .iter()
            .zip(self.air.bus.sizes.iter())
            .fold(0, |acc, (id, sz)| acc * sz + id) as usize;
        assert!(
            index < self.count.len(),
            "range exceeded: {} >= {}",
            index,
            self.count.len()
        );
        let val_atomic = &self.count[index];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for val in &self.count {
            val.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .count
            .iter()
            .map(|c| F::from_canonical_u32(c.load(std::sync::atomic::Ordering::SeqCst)))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(rows, 1)
    }
}

impl<SC: StarkGenericConfig, const N: usize> Chip<SC> for RangeTupleCheckerChip<N>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProofInput::simple_no_pis(Arc::new(self.air), trace)
    }
}

impl<const N: usize> ChipUsageGetter for RangeTupleCheckerChip<N> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }
    fn trace_width(&self) -> usize {
        NUM_RANGE_TUPLE_COLS
    }
}
