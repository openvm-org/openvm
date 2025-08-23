//! Range check a tuple simultaneously.
//! When you know you want to range check `(x, y)` to `x_bits, y_bits` respectively
//! and `2^{x_bits + y_bits} < ~2^20`, then you can use this chip to do the range check in one
//! interaction versus the two interactions necessary if you were to use
//! [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead.

use std::{
    mem::size_of,
    sync::{atomic::AtomicU32, Arc},
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir, PairBuilder},
    p3_field::{Field, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};

mod bus;
pub use bus::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
pub mod tests;

#[repr(C)]
#[derive(Default, Copy, Clone, AlignedBorrow)]
pub struct RangeTupleCols<T> {
    /// Number of range checks requested for each tuple combination
    pub mult: T,
}

#[derive(Default, Clone)]
pub struct RangeTuplePreprocessedCols<T> {
    /// Contains all possible tuple combinations within specified ranges
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
    pub count: Vec<Arc<AtomicU32>>,
}

pub type SharedRangeTupleCheckerChip<const N: usize> = Arc<RangeTupleCheckerChip<N>>;

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
            .map(|c| F::from_canonical_u32(c.swap(0, std::sync::atomic::Ordering::Relaxed)))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(rows, 1)
    }
}

impl<R, SC: StarkGenericConfig, const N: usize> Chip<R, CpuBackend<SC>> for RangeTupleCheckerChip<N>
where
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: R) -> AirProvingContext<CpuBackend<SC>> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProvingContext::simple_no_pis(Arc::new(trace))
    }
}

impl<const N: usize> ChipUsageGetter for RangeTupleCheckerChip<N> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.count.len())
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }
    fn trace_width(&self) -> usize {
        NUM_RANGE_TUPLE_COLS
    }
}
