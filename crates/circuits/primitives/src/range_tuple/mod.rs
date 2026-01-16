//! Range check a tuple simultaneously.
//! When you know you want to range check `(x, y)` to `x_bits, y_bits` respectively
//! and `2^{x_bits + y_bits} < ~2^20`, then you can use this chip to do the range check in one
//! interaction versus the two interactions necessary if you were to use
//! [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead.

use std::{
    borrow::Borrow,
    sync::{atomic::AtomicU32, Arc},
};

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{Field, FieldAlgebra, PrimeField32},
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

#[derive(Copy, Clone)]
pub struct RangeTupleCols<'a, T> {
    /// Contains all possible tuple combinations within specified ranges
    pub tuple: &'a [T],
    
    pub is_first: &'a [T],
    /// Number of range checks requested for each tuple combination
    pub mult: &'a T,
}

impl<'a, T> RangeTupleCols<'a, T> {
    fn from_slice<const N: usize>(slice: &'a [T]) -> Self {
        let (tuple, rest) = slice.split_at(N);
        let (is_first, rest) = rest.split_at(N - 1);
        Self {
            tuple,
            is_first,
            mult: &rest[0],
        }
    }
}

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
        N * 2
    }
}

impl<AB: InteractionBuilder + PairBuilder, const N: usize> Air<AB> for RangeTupleCheckerAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local = RangeTupleCols::from_slice::<N>((*local).borrow());
        let next = RangeTupleCols::from_slice::<N>((*next).borrow());

        // Constrain tuples for first and last rows
        for i in 0..N {
            builder.when_first_row().assert_zero(local.tuple[i]);
            builder.when_last_row().assert_eq(
                local.tuple[i],
                AB::F::from_canonical_u32(self.bus.sizes[i] - 1),
            );
        }

        // The leftmost tuple column can stay the same or increment by one
        builder.when_transition().assert_bool(next.tuple[0] - local.tuple[0]);
        // The middle tuple columns can stay the same, increment by one, or wrap (wrap is handled later)
        for i in 1..N-1 {
            builder.when_transition().when_ne(next.is_first[i - 1], AB::Expr::ONE).assert_bool(next.tuple[i] - local.tuple[i]);
        }
        // The rightmost tuple column should always either increment by one or wrap (wrap is handled later)
        builder.when_transition().when_ne(next.is_first[N-2], AB::Expr::ONE).assert_one(
            next.tuple[N-1] - local.tuple[N-1]
        );

        // Constrain the first row of is_first to always be one, and constain is_first to always be bool
        for i in 0..N-1 {
            builder.when_first_row().assert_one(local.is_first[i]);
            builder.assert_bool(local.is_first[i]);
        }

        // Constrain is_first based on differences between consecutive tuple rows
        builder.when_transition().assert_eq(
            next.tuple[0] - local.tuple[0],
            next.is_first[0]
        );
        for i in 1..N-1 {
            builder.when_transition().when_ne(next.is_first[i - 1], AB::Expr::ONE).assert_eq(
                next.tuple[i] - local.tuple[i],
                next.is_first[i]
            );
            builder.when_transition().when(next.is_first[i - 1]).assert_one(next.is_first[i]);
        }

        // Handle wrapping by constraining the start and end points of each counter
        for i in 0..N-1 {
            builder.when_transition().when(next.is_first[i]).assert_eq(
                local.tuple[i + 1], 
                AB::F::from_canonical_u32(self.bus.sizes[i + 1] - 1)
            );
            builder.when_transition().when(next.is_first[i]).assert_eq(
                next.tuple[i + 1], 
                AB::Expr::ZERO
            );
        }

        self.bus
            .receive(local.tuple.to_vec())
            .eval(builder, *local.mult);
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
        assert!(N > 1, "RangeTupleChecker requires at least 2 dimensions");
        let range_max = bus.sizes.iter().product();
        assert!(
            range_max > 0 && (range_max & (range_max - 1)) == 0,
            "RangeTupleChecker requires range_max ({}) to be a power of 2",
            range_max
        );
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

    pub fn generate_trace<F: Field + PrimeField32>(&self) -> RowMajorMatrix<F> {
        let mut unrolled_matrix = Vec::with_capacity(self.count.len() * 2 * N);
        
        for i in 0..self.count.len() {
            let mut tmp_idx = i as u32;
            let mut tuple = [0u32; N];
            for j in (0..N).rev() {
                tuple[j] = tmp_idx % self.air.bus.sizes[j];
                tmp_idx = tmp_idx / self.air.bus.sizes[j];
            }
            
            let mut first_nonzero = 0;
            for j in 0..N {
                if tuple[j] != 0 {
                    first_nonzero = j;
                }
            }
            
            let mut is_first = vec![0u32; N - 1];
            for j in 0..first_nonzero {
                is_first[j] = 0;
            }
            for j in first_nonzero..(N - 1) {
                is_first[j] = 1;
            }
            
            unrolled_matrix.extend(tuple);
            unrolled_matrix.extend(&is_first);
            unrolled_matrix.push(self.count[i].swap(0, std::sync::atomic::Ordering::Relaxed));
        }

        RowMajorMatrix::new(
            unrolled_matrix
                .iter()
                .map(|&v| F::from_canonical_u32(v))
                .collect(),
            N * 2,
        )
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
        N * 2
    }
}
