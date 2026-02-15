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
    get_air_name,
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{AirProvingContext, CpuBackend},
    BaseAirWithPublicValues, Chip, ChipUsageGetter, PartitionedBaseAir, StarkProtocolConfig, Val,
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
pub struct RangeTupleColsRef<'a, T> {
    /// Contains all possible tuple combinations within specified ranges. Has size N.
    pub tuple: &'a [T],
    /// Number of range checks requested for each tuple combination
    pub mult: &'a T,
}

impl<'a, T> RangeTupleColsRef<'a, T> {
    fn from_slice<const N: usize>(slice: &'a [T]) -> Self {
        let (tuple, rest) = slice.split_at(N);
        Self {
            tuple,
            mult: &rest[0],
        }
    }
}

pub struct RangeTupleColsRefMut<'a, T> {
    /// Contains all possible tuple combinations within specified ranges. Has size N.
    pub tuple: &'a mut [T],
    /// Number of range checks requested for each tuple combination
    pub mult: &'a mut T,
}

impl<'a, T> RangeTupleColsRefMut<'a, T> {
    fn from_slice_mut<const N: usize>(slice: &'a mut [T]) -> Self {
        let (tuple, rest) = slice.split_at_mut(N);
        Self {
            tuple,
            mult: &mut rest[0],
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
        N + 1
    }
}

// An explanation to the constraints is available in the README.
impl<AB: InteractionBuilder + PairBuilder, const N: usize> Air<AB> for RangeTupleCheckerAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local = RangeTupleColsRef::from_slice::<N>((*local).borrow());
        let next = RangeTupleColsRef::from_slice::<N>((*next).borrow());

        // (T1): The trace starts with `(0, ..., 0)`.
        // (T2): The trace ends with `(size[0]-1, ..., size[N-1]-1)`.
        for i in 0..N {
            builder.when_first_row().assert_zero(local.tuple[i]);
            builder
                .when_last_row()
                .assert_eq(local.tuple[i], AB::F::from_u32(self.bus.sizes[i] - 1));
        }

        // (T4): Between consecutive tuples, column `0` can stay the same or increment.
        builder
            .when_transition()
            .assert_bool(next.tuple[0] - local.tuple[0]);
        // (T5): Between consecutive tuples, all other columns can stay the same, increment, or
        // wrap.
        for i in 1..N - 1 {
            builder
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ZERO)
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ONE)
                .assert_eq(local.tuple[i], AB::F::from_u32(self.bus.sizes[i] - 1));
            builder
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ZERO)
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ONE)
                .assert_eq(next.tuple[i], AB::Expr::ZERO);
        }
        // (T3): Between consecutive tuples, column `N-1` can increment or wrap.
        builder
            .when_ne(next.tuple[N - 1] - local.tuple[N - 1], AB::Expr::ONE)
            .assert_eq(
                local.tuple[N - 1],
                AB::F::from_u32(self.bus.sizes[N - 1] - 1),
            );
        builder
            .when_ne(next.tuple[N - 1] - local.tuple[N - 1], AB::Expr::ONE)
            .assert_eq(next.tuple[N - 1], AB::Expr::ZERO);

        // (T6): Between consecutive tuples, column `i` increments or wraps if and only if column
        // `i+1` wraps.
        for i in 0..N - 1 {
            let x = next.tuple[i] - local.tuple[i];
            let y = next.tuple[i + 1] - local.tuple[i + 1];
            let a = -AB::F::from_u32(self.bus.sizes[i] - 1);
            let b = -AB::F::from_u32(self.bus.sizes[i + 1] - 1);
            // See range_tuple/README.md
            builder.assert_zero(
                y.clone() * (y.clone() - AB::Expr::ONE) * (-x.clone() * (a + AB::F::ONE) + a)
                    + x.clone() * x.clone() * (y.clone() * (b + b - AB::F::ONE) - b * b),
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
        let mut rows = F::zero_vec(self.count.len() * (N + 1));

        for (i, row) in rows.chunks_exact_mut(N + 1).enumerate() {
            let cols = RangeTupleColsRefMut::from_slice_mut::<N>(row);
            let mut tmp_idx = i as u32;
            for j in (0..N).rev() {
                cols.tuple[j] = F::from_u32(tmp_idx % self.air.bus.sizes[j]);
                tmp_idx /= self.air.bus.sizes[j];
            }
            *cols.mult = F::from_u32(self.count[i].swap(0, std::sync::atomic::Ordering::Relaxed));
        }

        RowMajorMatrix::new(rows, N + 1)
    }
}

impl<R, SC: StarkProtocolConfig, const N: usize> Chip<R, CpuBackend<SC>>
    for RangeTupleCheckerChip<N>
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
        N + 1
    }
}
