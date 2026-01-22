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

/*
For consecutive tuples `(local, next)`, we say that,

- Column `i` stays the same if `local[i] == next[i]`.
- Column `i` increments if `local[i] + 1 == next[i]`.
- Column `i` wraps if `local[i] == size[i] - 1` and `next[i] == 0`.

The AIR enforces the following constraints for the `tuple` column:

- (T1): The trace starts with `(0, ..., 0)`.
- (T2): The trace ends with `(size[0]-1, ..., size[N-1]-1)`.
- (T3): Between consecutive tuples, column `N-1` can increment or wrap.
- (T4): Between consecutive tuples, column `0` can stay the same or increment.
- (T5): Between consecutive tuples, all other columns can stay the same, increment, or wrap.
- (T6): Between consecutive tuples, column `i` increments or wraps if and only if column `i+1` wraps.

The constraints imply that, if `local` is a valid tuple (i.e. all values are in range), then `next` is also a valid tuple. The proof of this is as follows:

By (T3), column `N-1` must increment or wrap, and when this is combined with (T4) + (T5) + (T6), it is implied that there exists an `0 <= i <= N-1` where columns `0` to `i-1` stay the same, columns `i+1` to `N-1` wrap, and column `i` increments. By definition, all columns except column `i` are valid and stay in bounds. As such, the only possibly problematic column is column `i`. However, if `next[i] >= size[i]`, then it is impossible for column `i` to ever wrap in any following rows, so the value will never become `size[i]-1`, which is required by (T2).

Additionally, the constraints also imply that `next` is the lexographically next valid tuple after `local`.

Note that the length of the trace will be exactly `size[0] * ... * size[N-1]`, which is valid only if `size[i]` is always a power of 2.

___

Another remaining question is what polynomial constraints can be used to obtain (T6).

Define:

- `x := next[i] - local[i]`
- `y := next[i + 1] - local[i + 1]`
- `a := -(size[i] - 1)`
- `b := -(size[i+1] - 1)`

Note that constraints (T3), (T4), (T5) already force $x \in \{0, 1, a\}$, $y \in \{0, 1, b\}$. As such, to get T6, we only need to constrain the following table:

| (x,y) | valid configuration |
|-------|---------------------|
| (0,0) | yes                 |
| (0,1) | yes                 |
| (0,b) | no                  |
| (1,0) | no                  |
| (1,1) | no                  |
| (1,b) | yes                 |
| (a,0) | no                  |
| (a,1) | no                  |
| (a,b) | yes                 |

Consider the table with columns for the polynomials $(x-1)(x-a)y(y-1)$, $x^2(y-b)^2$, and $(x-1)(x-a)y(y-1) - x^2(y-b)^2$:

| (x,y) | $(x-1)(x-a)y(y-1)$ | $x^2(y-b)^2$ | $(x-1)(x-a)y(y-1) - x^2(y-b)^2$ |
|-------|------------------|------------|-------------------------------|
| (0,0) | 0                | 0          | 0                             |
| (0,1) | 0                | 0          | 0                             |
| (0,b) | nonzero          | 0          | nonzero                       |
| (1,0) | 0                | nonzero    | nonzero                       |
| (1,1) | 0                | nonzero    | nonzero                       |
| (1,b) | 0                | 0          | 0                             |
| (a,0) | 0                | nonzero    | nonzero                       |
| (a,1) | 0                | nonzero    | nonzero                       |
| (a,b) | 0                | 0          | 0                             |

Note that $(x-1)(x-a)y(y-1) - x^2(y-b)^2 = ay^2 - axy^2 - xy^2 - ay - x^2y + 2bx^2y + axy + xy - b^2x^2$ which has degree 3.

Thus, if we add the constraint $ay^2 - axy^2 - xy^2 - ay - x^2y + 2bx^2y + axy + xy - b^2x^2 = 0$, we are able to fully obtain T6.
*/
impl<AB: InteractionBuilder + PairBuilder, const N: usize> Air<AB> for RangeTupleCheckerAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local = RangeTupleColsRef::from_slice::<N>((*local).borrow());
        let next = RangeTupleColsRef::from_slice::<N>((*next).borrow());

        // (T1): The trace starts with `(0, ..., 0)`.
        // (T2): The trace ends with `(size[0]-1, ..., size[N-1]-1)`.
        for i in 0..N {
            builder.when_first_row().assert_zero(local.tuple[i]);
            builder.when_last_row().assert_eq(
                local.tuple[i],
                AB::F::from_canonical_u32(self.bus.sizes[i] - 1),
            );
        }

        // (T4): Between consecutive tuples, column `0` can stay the same or increment.
        builder
            .when_transition()
            .assert_bool(next.tuple[0] - local.tuple[0]);
        // (T5): Between consecutive tuples, all other columns can stay the same, increment, or wrap.
        for i in 1..N - 1 {
            builder
                .when_transition()
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ZERO)
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ONE)
                .assert_eq(local.tuple[i], AB::F::from_canonical_u32(self.bus.sizes[i] - 1));
            builder
                .when_transition()
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ZERO)
                .when_ne(next.tuple[i] - local.tuple[i], AB::Expr::ONE)
                .assert_eq(next.tuple[i], AB::Expr::ZERO);
        }
        // (T3): Between consecutive tuples, column `N-1` can increment or wrap.
        builder
            .when_transition()
            .when_ne(next.tuple[N-1] - local.tuple[N-1], AB::Expr::ONE)
            .assert_eq(local.tuple[N-1], AB::F::from_canonical_u32(self.bus.sizes[N-1] - 1));
        builder
            .when_transition()
            .when_ne(next.tuple[N-1] - local.tuple[N-1], AB::Expr::ONE)
            .assert_eq(next.tuple[N-1], AB::Expr::ZERO);


        // (T6): Between consecutive tuples, column `i` increments or wraps if and only if column `i+1` wraps.
        for i in 0..N - 1 {
            let x = next.tuple[i] - local.tuple[i];
            let y = next.tuple[i + 1] - local.tuple[i + 1];
            let a = -AB::F::from_canonical_u32(self.bus.sizes[i] - 1); 
            let b = -AB::F::from_canonical_u32(self.bus.sizes[i + 1] - 1);
            builder
                .when_transition()
                .assert_zero(
                    y.clone()*y.clone()*a - 
                    x.clone()*y.clone()*y.clone()*a - 
                    x.clone()*y.clone()*y.clone() - 
                    y.clone()*a - 
                    x.clone()*x.clone()*y.clone() +
                    x.clone()*x.clone()*y.clone()*b + 
                    x.clone()*x.clone()*y.clone()*b + 
                    x.clone()*y.clone()*a + 
                    x.clone()*y.clone() - 
                    x.clone()*x.clone()*b*b
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
        let mut unrolled_matrix = Vec::with_capacity(self.count.len() * (N + 1));

        for i in 0..self.count.len() {
            let mut tmp_idx = i as u32;
            let mut tuple = [0u32; N];
            for j in (0..N).rev() {
                tuple[j] = tmp_idx % self.air.bus.sizes[j];
                tmp_idx /= self.air.bus.sizes[j];
            }
            unrolled_matrix.extend(tuple);
            unrolled_matrix.push(self.count[i].swap(0, std::sync::atomic::Ordering::Relaxed));
        }

        RowMajorMatrix::new(
            unrolled_matrix
                .iter()
                .map(|&v| F::from_canonical_u32(v))
                .collect(),
            N + 1,
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
        N + 1
    }
}
