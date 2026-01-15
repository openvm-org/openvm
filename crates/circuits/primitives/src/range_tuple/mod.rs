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
    /// For 0 <= i <= N-1, tuple_inverse[i] = inv(tuple[i] - (bus.sizes[i] - 1))
    pub tuple_inverse: &'a [T],
    /// Define, for 0 <= i <= N-1, is_last[i] := tuple_inverse[i] * (tuple[i] - (bus.sizes[i] - 1))
    /// Then, for 0 <= i <= N-1, prefix_product[i] = is_last[0] * ... * is_last[i]
    /// Note that when N=2, we do not need this column as we can inline prefix_product[0]
    pub prefix_product: &'a [T],
    /// Number of range checks requested for each tuple combination
    pub mult: &'a T,
}

impl<'a, T> RangeTupleCols<'a, T> {
    fn from_slice<const N: usize>(slice: &'a [T]) -> Self {
        let (tuple, rest) = slice.split_at(N);
        let (tuple_inverse, rest) = rest.split_at(N - 1);
        let (prefix_product, rest) = rest.split_at(if N == 2 { 0 } else { N - 1 });
        Self {
            tuple,
            tuple_inverse,
            prefix_product,
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
        N * 2 + if N == 2 { 0 } else { N - 1 }
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
        // Constrain tuple_inverse
        for i in 0..N - 1 {
            builder.when_transition().assert_eq(
                local.tuple_inverse[i]
                    * (local.tuple[i] - AB::F::from_canonical_u32(self.bus.sizes[i] - 1))
                    * (local.tuple[i] - AB::F::from_canonical_u32(self.bus.sizes[i] - 1)),
                local.tuple[i] - AB::F::from_canonical_u32(self.bus.sizes[i] - 1),
            )
        }
        // Constrain prefix_product
        for i in 1..N - 1 {
            if i == 0 {
                builder.when_transition().assert_eq(
                    local.prefix_product[i],
                    AB::Expr::ONE
                        - (local.tuple_inverse[i]
                            * (local.tuple[i] - AB::F::from_canonical_u32(self.bus.sizes[i] - 1))),
                )
            } else {
                builder.when_transition().assert_eq(
                    local.prefix_product[i],
                    local.prefix_product[i - 1]
                        * (AB::Expr::ONE
                            - (local.tuple_inverse[i]
                                * (local.tuple[i]
                                    - AB::F::from_canonical_u32(self.bus.sizes[i] - 1)))),
                )
            }
        }
        if N == 2 {
            // when N==2, we can inline prefix_product to remove the column
            builder.when_transition().assert_eq(
                next.tuple[0],
                (local.tuple[0] + AB::Expr::ONE)
                    * (local.tuple_inverse[0]
                        * (local.tuple[0] - AB::F::from_canonical_u32(self.bus.sizes[0] - 1))),
            );
            builder.when_transition().assert_eq(
                next.tuple[1],
                local.tuple[1]
                    + (AB::Expr::ONE
                        - local.tuple_inverse[0]
                            * (local.tuple[0] - AB::F::from_canonical_u32(self.bus.sizes[0] - 1))),
            );
        } else {
            for i in 0..N {
                if i == 0 {
                    // the leftmost column changes with every row
                    // it wraps back to zero if it's previous value has reached its maximum
                    builder.when_transition().assert_eq(
                        next.tuple[i],
                        (local.tuple[i] + AB::Expr::ONE)
                            * (AB::Expr::ONE - local.prefix_product[i]),
                    );
                } else if i == N - 1 {
                    // the rightmost tuple column only changes of the previous values of all other columns are at their maximums
                    // there is no need for this column to ever wrap
                    builder
                        .when_transition()
                        .assert_eq(next.tuple[i], local.tuple[i] + local.prefix_product[i - 1]);
                } else {
                    // for all other tuple columns, the value of the column changes if the previous
                    // values of the columns to the left of the current column are at their maximums
                    // the column must wrap if it, and all columns to the left of it are at their maximums
                    builder.when_transition().assert_eq(
                        next.tuple[i],
                        (local.tuple[i] + local.prefix_product[i - 1])
                            * (AB::Expr::ONE - local.prefix_product[i]),
                    );
                }
            }
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
            .rev()
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
        let mut tuple = [0u32; N];
        for i in 0..self.count.len() {
            let tuple_inverse: Vec<u32> = (0..N - 1)
                .map(|j| {
                    if tuple[j] != (self.air.bus.sizes[j] - 1) {
                        (F::from_canonical_u32(tuple[j])
                            - F::from_canonical_u32(self.air.bus.sizes[j] - 1))
                        .inverse()
                        .as_canonical_u32()
                    } else {
                        0u32
                    }
                })
                .collect();
            let mut prev = F::ONE;
            let prefix_product: Vec<u32> = (0..N - 1)
                .map(|j| {
                    prev = prev
                        * (F::ONE
                            - F::from_canonical_u32(tuple_inverse[j])
                                * (F::from_canonical_u32(tuple[j])
                                    - F::from_canonical_u32(self.air.bus.sizes[j] - 1)));
                    prev.as_canonical_u32()
                })
                .collect();

            unrolled_matrix.extend(tuple);
            unrolled_matrix.extend(tuple_inverse);
            if N != 2 {
                unrolled_matrix.extend(prefix_product);
            }
            unrolled_matrix.push(self.count[i].swap(0, std::sync::atomic::Ordering::Relaxed));

            for j in 0..N {
                if tuple[j] < self.air.bus.sizes[j] - 1 {
                    tuple[j] += 1;
                    break;
                }
                tuple[j] = 0;
            }
        }

        RowMajorMatrix::new(
            unrolled_matrix
                .iter()
                .map(|&v| F::from_canonical_u32(v))
                .collect(),
            N * 2 + if N == 2 { 0 } else { N - 1 },
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
        N * 2 + if N == 2 { 0 } else { N - 1 }
    }
}
