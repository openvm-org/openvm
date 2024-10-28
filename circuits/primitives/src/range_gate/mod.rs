//! Range check for a fixed bit size without using preprocessed trace.
//!
//! Caution: We almost always prefer to use the [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead of this chip.

use std::{
    borrow::Borrow,
    mem::{size_of, transmute},
    sync::atomic::AtomicU32,
};

use ax_circuit_derive::AlignedBorrow;
use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::indices_arr;

pub use crate::range::RangeCheckBus;

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Default, AlignedBorrow)]
pub struct RangeGateCols<T> {
    pub counter: T,
    pub mult: T,
}

impl<T: Clone> RangeGateCols<T> {
    pub fn from_slice(slice: &[T]) -> Self {
        let counter = slice[0].clone();
        let mult = slice[1].clone();

        Self { counter, mult }
    }
}

pub const NUM_RANGE_GATE_COLS: usize = size_of::<RangeGateCols<u8>>();
pub const RANGE_GATE_COL_MAP: RangeGateCols<usize> = make_col_map();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RangeCheckerGateAir {
    pub bus: RangeCheckBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for RangeCheckerGateAir {}
impl<F: Field> PartitionedBaseAir<F> for RangeCheckerGateAir {}
impl<F: Field> BaseAir<F> for RangeCheckerGateAir {
    fn width(&self) -> usize {
        NUM_RANGE_GATE_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for RangeCheckerGateAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &RangeGateCols<AB::Var> = (*local).borrow();
        let next: &RangeGateCols<AB::Var> = (*next).borrow();

        builder
            .when_first_row()
            .assert_eq(local.counter, AB::Expr::zero());
        builder
            .when_transition()
            .assert_eq(local.counter + AB::Expr::one(), next.counter);
        // The trace height is not part of the vkey, so we must enforce it here.
        builder.when_last_row().assert_eq(
            local.counter,
            AB::F::from_canonical_u32(self.bus.range_max - 1),
        );
        // Omit creating separate bridge.rs file for brevity
        self.bus.receive(local.counter).eval(builder, local.mult);
    }
}

/// This chip gets requests to verify that a number is in the range
/// [0, MAX). In the trace, there is a counter column and a multiplicity
/// column. The counter column is generated using a gate, as opposed to
/// the other RangeCheckerChip.
#[derive(Debug)]
pub struct RangeCheckerGateChip {
    pub air: RangeCheckerGateAir,
    pub count: Vec<AtomicU32>,
}

impl RangeCheckerGateChip {
    pub fn new(bus: RangeCheckBus) -> Self {
        let count = (0..bus.range_max).map(|_| AtomicU32::new(0)).collect();

        Self {
            air: RangeCheckerGateAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> RangeCheckBus {
        self.air.bus
    }

    pub fn bus_index(&self) -> usize {
        self.air.bus.index
    }

    pub fn range_max(&self) -> u32 {
        self.air.bus.range_max
    }

    pub fn air_width(&self) -> usize {
        2
    }

    pub fn add_count(&self, val: u32) {
        assert!(
            val < self.range_max(),
            "range exceeded: {} >= {}",
            val,
            self.range_max()
        );
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn clear(&self) {
        for i in 0..self.count.len() {
            self.count[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .count
            .iter()
            .enumerate()
            .flat_map(|(i, count)| {
                let c = count.load(std::sync::atomic::Ordering::Relaxed);
                vec![F::from_canonical_usize(i), F::from_canonical_u32(c)]
            })
            .collect();
        RowMajorMatrix::new(rows, NUM_RANGE_GATE_COLS)
    }
}

const fn make_col_map() -> RangeGateCols<usize> {
    let indices_arr = indices_arr::<NUM_RANGE_GATE_COLS>();
    unsafe { transmute::<[usize; NUM_RANGE_GATE_COLS], RangeGateCols<usize>>(indices_arr) }
}
