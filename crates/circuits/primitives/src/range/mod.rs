//! Range check for a fixed bit size via preprocessed trace.
//!
//! Caution: We almost always prefer to use the
//! [VariableRangeCheckerChip](super::var_range::VariableRangeCheckerChip) instead of this chip.
// Adapted from Valida

use core::mem::size_of;
use std::{
    borrow::{Borrow, BorrowMut},
    sync::atomic::AtomicU32,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir, PairBuilder},
    p3_field::Field,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

mod bus;

#[cfg(test)]
pub mod tests;

pub use bus::*;

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct RangeCols<T> {
    /// Number of range checks for each value
    pub mult: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone)]
#[repr(C)]
pub struct RangePreprocessedCols<T> {
    /// Contains all possible values within range [0, max)
    pub counter: T,
}

pub const NUM_RANGE_COLS: usize = size_of::<RangeCols<u8>>();
pub const NUM_RANGE_PREPROCESSED_COLS: usize = size_of::<RangePreprocessedCols<u8>>();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct RangeCheckerAir {
    pub bus: RangeCheckBus,
}

impl RangeCheckerAir {
    pub fn range_max(&self) -> u32 {
        self.bus.range_max
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RangeCheckerAir {}
impl<F: Field> PartitionedBaseAir<F> for RangeCheckerAir {}
impl<F: Field> BaseAir<F> for RangeCheckerAir {
    fn width(&self) -> usize {
        NUM_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        // Create lookup table with all values 0..range_max
        let column = (0..self.range_max()).map(F::from_canonical_u32).collect();
        Some(RowMajorMatrix::new_col(column))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for RangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &RangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &RangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.counter)
            .eval(builder, local.mult);
    }
}

pub struct RangeCheckerChip {
    pub air: RangeCheckerAir,
    /// Tracks multiplicity of each value in range
    count: Vec<AtomicU32>,
}

impl RangeCheckerChip {
    pub fn new(bus: RangeCheckBus) -> Self {
        let mut count = vec![];
        for _ in 0..bus.range_max {
            count.push(AtomicU32::new(0));
        }

        Self {
            air: RangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> RangeCheckBus {
        self.air.bus
    }

    pub fn range_max(&self) -> u32 {
        self.air.range_max()
    }

    pub fn add_count(&self, val: u32) {
        // Increment the count for this value when range checked
        let val_atomic = &self.count[val as usize];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(self.air.range_max() as usize * NUM_RANGE_COLS);
        for (n, row) in rows.chunks_exact_mut(NUM_RANGE_COLS).enumerate() {
            let cols: &mut RangeCols<F> = (*row).borrow_mut();
            // Set multiplicity for each value in range
            cols.mult =
                F::from_canonical_u32(self.count[n].swap(0, std::sync::atomic::Ordering::Relaxed));
        }
        RowMajorMatrix::new(rows, NUM_RANGE_COLS)
    }
}
