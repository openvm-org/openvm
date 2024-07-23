use std::sync::Arc;

use crate::range_gate::RangeCheckerGateChip;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

pub use air::AssertSortedAir;

/// This chip constrains that consecutive rows are sorted lexicographically.
///
/// Each row consists of a key decomposed into limbs. Each limb has its own max number of
/// bits, given by the limb_bits array. The chip assumes that each limb is within its
/// given max limb_bits.
///
/// The AssertSortedChip uses the IsLessThanTupleChip as a subchip to check that the rows
/// are sorted lexicographically.
#[derive(Clone, Debug)]
pub struct AssertSortedChip {
    air: AssertSortedAir,
    range_checker: Arc<RangeCheckerGateChip>,
}

impl AssertSortedChip {
    pub fn new(
        bus_index: usize,
        limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: AssertSortedAir::new(bus_index, limb_bits, decomp),
            range_checker,
        }
    }
}
