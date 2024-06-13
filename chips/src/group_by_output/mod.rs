use std::sync::Arc;

use crate::{
    assert_sorted::{AssertSortedAir, AssertSortedChip},
    range_gate::RangeCheckerGateChip,
};
use getset::Getters;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Getters)]
pub struct GroupByOutputAir {
    assert_sorted_air: AssertSortedAir,
}

/// This chip constrains that consecutive rows are sorted lexicographically.
///
/// Each row consists of a key decomposed into limbs. Each limb has its own max number of
/// bits, given by the limb_bits array. The chip assumes that each limb is within its
/// given max limb_bits.
///
/// The AssertSortedChip uses the IsLessThanTupleChip as a subchip to check that the rows
/// are sorted lexicographically.
#[derive(Default)]
pub struct GroupByOutputChip {
    air: AssertSortedChip,
}

impl GroupByOutputChip {
    pub fn new(
        bus_index: usize,
        range_max: u32,
        limb_bits: Vec<usize>,
        decomp: usize,
        keys: Vec<Vec<u32>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: GroupByOutputAir {
                assert_sorted_air: AssertSortedAir {
                    bus_index,
                    range_max,
                    limb_bits,
                    decomp,
                    keys,
                    range_checker,
                },
            },
        }
    }
}
