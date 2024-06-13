use std::sync::Arc;

use crate::{is_less_than_tuple::IsLessThanTupleAir, range_gate::RangeCheckerGateChip};
use getset::Getters;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Getters)]
pub struct AssertSortedAir {
    #[getset(get = "pub")]
    is_less_than_tuple_air: IsLessThanTupleAir,
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
pub struct AssertSortedChip {
    air: AssertSortedAir,
    range_checker: Arc<RangeCheckerGateChip>,
}

impl AssertSortedChip {
    pub fn new(
        bus_index: usize,
        range_max: u32,
        limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: AssertSortedAir {
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index, range_max, limb_bits, decomp,
                ),
            },
            range_checker,
        }
    }
}
