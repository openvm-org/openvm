use std::sync::Arc;

use crate::{
    is_equal_vec::IsEqualVecAir, is_less_than_tuple::IsLessThanTupleAir,
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
pub struct GroupByAir {
    input_bus: usize,
    output_bus: usize,

    is_equal_vec_air: IsEqualVecAir,
    is_less_than_tuple_air: IsLessThanTupleAir,

    page_width: usize,
    group_by_cols: Vec<usize>,
    aggregated_col: Vec<usize>,
    // for now, will default to addition
    // operation: SubAir::eval,
}

/// This chip constrains that group_by columns are sent to itself in that order, and rows are sorted in that order. Then the
/// aggregated columns are summed and with the group_by columns are sent to output bus.
///
/// Each row consists of a key decomposed into limbs. Each limb has its own max number of
/// bits, given by the limb_bits array. The chip assumes that each limb is within its
/// given max limb_bits.
///
/// The AssertSortedChip uses the IsLessThanTupleChip as a subchip to check that the rows
/// are sorted lexicographically.
#[derive(Default)]
pub struct GroupByChip {
    air: GroupByAir,
    range_checker: Arc<RangeCheckerGateChip>,
}

impl GroupByChip {
    pub fn new(
        bus_index: usize,
        range_max: u32,
        limb_bits: Vec<usize>,
        decomp: usize,
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: Vec<usize>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: GroupByAir {
                input_bus: bus_index,
                output_bus: bus_index,
                is_equal_vec_air: IsEqualVecAir::new(page_width),
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index, range_max, limb_bits, decomp,
                ),
                page_width,
                group_by_cols,
                aggregated_col,
            },
            range_checker,
        }
    }
}
