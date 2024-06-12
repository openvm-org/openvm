use crate::is_equal_vec::IsEqualVecAir;
use getset::Getters;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Getters)]
pub struct GroupByAir {
    internal_bus: usize,
    output_bus: usize,

    is_equal_vec_air: IsEqualVecAir,

    page_width: usize,
    group_by_cols: Vec<usize>,
    aggregated_col: usize,
    // for now, will default to addition
    // operation: SubAir::eval,
}

impl GroupByAir {
    pub fn get_width(&self) -> usize {
        3 * self.page_width + self.group_by_cols.len() + 4
    }
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
}

impl GroupByChip {
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
    ) -> Self {
        Self {
            air: GroupByAir {
                internal_bus,
                output_bus,
                page_width,
                group_by_cols,
                aggregated_col,
                is_equal_vec_air: IsEqualVecAir::new(page_width),
            },
        }
    }
}
