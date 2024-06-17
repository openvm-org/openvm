use crate::{group_by::group_by_input::columns::GroupByCols, is_equal_vec::IsEqualVecAir};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct GroupByAir {
    internal_bus: usize,
    output_bus: usize,

    is_equal_vec_air: IsEqualVecAir,

    // does not include is_allocated column
    page_width: usize,
    pub group_by_cols: Vec<usize>,
    pub aggregated_col: usize,
    // for now, will default to addition
    // operation: SubAir::eval,
}

impl GroupByAir {
    pub fn new(
        page_width: usize,
        group_by_cols: Vec<usize>,
        aggregated_col: usize,
        internal_bus: usize,
        output_bus: usize,
    ) -> Self {
        Self {
            page_width,
            group_by_cols: group_by_cols.clone(),
            aggregated_col,
            is_equal_vec_air: IsEqualVecAir::new(group_by_cols.len()),
            internal_bus,
            output_bus,
        }
    }
    pub fn get_width(&self) -> usize {
        self.page_width + 3 * self.group_by_cols.len() + 5
    }
    pub fn aux_width(&self) -> usize {
        3 * self.group_by_cols.len() + 5
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
// pub struct GroupByChip {
//     pub air: GroupByAir,
// }

// impl GroupByChip {
//     pub fn new(
//         page_width: usize,
//         group_by_cols: Vec<usize>,
//         aggregated_col: usize,
//         internal_bus: usize,
//         output_bus: usize,
//     ) -> Self {
//         Self {
//             air: GroupByAir {
//                 internal_bus,
//                 output_bus,
//                 page_width,
//                 group_by_cols,
//                 aggregated_col,
//                 is_equal_vec_air: IsEqualVecAir::new(page_width),
//             },
//         }
//     }
// }
pub struct GroupByChip {}
