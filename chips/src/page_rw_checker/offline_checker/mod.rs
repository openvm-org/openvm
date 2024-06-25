use columns::OfflineCheckerCols;

use crate::{is_equal_vec::IsEqualVecAir, is_less_than_tuple::IsLessThanTupleAir};

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

pub struct OfflineChecker {
    page_bus_index: usize,
    ops_bus_index: usize,

    idx_len: usize,
    data_len: usize,
    idx_decomp: usize,

    is_equal_idx_air: IsEqualVecAir,
    lt_idx_clk_air: IsLessThanTupleAir,
}

impl OfflineChecker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        ops_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        clk_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            page_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_decomp,
            is_equal_idx_air: IsEqualVecAir::new(idx_len),
            lt_idx_clk_air: IsLessThanTupleAir::new(
                range_bus_index,
                [vec![idx_limb_bits; idx_len], vec![clk_bits]].concat(),
                idx_decomp,
            ),
        }
    }

    pub fn air_width(&self) -> usize {
        OfflineCheckerCols::<usize>::width(self)
    }
}
