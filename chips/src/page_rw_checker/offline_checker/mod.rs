use crate::is_less_than_tuple::columns::IsLessThanTupleAuxCols;

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

pub struct OfflineChecker {
    page_bus_index: usize,
    range_bus_index: usize,
    ops_bus_index: usize,

    idx_len: usize,
    data_len: usize,
    idx_clk_limb_bits: Vec<usize>,
    idx_decomp: usize,
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
            range_bus_index,
            ops_bus_index,
            idx_len,
            data_len,
            idx_clk_limb_bits: [vec![idx_limb_bits; idx_len], vec![clk_bits]].concat(),
            idx_decomp,
        }
    }

    fn page_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }

    pub fn air_width(&self) -> usize {
        10 + self.page_width()
            + 2 * (self.idx_len + self.data_len)
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.idx_clk_limb_bits.clone(),
                self.idx_decomp,
                self.idx_len + 1,
            )
    }
}
