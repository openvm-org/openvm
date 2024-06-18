use columns::OfflineCheckerCols;

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

    pub fn air_width(&self) -> usize {
        OfflineCheckerCols::<usize>::width(self)
    }
}
