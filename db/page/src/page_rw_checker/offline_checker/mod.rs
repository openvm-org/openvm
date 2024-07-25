use afs_primitives::offline_checker::OfflineChecker;

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

use columns::PageOfflineCheckerCols;

pub struct PageOfflineChecker {
    offline_checker: OfflineChecker,
    page_bus_index: usize,
}

impl PageOfflineChecker {
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
            offline_checker: OfflineChecker::new(
                [vec![idx_limb_bits; idx_len], vec![clk_bits]].concat(),
                idx_decomp,
                idx_len,
                data_len,
                range_bus_index,
                ops_bus_index,
            ),
            page_bus_index,
        }
    }

    pub fn air_width(&self) -> usize {
        PageOfflineCheckerCols::<usize>::width(self)
    }
}
