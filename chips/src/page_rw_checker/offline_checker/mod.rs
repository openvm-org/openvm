use columns::OfflineCheckerCols;

use crate::offline_checker::GeneralOfflineChecker;

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;

pub struct OfflineChecker {
    general_offline_checker: GeneralOfflineChecker,
    page_bus_index: usize,
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
        let general_offline_checker = GeneralOfflineChecker::new(
            [vec![idx_limb_bits; idx_len], vec![clk_bits]].concat(),
            idx_decomp,
            idx_len,
            data_len,
            range_bus_index,
            ops_bus_index,
        );
        Self {
            general_offline_checker,
            page_bus_index,
        }
    }

    pub fn air_width(&self) -> usize {
        OfflineCheckerCols::<usize>::width(self)
    }
}
