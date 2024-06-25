use std::sync::Arc;

use getset::Getters;

use crate::{indexed_output_page_air::IndexedOutputPageAir, range_gate::RangeCheckerGateChip};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Getters)]
pub struct PageIndexScanOutputAir {
    /// The bus index for page row receives
    pub page_bus_index: usize,

    pub final_page_air: IndexedOutputPageAir,
}

/// This chip receives rows from the PageIndexScanInputChip and constrains that:
///
/// 1. All allocated rows are before unallocated rows
/// 2. The allocated rows are sorted in ascending order by index
/// 3. The allocated rows of the new page are exactly the result of the index scan (via interactions)
pub struct PageIndexScanOutputChip {
    pub air: PageIndexScanOutputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl PageIndexScanOutputChip {
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: PageIndexScanOutputAir {
                page_bus_index,
                final_page_air: IndexedOutputPageAir::new(
                    range_checker.bus_index(),
                    idx_len,
                    data_len,
                    idx_limb_bits,
                    decomp,
                ),
            },
            range_checker,
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.air.final_page_air.idx_len + self.air.final_page_air.data_len
    }

    pub fn aux_width(&self) -> usize {
        self.air.final_page_air.aux_width()
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
