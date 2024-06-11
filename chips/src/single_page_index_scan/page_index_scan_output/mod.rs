use std::sync::Arc;

use getset::Getters;

use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    range_gate::RangeCheckerGateChip,
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Getters)]
pub struct PageIndexScanOutputAir {
    /// The bus index for sends to range chip
    pub bus_index: usize,
    /// The length of each index in the page table
    pub idx_len: usize,
    /// The length of each data entry in the page table
    pub data_len: usize,

    #[getset(get = "pub")]
    is_less_than_tuple_air: IsLessThanTupleAir,
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
        bus_index: usize,
        idx_len: usize,
        data_len: usize,
        range_max: u32,
        idx_limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: PageIndexScanOutputAir {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index,
                    range_max,
                    idx_limb_bits,
                    decomp,
                ),
            },
            range_checker,
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.air.idx_len + self.air.data_len
    }

    pub fn aux_width(&self) -> usize {
        1 + IsLessThanTupleAuxCols::<usize>::get_width(
            self.air.is_less_than_tuple_air().limb_bits(),
            self.air.is_less_than_tuple_air().decomp(),
            self.air.idx_len,
        )
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
