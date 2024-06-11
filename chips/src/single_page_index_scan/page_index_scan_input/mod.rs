use std::sync::Arc;

use crate::{
    is_less_than_tuple::{columns::IsLessThanTupleAuxCols, IsLessThanTupleAir},
    range_gate::RangeCheckerGateChip,
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Clone)]
pub enum Comp {
    #[default]
    Lt,
}

pub enum PageIndexScanInputAir {
    Lt {
        /// The bus index
        bus_index: usize,
        /// The length of each index in the page table
        idx_len: usize,
        /// The length of each data entry in the page table
        data_len: usize,

        is_less_than_tuple_air: IsLessThanTupleAir,
    },
}

/// Given a fixed predicate of the form index OP x, where OP is one of {<, <=, =, >=, >}
/// and x is a private input, the PageIndexScanInputChip implements a chip such that the chip:
///
/// 1. Has public value x
/// 2. Sends all rows of the page that match the predicate index OP x where x is the public value
pub struct PageIndexScanInputChip {
    pub air: PageIndexScanInputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub cmp: Comp,
}

impl PageIndexScanInputChip {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bus_index: usize,
        idx_len: usize,
        data_len: usize,
        range_max: u32,
        idx_limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
        cmp: Comp,
    ) -> Self {
        Self {
            air: PageIndexScanInputAir::Lt {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index,
                    range_max,
                    idx_limb_bits.clone(),
                    decomp,
                ),
            },
            range_checker,
            cmp,
        }
    }

    pub fn page_width(&self) -> usize {
        match &self.air {
            PageIndexScanInputAir::Lt {
                idx_len, data_len, ..
            } => 1 + idx_len + data_len,
        }
    }

    pub fn aux_width(&self) -> usize {
        match &self.air {
            PageIndexScanInputAir::Lt {
                bus_index: _,
                idx_len,
                data_len: _,
                is_less_than_tuple_air,
            } => {
                idx_len
                    + 1
                    + 1
                    + IsLessThanTupleAuxCols::<usize>::get_width(
                        is_less_than_tuple_air.limb_bits(),
                        is_less_than_tuple_air.decomp(),
                        *idx_len,
                    )
            }
        }
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
