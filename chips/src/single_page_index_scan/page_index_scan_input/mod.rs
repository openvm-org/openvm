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

#[derive(Default)]
pub enum Comp {
    #[default]
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}

#[derive(Default, Getters)]
pub struct PageIndexScanInputAir {
    pub bus_index: usize,
    pub idx_len: usize,
    pub data_len: usize,

    #[getset(skip)]
    is_less_than_tuple_air: IsLessThanTupleAir,
}

pub struct PageIndexScanInputChip {
    pub air: PageIndexScanInputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl PageIndexScanInputChip {
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
            air: PageIndexScanInputAir {
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
        }
    }

    pub fn page_width(&self) -> usize {
        1 + self.air.idx_len + self.air.data_len
    }

    pub fn aux_width(&self) -> usize {
        self.air.idx_len
            + 1
            + 1
            + IsLessThanTupleAuxCols::<usize>::get_width(
                self.air.is_less_than_tuple_air.limb_bits(),
                self.air.is_less_than_tuple_air.decomp(),
                self.air.idx_len,
            )
    }

    pub fn air_width(&self) -> usize {
        self.page_width() + self.aux_width()
    }
}
