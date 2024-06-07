use std::sync::Arc;

use getset::Getters;

use crate::{is_less_than_tuple::IsLessThanTupleAir, range_gate::RangeCheckerGateChip};

pub mod air;
pub mod chip;
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
pub struct PageIndexScanAir {
    #[getset(get = "pub")]
    pub bus_index: usize,
    #[getset(get = "pub")]
    pub idx_len: usize,
    #[getset(get = "pub")]
    pub data_len: usize,

    #[getset(get = "pub")]
    is_less_than_tuple_air: IsLessThanTupleAir,
}

pub struct PageIndexScanChip {
    pub air: PageIndexScanAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl PageIndexScanChip {
    pub fn new(
        bus_index: usize,
        idx_len: usize,
        data_len: usize,
        range_max: u32,
        limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: PageIndexScanAir {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index,
                    range_max,
                    limb_bits.clone(),
                    decomp,
                ),
            },
            range_checker,
        }
    }
}
