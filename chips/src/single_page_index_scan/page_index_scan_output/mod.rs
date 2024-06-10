use std::sync::Arc;

use getset::Getters;

use crate::{is_less_than_tuple::IsLessThanTupleAir, range_gate::RangeCheckerGateChip};

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
pub struct PageIndexScanOutputAir {
    pub bus_index: usize,
    pub idx_len: usize,
    pub data_len: usize,

    #[getset(get = "pub")]
    is_less_than_tuple_air: IsLessThanTupleAir,
}

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
        limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: PageIndexScanOutputAir {
                bus_index,
                idx_len,
                data_len,
                is_less_than_tuple_air: IsLessThanTupleAir::new(
                    bus_index, range_max, limb_bits, decomp,
                ),
            },
            range_checker,
        }
    }
}
