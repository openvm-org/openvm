use crate::is_less_than_tuple::IsLessThanTupleAir;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

// pub enum Comp {
//     Lt,
//     Lte,
//     Eq,
//     Gte,
//     Gt,
// }

pub struct PageIndexScanAir {
    pub bus_index: usize,
    pub idx_len: usize,
    pub data_len: usize,

    pub limb_bits: Vec<usize>,
    pub decomp: usize,

    is_less_than_tuple_air: IsLessThanTupleAir,
    // pub cmp: Comp,
}

pub struct PageIndexScanChip {
    pub air: PageIndexScanAir,
}
