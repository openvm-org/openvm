use getset::Getters;

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
pub struct PageIndexScanVerifyAir {
    #[getset(get = "pub")]
    pub bus_index: usize,
    #[getset(get = "pub")]
    pub idx_len: usize,
    #[getset(get = "pub")]
    pub data_len: usize,
}

pub struct PageIndexScanVerifyChip {
    pub air: PageIndexScanVerifyAir,
}

impl PageIndexScanVerifyChip {
    pub fn new(bus_index: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            air: PageIndexScanVerifyAir {
                bus_index,
                idx_len,
                data_len,
            },
        }
    }
}
