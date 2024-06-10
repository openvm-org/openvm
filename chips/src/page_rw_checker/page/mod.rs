pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct PageAir {
    page_bus: usize,
    idx_len: usize,
    data_len: usize,
}

impl PageAir {
    pub fn new(page_bus: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            page_bus,
            idx_len,
            data_len,
        }
    }

    pub fn air_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }
}
