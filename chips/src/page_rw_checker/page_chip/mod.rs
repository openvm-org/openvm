pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

pub struct PageChip {
    bus_index: usize,
    idx_len: usize,
    data_len: usize,
}

impl PageChip {
    pub fn new(bus_index: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            bus_index,
            idx_len,
            data_len,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn air_width(&self) -> usize {
        1 + self.idx_len + self.data_len
    }
}
