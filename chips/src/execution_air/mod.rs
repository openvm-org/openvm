use getset::Getters;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
pub mod tests;

#[derive(Clone, Getters)]
pub struct ExecutionAir {
    #[getset(get = "pub")]
    bus_index: usize,
    #[getset(get = "pub")]
    idx_len: usize,
    #[getset(get = "pub")]
    data_len: usize,
}

impl ExecutionAir {
    pub fn new(bus_index: usize, idx_len: usize, data_len: usize) -> Self {
        Self {
            bus_index,
            idx_len,
            data_len,
        }
    }

    pub fn air_width(&self) -> usize {
        3 + self.idx_len + self.data_len
    }
}
