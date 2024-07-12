pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

use getset::Getters;

#[derive(Clone, Default, Getters)]
// a single row chip meant to start the flow from the root
pub struct RootSignalAir<const COMMITMENT_LEN: usize> {
    #[getset(get = "pub")]
    bus_index: usize,
    #[getset(get = "pub")]
    is_init: bool,
    #[getset(get = "pub")]
    idx_len: usize,
}

impl<const COMMITMENT_LEN: usize> RootSignalAir<COMMITMENT_LEN> {
    pub fn new(bus_index: usize, is_init: bool, idx_len: usize) -> Self {
        RootSignalAir {
            bus_index,
            is_init,
            idx_len,
        }
    }
    pub fn air_width(&self) -> usize {
        COMMITMENT_LEN + 2 + (1 - self.is_init as usize) * 2 * self.idx_len
    }
}
