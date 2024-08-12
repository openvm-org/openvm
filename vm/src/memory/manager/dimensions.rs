use derive_new::new;
use p3_util::log2_ceil_usize;

// indicates that there are 2^`as_height` address spaces numbered starting from `as_offset`,
// and that each address space has 2^`address_height` addresses numbered starting from 0
#[derive(Clone, Copy, new)]
pub struct MemoryDimensions {
    pub as_height: usize,
    // TODO[osama]: name this address_log_height or pointer_log_height
    pub address_height: usize,
    pub as_offset: usize,
}

impl MemoryDimensions {
    pub fn overall_height(&self) -> usize {
        self.as_height + self.address_height
    }

    pub fn as_max_bits(&self) -> usize {
        log2_ceil_usize(self.as_offset + (1 << self.as_height))
    }
}
