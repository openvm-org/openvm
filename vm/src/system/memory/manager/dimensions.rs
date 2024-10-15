use derive_new::new;

// indicates that there are 2^`as_height` address spaces numbered starting from `as_offset`,
// and that each address space has 2^`address_height` addresses numbered starting from 0
#[derive(Clone, Copy, Debug, new)]
pub struct MemoryDimensions {
    /// Address space height
    pub as_height: usize,
    /// Pointer height
    pub address_height: usize,
    /// Address space offset
    pub as_offset: usize,
}

impl MemoryDimensions {
    pub fn overall_height(&self) -> usize {
        self.as_height + self.address_height
    }
}
