use derive_new::new;
use p3_field::PrimeField32;
use p3_util::log2_strict_usize;

use crate::{arch::MemoryConfig, system::memory::CHUNK};

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
    /// Convert an address label (address space, block id) to its index in the memory merkle tree.
    pub fn label_to_index<F: PrimeField32>(&self, label: (F, usize)) -> usize {
        let (addr_space, block_id) = label;
        ((addr_space.as_canonical_u32() as usize - self.as_offset) << self.address_height)
            + block_id
    }
}

impl MemoryConfig {
    pub fn memory_dimensions(&self) -> MemoryDimensions {
        MemoryDimensions {
            as_height: self.as_height,
            address_height: self.pointer_max_bits - log2_strict_usize(CHUNK),
            as_offset: self.as_offset,
        }
    }
}
