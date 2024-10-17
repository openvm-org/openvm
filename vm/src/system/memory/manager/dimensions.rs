use derive_new::new;
use p3_field::{AbstractField, PrimeField32};

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
        self.as_height + self.address_height + 1
    }

    pub fn as_label_to_as<F: AbstractField>(&self, as_label: u64) -> F {
        let address_space = (as_label >> (self.address_height + 1)) + self.as_offset as u64;
        F::from_canonical_u32(address_space.try_into().unwrap())
    }

    pub fn as_to_as_label<F: PrimeField32>(&self, address_space: F) -> u64 {
        (address_space.as_canonical_u64() - self.as_offset as u64) << (self.address_height + 1)
    }
}
