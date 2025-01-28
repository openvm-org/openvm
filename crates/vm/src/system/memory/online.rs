use std::fmt::Debug;

use openvm_stark_backend::p3_field::PrimeField32;

use super::paged_vec::{AddressMap, PAGE_SIZE};
use crate::{
    arch::MemoryConfig,
    system::memory::{offline::INITIAL_TIMESTAMP, MemoryImage, RecordId},
};

/// A simple data structure to read to/write from memory.
///
/// Stores a log of memory accesses to reconstruct aspects of memory state for trace generation.
#[derive(Debug)]
pub struct Memory<F> {
    pub(super) data: AddressMap<F, PAGE_SIZE>,
    next_record_id: usize,
    timestamp: u32,
}

impl<F: PrimeField32> Memory<F> {
    pub fn new(mem_config: &MemoryConfig) -> Self {
        Self {
            data: AddressMap::from_mem_config(mem_config),
            timestamp: INITIAL_TIMESTAMP + 1,
            next_record_id: 0,
        }
    }

    /// Instantiates a new `Memory` data structure from an image.
    pub fn from_image(image: MemoryImage<F>) -> Self {
        Self {
            data: image,
            timestamp: INITIAL_TIMESTAMP + 1,
            next_record_id: 0,
        }
    }

    /// Writes an array of values to the memory at the specified address space and start index.
    ///
    /// Returns the `RecordId` for the memory record and the previous data.
    pub fn write<const N: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: [F; N],
    ) -> (RecordId, [F; N]) {
        assert!(N.is_power_of_two());

        let prev_data = self.data.set_range(&(address_space, pointer), &values);
        self.next_record_id += 1;
        self.timestamp += 1;

        (RecordId(self.next_record_id - 1), prev_data)
    }

    /// Reads an array of values from the memory at the specified address space and start index.
    pub fn read<const N: usize>(&mut self, address_space: u32, pointer: u32) -> (RecordId, [F; N]) {
        assert!(N.is_power_of_two());

        let values = if address_space == 0 {
            assert_eq!(N, 1, "cannot batch read from address space 0");
            [F::from_canonical_u32(pointer); N]
        } else {
            self.range_array::<N>(address_space, pointer)
        };
        self.next_record_id += 1;
        self.timestamp += 1;
        (RecordId(self.next_record_id - 1), values)
    }

    pub fn increment_timestamp_by(&mut self, amount: u32) {
        self.timestamp += amount;
        self.next_record_id += 1;
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    #[inline(always)]
    pub fn get(&self, address_space: u32, pointer: u32) -> F {
        *self.data.get(&(address_space, pointer)).unwrap_or(&F::ZERO)
    }

    #[inline(always)]
    fn range_array<const N: usize>(&self, address_space: u32, pointer: u32) -> [F; N] {
        self.data.get_range(&(address_space, pointer))
    }
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::Memory;
    use crate::arch::MemoryConfig;

    macro_rules! bba {
        [$($x:expr),*] => {
            [$(BabyBear::from_canonical_u32($x)),*]
        }
    }

    #[test]
    fn test_write_read() {
        let mut memory = Memory::new(&MemoryConfig::default());
        let address_space = 1;

        memory.write(address_space, 0, bba![1, 2, 3, 4]);

        let (_, data) = memory.read::<2>(address_space, 0);
        assert_eq!(data, bba![1, 2]);

        memory.write(address_space, 2, bba![100]);

        let (_, data) = memory.read::<4>(address_space, 0);
        assert_eq!(data, bba![1, 2, 100, 4]);
    }
}
