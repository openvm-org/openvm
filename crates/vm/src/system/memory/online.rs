use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::paged_vec::{AddressMap, PAGE_SIZE};
use crate::{
    arch::MemoryConfig,
    system::memory::{offline::INITIAL_TIMESTAMP, MemoryImage, RecordId},
};

/// API for guest memory conforming to OpenVM ISA
pub trait GuestMemory {
    /// Returns `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)` or `repr(transparent)`,
    /// and it must be the exact type used to represent a single memory cell in
    /// address space `address_space`. For standard usage,
    /// `T` is either `u8` or `F` where `F` is the base field of the ZK backend.
    unsafe fn read<T: Copy, const BLOCK_SIZE: usize>(
        &mut self, // &mut potentially for logs?
        address_space: u32,
        pointer: u32,
    ) -> [T; BLOCK_SIZE];

    /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// See [`GuestMemory::read`].
    unsafe fn write<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: &[T; BLOCK_SIZE],
    );

    /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}` and returns
    /// the previous values.
    ///
    /// # Safety
    /// See [`GuestMemory::read`].
    #[inline(always)]
    unsafe fn replace<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: &[T; BLOCK_SIZE],
    ) -> [T; BLOCK_SIZE] {
        let prev = self.read(address_space, pointer);
        self.write(address_space, pointer, values);
        prev
    }
}

// TO BE DELETED
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryLogEntry<T> {
    Read {
        address_space: u32,
        pointer: u32,
        len: usize,
    },
    Write {
        address_space: u32,
        pointer: u32,
        data: Vec<T>,
    },
    IncrementTimestampBy(u32),
}

/// A simple data structure to read to/write from memory.
/// Internally storage is in raw bytes (untyped) to align with host memory.
///
/// Stores a log of memory accesses to reconstruct aspects of memory state for trace generation.
pub struct Memory {
    // TODO: the memory struct should contain an array of the byte size of the type per address
    // space, passed in from MemoryConfig
    pub(super) data: AddressMap<PAGE_SIZE>,
    // TODO: delete
    pub(super) log: Vec<MemoryLogEntry<u8>>,
    timestamp: u32,
}

impl Memory {
    pub fn new(mem_config: &MemoryConfig) -> Self {
        Self {
            data: AddressMap::from_mem_config(mem_config),
            timestamp: INITIAL_TIMESTAMP + 1,
            log: Vec::with_capacity(mem_config.access_capacity),
        }
    }

    /// Instantiates a new `Memory` data structure from an image.
    pub fn from_image(image: MemoryImage, access_capacity: usize) -> Self {
        Self {
            data: image,
            timestamp: INITIAL_TIMESTAMP + 1,
            log: Vec::with_capacity(access_capacity),
        }
    }

    fn last_record_id(&self) -> RecordId {
        // TEMP[jpw]
        RecordId(0)
        // RecordId(self.log.len() - 1)
    }

    /// Writes an array of values to the memory at the specified address space and start index.
    ///
    /// Returns the `RecordId` for the memory record and the previous data.
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)`, and it must be the exact type used to
    /// represent a single memory cell in address space `address_space`. For standard usage, `T`
    /// is either `u8` or `F` where `F` is the base field of the ZK backend.
    // @dev: `values` is passed by reference since the data is copied into memory. Even though the
    // compiler probably optimizes it, we use reference to avoid any unnecessary copy of
    // `values` onto the stack in the function call.
    pub unsafe fn write<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: &[T; BLOCK_SIZE],
    ) -> (RecordId, [T; BLOCK_SIZE]) {
        debug_assert!(BLOCK_SIZE.is_power_of_two());

        let prev_data = self.data.replace(address_space, pointer, values);

        // self.log.push(MemoryLogEntry::Write {
        //     address_space,
        //     pointer,
        //     data: values.to_vec(),
        // });
        self.timestamp += 1;

        (self.last_record_id(), prev_data)
    }

    /// Reads an array of values from the memory at the specified address space and start index.
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)`, and it must be the exact type used to
    /// represent a single memory cell in address space `address_space`. For standard usage, `T`
    /// is either `u8` or `F` where `F` is the base field of the ZK backend.
    #[inline(always)]
    pub unsafe fn read<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> (RecordId, [T; BLOCK_SIZE]) {
        assert!(BLOCK_SIZE.is_power_of_two());
        debug_assert_ne!(address_space, 0);

        // self.log.push(MemoryLogEntry::Read {
        //     address_space,
        //     pointer,
        //     len: N,
        // });

        let values = self.data.read(address_space, pointer);
        self.timestamp += 1;
        (self.last_record_id(), values)
    }

    pub fn increment_timestamp_by(&mut self, amount: u32) {
        self.timestamp += amount;
        self.log.push(MemoryLogEntry::IncrementTimestampBy(amount))
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)`, and it must be the exact type used to
    /// represent a single memory cell in address space `address_space`. For standard usage, `T`
    /// is either `u8` or `F` where `F` is the base field of the ZK backend.
    #[inline(always)]
    pub unsafe fn get<T: Copy>(&self, address_space: u32, pointer: u32) -> T {
        self.data.get((address_space, pointer))
    }
}

#[cfg(test)]
mod tests {
    use super::Memory;
    use crate::arch::MemoryConfig;

    #[test]
    fn test_write_read() {
        let mut memory = Memory::new(&MemoryConfig::default());
        let address_space = 1;

        unsafe {
            memory.write(address_space, 0, &[1u8, 2, 3, 4]);

            let (_, data) = memory.read::<u8, 2>(address_space, 0);
            assert_eq!(data, [1u8, 2]);

            memory.write(address_space, 2, &[100u8]);

            let (_, data) = memory.read::<u8, 4>(address_space, 0);
            assert_eq!(data, [1u8, 2, 100, 4]);
        }
    }
}
