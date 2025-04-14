use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{
    paged_vec::{AddressMap, PAGE_SIZE},
    PagedVec,
};
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
        &mut self,
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

// perf[jpw]: since we restrict `timestamp < 2^29`, we could pack `timestamp, log2(block_size)`
// into a single u32 to save half the memory, since `block_size` is a power of 2 and its log2
// is less than 2^3.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct AccessMetadata {
    timestamp: u32,
    block_size: u32,
}

/// Online memory that stores additional information for trace generation purposes.
/// In particular, keeps track of timestamp.
pub struct TracingMemory {
    pub timestamp: u32,
    // TODO: the memory struct should contain an array of the byte size of the type per address
    // space, passed in from MemoryConfig
    /// The underlying data memory, with memory cells typed by address space: see [AddressMap].
    pub(super) data: AddressMap<PAGE_SIZE>,
    /// A map of `addr_space -> (ptr / min_block_size[addr_space] -> (timestamp: u32, block_size:
    /// u32))` for the timestamp and block size of the latest access.
    pub(super) meta: Vec<PagedVec<PAGE_SIZE>>,
    /// For each `addr_space`, the minimum block size allowed for memory accesses. In other words,
    /// all memory accesses in `addr_space` must be aligned to this block size.
    pub(super) min_block_size: Vec<u32>,
    // TODO: access adapter
}

impl TracingMemory {
    pub fn new(mem_config: &MemoryConfig) -> Self {
        assert_eq!(mem_config.as_offset, 1);
        let mem_size = 1usize << mem_config.pointer_max_bits;
        let num_addr_sp = 1 + (1 << mem_config.as_height);
        let meta = vec![
            PagedVec::new(
                mem_size
                    .checked_mul(size_of::<AccessMetadata>())
                    .unwrap()
                    .div_ceil(PAGE_SIZE)
            );
            num_addr_sp
        ];
        let mut min_block_size = vec![1; meta.len()];
        // TMP: hardcoding for now
        min_block_size[1] = 4;
        min_block_size[2] = 4;
        Self {
            data: AddressMap::from_mem_config(mem_config),
            meta,
            min_block_size,
            timestamp: INITIAL_TIMESTAMP + 1,
        }
    }

    /// Instantiates a new `Memory` data structure from an image.
    pub fn from_image(image: MemoryImage, access_capacity: usize) -> Self {
        let mut meta = vec![PagedVec::new(0); image.as_offset as usize];
        for (paged_vec, cell_size) in image.paged_vecs.iter().zip(&image.cell_size) {
            let num_cells = paged_vec.memory_size() / cell_size;
            meta.push(PagedVec::new(
                num_cells
                    .checked_mul(size_of::<AccessMetadata>())
                    .unwrap()
                    .div_ceil(PAGE_SIZE),
            ));
        }
        let mut min_block_size = vec![1; meta.len()];
        // TMP: hardcoding for now
        min_block_size[1] = 4;
        min_block_size[2] = 4;
        Self {
            data: image,
            meta,
            min_block_size,
            timestamp: INITIAL_TIMESTAMP + 1,
        }
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

    /// Atomic read operation which increments the timestamp by 1.
    /// Returns `(t_prev, [pointer:BLOCK_SIZE]_{address_space})` where `t_prev` is the
    /// timestamp of the last memory access.
    ///
    /// The previous memory access is treated as atomic even if previous accesses were for
    /// a smaller block size. This is made possible by internal memory access adapters
    /// that split/merge memory blocks. More specifically, the last memory access corresponding
    /// to `t_prev` may refer to an atomic access inserted by the memory access adapters.
    ///
    /// # Assumptions
    /// The `BLOCK_SIZE` is a multiple of `ALIGN`, which must equal the minimum block size
    /// of `address_space`.
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)` or `repr(transparent)`,
    /// and it must be the exact type used to represent a single memory cell in
    /// address space `address_space`. For standard usage,
    /// `T` is either `u8` or `F` where `F` is the base field of the ZK backend.
    ///
    /// In addition:
    /// - `address_space` must be valid.
    #[inline(always)]
    pub unsafe fn read<T: Copy, const BLOCK_SIZE: usize, const ALIGN: u32>(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> (u32, [T; BLOCK_SIZE]) {
        assert!(BLOCK_SIZE.is_power_of_two());
        debug_assert_ne!(address_space, 0);
        debug_assert_eq!(ALIGN, self.min_block_size[address_space as usize]);
        let values = self.data.read(address_space, pointer);
        self.timestamp += 1;
        // Handle timestamp and block size:
        assert_eq!(
            pointer % ALIGN,
            0,
            "pointer={pointer} not aligned to {ALIGN}"
        );
        let access_ptr = pointer / ALIGN;
        // TODO: address space should be checked elsewhere
        let meta = unsafe { self.meta.get_unchecked_mut(address_space as usize) };
        let AccessMetadata {
            timestamp,
            block_size,
        } = meta.get((access_ptr as usize) * size_of::<AccessMetadata>());

        (todo!("t_prev"), values)
    }

    pub fn increment_timestamp_by(&mut self, amount: u32) {
        self.timestamp += amount;
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
    use super::TracingMemory;
    use crate::arch::MemoryConfig;

    #[test]
    fn test_write_read() {
        let mut memory = TracingMemory::new(&MemoryConfig::default());
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
