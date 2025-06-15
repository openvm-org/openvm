use std::fmt::Debug;

use getset::Getters;
use itertools::{izip, zip_eq};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::{
    adapter::{AccessAdapterInventory, AdapterInventoryTraceCursor},
    offline_checker::MemoryBus,
    paged_vec::{AddressMap, PAGE_SIZE},
    Address, MemoryAddress, PagedVec,
};
use crate::{arch::MemoryConfig, system::memory::MemoryImage};

pub const INITIAL_TIMESTAMP: u32 = 0;

#[derive(Debug, Clone, derive_new::new)]
pub struct GuestMemory {
    pub memory: AddressMap<PAGE_SIZE>,
}

impl GuestMemory {
    /// Returns `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)` or `repr(transparent)`,
    /// and it must be the exact type used to represent a single memory cell in
    /// address space `address_space`. For standard usage,
    /// `T` is either `u8` or `F` where `F` is the base field of the ZK backend.
    pub unsafe fn read<T, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE]
    where
        T: Copy + Debug,
    {
        debug_assert_eq!(
            size_of::<T>(),
            self.memory.cell_size[(addr_space - self.memory.as_offset) as usize]
        );
        let read = self
            .memory
            .paged_vecs
            .get_unchecked((addr_space - self.memory.as_offset) as usize)
            .get((ptr as usize) * size_of::<T>());
        read
    }

    /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// See [`GuestMemory::read`].
    pub unsafe fn write<T, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        values: &[T; BLOCK_SIZE],
    ) where
        T: Copy + Debug,
    {
        debug_assert_eq!(
            size_of::<T>(),
            self.memory.cell_size[(addr_space - self.memory.as_offset) as usize],
            "addr_space={addr_space}"
        );
        self.memory
            .paged_vecs
            .get_unchecked_mut((addr_space - self.memory.as_offset) as usize)
            .set((ptr as usize) * size_of::<T>(), values);
    }

    /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}` and returns
    /// the previous values.
    ///
    /// # Safety
    /// See [`GuestMemory::read`].
    #[inline(always)]
    pub unsafe fn replace<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: &[T; BLOCK_SIZE],
    ) -> [T; BLOCK_SIZE]
    where
        T: Copy + Debug,
    {
        let prev = self.read(address_space, pointer);
        self.write(address_space, pointer, values);
        prev
    }
}

// /// API for guest memory conforming to OpenVM ISA
// pub trait GuestMemory {
//     /// Returns `[pointer:BLOCK_SIZE]_{address_space}`
//     ///
//     /// # Safety
//     /// The type `T` must be stack-allocated `repr(C)` or `repr(transparent)`,
//     /// and it must be the exact type used to represent a single memory cell in
//     /// address space `address_space`. For standard usage,
//     /// `T` is either `u8` or `F` where `F` is the base field of the ZK backend.
//     unsafe fn read<T, const BLOCK_SIZE: usize>(
//         &self,
//         address_space: u32,
//         pointer: u32,
//     ) -> [T; BLOCK_SIZE]
//     where
//         T: Copy + Debug;

//     /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}`
//     ///
//     /// # Safety
//     /// See [`GuestMemory::read`].
//     unsafe fn write<T, const BLOCK_SIZE: usize>(
//         &mut self,
//         address_space: u32,
//         pointer: u32,
//         values: &[T; BLOCK_SIZE],
//     ) where
//         T: Copy + Debug;

//     /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}` and returns
//     /// the previous values.
//     ///
//     /// # Safety
//     /// See [`GuestMemory::read`].
//     #[inline(always)]
//     unsafe fn replace<T, const BLOCK_SIZE: usize>(
//         &mut self,
//         address_space: u32,
//         pointer: u32,
//         values: &[T; BLOCK_SIZE],
//     ) -> [T; BLOCK_SIZE]
//     where
//         T: Copy + Debug,
//     {
//         let prev = self.read(address_space, pointer);
//         self.write(address_space, pointer, values);
//         prev
//     }
// }

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
#[derive(Clone, Copy, PartialEq, Eq, Debug, derive_new::new)]
pub struct AccessMetadata {
    pub timestamp: u32,
    pub block_size: u32,
}

impl AccessMetadata {
    /// A marker indicating that the element is a part of a larger block which starts earlier.
    pub const OCCUPIED: u32 = u32::MAX;
}

/// Online memory that stores additional information for trace generation purposes.
/// In particular, keeps track of timestamp.
#[derive(Getters)]
pub struct TracingMemory<F> {
    pub timestamp: u32,
    /// The initial block size -- this depends on the type of boundary chip.
    initial_block_size: usize,
    /// The underlying data memory, with memory cells typed by address space: see [AddressMap].
    // TODO: make generic in GuestMemory
    #[getset(get = "pub")]
    pub data: GuestMemory,
    /// A map of `addr_space -> (ptr / min_block_size[addr_space] -> (timestamp: u32, block_size:
    /// u32))` for the timestamp and block size of the latest access.
    pub(super) meta: Vec<PagedVec<PAGE_SIZE>>,
    /// For each `addr_space`, the minimum block size allowed for memory accesses. In other words,
    /// all memory accesses in `addr_space` must be aligned to this block size.
    pub min_block_size: Vec<u32>,
    pub access_adapter_inventory: AccessAdapterInventory<F>,
}

impl<F: PrimeField32> TracingMemory<F> {
    // TODO: per-address space memory capacity specification
    pub fn new(
        mem_config: &MemoryConfig,
        range_checker: SharedVariableRangeCheckerChip,
        memory_bus: MemoryBus,
        initial_block_size: usize,
    ) -> Self {
        assert_eq!(mem_config.as_offset, 1);
        let num_cells = 1usize << mem_config.pointer_max_bits; // max cells per address space
        let num_addr_sp = 1 + (1 << mem_config.as_height);
        let mut min_block_size = vec![1; num_addr_sp];
        // TMP: hardcoding for now
        min_block_size[1] = 4;
        min_block_size[2] = 4;
        min_block_size[3] = 4;
        let meta = min_block_size
            .iter()
            .map(|&min_block_size| {
                PagedVec::new(
                    num_cells
                        .checked_mul(size_of::<AccessMetadata>())
                        .unwrap()
                        .div_ceil(PAGE_SIZE * min_block_size as usize),
                )
            })
            .collect();
        Self {
            data: GuestMemory::new(AddressMap::from_mem_config(mem_config)),
            meta,
            min_block_size,
            timestamp: INITIAL_TIMESTAMP + 1,
            initial_block_size,
            access_adapter_inventory: AccessAdapterInventory::new(
                range_checker,
                memory_bus,
                mem_config.clk_max_bits,
                mem_config.max_access_adapter_n,
            ),
        }
    }

    /// Instantiates a new `Memory` data structure from an image.
    pub fn with_image(mut self, image: MemoryImage, _access_capacity: usize) -> Self {
        for (i, (paged_vec, cell_size)) in izip!(&image.paged_vecs, &image.cell_size).enumerate() {
            let num_cells = paged_vec.bytes_capacity() / cell_size;

            self.meta[i] = PagedVec::new(
                num_cells
                    .checked_mul(size_of::<AccessMetadata>())
                    .unwrap()
                    .div_ceil(PAGE_SIZE * self.min_block_size[i] as usize),
            );
        }
        self.data = GuestMemory::new(image);
        self
    }

    #[inline(always)]
    fn assert_alignment(&self, block_size: usize, align: usize, addr_space: u32, ptr: u32) {
        debug_assert!(block_size.is_power_of_two());
        debug_assert_eq!(block_size % align, 0);
        debug_assert_ne!(addr_space, 0);
        debug_assert_eq!(align as u32, self.min_block_size[addr_space as usize]);
        assert_eq!(
            ptr % (align as u32),
            0,
            "pointer={ptr} not aligned to {align}"
        );
    }

    pub(crate) fn execute_splits<const UPDATE_META: bool>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        align: usize,
        values: &[F],
        timestamp: u32,
    ) {
        if UPDATE_META {
            for i in 0..(values.len() / align) {
                self.set_meta_block(
                    address.address_space as usize,
                    address.pointer as usize + i * align,
                    align,
                    align,
                    timestamp,
                );
            }
        }
        let mut size = align;
        let MemoryAddress {
            address_space,
            pointer,
        } = address;
        while size < values.len() {
            size *= 2;
            for i in (0..values.len()).step_by(size) {
                self.access_adapter_inventory.execute_split(
                    MemoryAddress {
                        address_space,
                        pointer: pointer + i as u32,
                    },
                    &values[i..i + size],
                    timestamp,
                );
            }
        }
    }

    pub(crate) fn execute_merges<const UPDATE_META: bool>(
        &mut self,
        address: MemoryAddress<u32, u32>,
        align: usize,
        values: &[F],
        timestamps: &[u32],
    ) {
        if UPDATE_META {
            self.set_meta_block(
                address.address_space as usize,
                address.pointer as usize,
                align,
                values.len(),
                *timestamps.iter().max().unwrap(),
            );
        }
        let mut size = align;
        let MemoryAddress {
            address_space,
            pointer,
        } = address;
        while size < values.len() {
            size *= 2;
            for i in (0..values.len()).step_by(size) {
                let left_timestamp = timestamps[(i / align)..((i + size / 2) / align)]
                    .iter()
                    .max()
                    .unwrap();
                let right_timestamp = timestamps[((i + size / 2) / align)..((i + size) / align)]
                    .iter()
                    .max()
                    .unwrap();
                self.access_adapter_inventory.execute_merge(
                    MemoryAddress {
                        address_space,
                        pointer: pointer + i as u32,
                    },
                    &values[i..i + size],
                    *left_timestamp,
                    *right_timestamp,
                );
            }
        }
    }

    /// Updates the metadata with the given block.
    fn set_meta_block(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
        block_size: usize,
        timestamp: u32,
    ) {
        let ptr = pointer / align;
        let meta = unsafe { self.meta.get_unchecked_mut(address_space) };
        meta.set(
            ptr * size_of::<AccessMetadata>(),
            &AccessMetadata {
                timestamp,
                block_size: block_size as u32,
            },
        );
        for i in 1..(block_size / align) {
            meta.set(
                (ptr + i) * size_of::<AccessMetadata>(),
                &AccessMetadata {
                    timestamp,
                    block_size: AccessMetadata::OCCUPIED,
                },
            );
        }
    }

    /// Returns the timestamp of the previous access to `[pointer:BLOCK_SIZE]_{address_space}`.
    /// If we need to split/merge/initialize something for this, we first do all the necessary
    /// actions. In the end of this process, we have this segment intact in our `meta`.
    ///
    /// Caller must ensure alignment (e.g. via `assert_alignment`) prior to calling this function.
    fn prev_access_time<const BLOCK_SIZE: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
    ) -> u32 {
        let num_segs = BLOCK_SIZE / align;

        let begin = pointer / align;
        let end = begin + BLOCK_SIZE / align;

        let mut prev_ts = INITIAL_TIMESTAMP;
        let mut block_timestamps = vec![INITIAL_TIMESTAMP; num_segs];
        let mut cur_ptr = begin;
        let need_to_merge = loop {
            if cur_ptr >= end {
                break true;
            }
            let mut current_metadata = self.meta[address_space]
                .get::<AccessMetadata>(cur_ptr * size_of::<AccessMetadata>());
            if current_metadata.block_size == BLOCK_SIZE as u32 && cur_ptr + num_segs == end {
                // We do not have to do anything
                prev_ts = current_metadata.timestamp;
                break false;
            } else if current_metadata.block_size == 0 {
                // Initialize
                if self.initial_block_size < align {
                    // Only happens in volatile, so empty initial memory
                    self.set_meta_block(
                        address_space,
                        cur_ptr * align,
                        align,
                        align,
                        INITIAL_TIMESTAMP,
                    );
                    current_metadata = AccessMetadata::new(INITIAL_TIMESTAMP, align as u32);
                } else {
                    cur_ptr -= cur_ptr % (self.initial_block_size / align);
                    self.set_meta_block(
                        address_space,
                        cur_ptr * align,
                        align,
                        self.initial_block_size,
                        INITIAL_TIMESTAMP,
                    );
                    current_metadata =
                        AccessMetadata::new(INITIAL_TIMESTAMP, self.initial_block_size as u32);
                }
            }
            prev_ts = prev_ts.max(current_metadata.timestamp);
            while current_metadata.block_size == AccessMetadata::OCCUPIED {
                cur_ptr -= 1;
                current_metadata = self.meta[address_space]
                    .get::<AccessMetadata>(cur_ptr * size_of::<AccessMetadata>());
            }
            block_timestamps[cur_ptr.saturating_sub(begin)
                ..((cur_ptr + (current_metadata.block_size as usize) / align).min(end) - begin)]
                .fill(current_metadata.timestamp);
            if current_metadata.block_size > align as u32 {
                // Split
                let address = MemoryAddress::new(address_space as u32, (cur_ptr * align) as u32);
                let values = (0..current_metadata.block_size as usize)
                    .map(|i| {
                        self.data
                            .memory
                            .get_f(address.address_space, address.pointer + (i as u32))
                    })
                    .collect::<Vec<_>>();
                self.execute_splits::<true>(address, align, &values, current_metadata.timestamp);
            }
            cur_ptr += current_metadata.block_size as usize / align;
        };
        if need_to_merge {
            // Merge
            let values = (0..BLOCK_SIZE)
                .map(|i| {
                    self.data
                        .memory
                        .get_f(address_space as u32, (pointer + i) as u32)
                })
                .collect::<Vec<_>>();
            self.execute_merges::<true>(
                MemoryAddress::new(address_space as u32, pointer as u32),
                align,
                &values,
                &block_timestamps,
            );
        }
        prev_ts
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
    pub unsafe fn read<T, const BLOCK_SIZE: usize, const ALIGN: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        self.assert_alignment(BLOCK_SIZE, ALIGN, address_space, pointer);
        let t_prev =
            self.prev_access_time::<BLOCK_SIZE>(address_space as usize, pointer as usize, ALIGN);
        let t_curr = self.timestamp;
        self.timestamp += 1;
        let values = self.data.read(address_space, pointer);
        self.set_meta_block(
            address_space as usize,
            pointer as usize,
            ALIGN,
            BLOCK_SIZE,
            t_curr,
        );

        (t_prev, values)
    }

    /// Atomic write operation that writes `values` into `[pointer:BLOCK_SIZE]_{address_space}` and
    /// then increments the timestamp by 1. Returns `(t_prev, values_prev)` which equal the
    /// timestamp and value `[pointer:BLOCK_SIZE]_{address_space}` of the last memory access.
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
    pub unsafe fn write<T, const BLOCK_SIZE: usize, const ALIGN: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: &[T; BLOCK_SIZE],
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        self.assert_alignment(BLOCK_SIZE, ALIGN, address_space, pointer);
        let t_prev =
            self.prev_access_time::<BLOCK_SIZE>(address_space as usize, pointer as usize, ALIGN);
        let values_prev = self.data.replace(address_space, pointer, values);
        let t_curr = self.timestamp;
        self.timestamp += 1;
        self.set_meta_block(
            address_space as usize,
            pointer as usize,
            ALIGN,
            BLOCK_SIZE,
            t_curr,
        );

        (t_prev, values_prev)
    }

    pub fn increment_timestamp(&mut self) {
        self.timestamp += 1;
    }

    pub fn increment_timestamp_by(&mut self, amount: u32) {
        self.timestamp += amount;
    }

    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Returns iterator over `((addr_space, ptr), (timestamp, block_size))` of the address, last
    /// accessed timestamp, and block size of all memory blocks that have been accessed since this
    /// instance of [TracingMemory] was constructed. This is similar to a soft-dirty mechanism,
    /// where the memory data is loaded from an initial image and considered "clean", and then
    /// all future accesses are marked as "dirty".
    // block_size is initialized to 0, so nonzero block_size happens to also mark "dirty" cells
    // **Assuming** for now that only the start of a block has nonzero block_size
    pub fn touched_blocks(&self) -> impl Iterator<Item = (Address, AccessMetadata)> + '_ {
        zip_eq(&self.meta, &self.min_block_size)
            .enumerate()
            .flat_map(move |(addr_space, (page, &align))| {
                page.iter::<AccessMetadata>()
                    .filter_map(move |(idx, metadata)| {
                        (metadata.block_size != 0
                            && metadata.block_size != AccessMetadata::OCCUPIED)
                            .then_some(((addr_space as u32, idx as u32 * align), metadata))
                    })
            })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::TracingMemory;
//     use crate::arch::MemoryConfig;

//     #[test]
//     fn test_write_read() {
//         let mut memory = TracingMemory::new(&MemoryConfig::default());
//         let address_space = 1;

//         unsafe {
//             memory.write(address_space, 0, &[1u8, 2, 3, 4]);

//             let (_, data) = memory.read::<u8, 2>(address_space, 0);
//             assert_eq!(data, [1u8, 2]);

//             memory.write(address_space, 2, &[100u8]);

//             let (_, data) = memory.read::<u8, 4>(address_space, 0);
//             assert_eq!(data, [1u8, 2, 100, 4]);
//         }
//     }
// }
