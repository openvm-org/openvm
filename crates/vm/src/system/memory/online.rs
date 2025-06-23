use std::fmt::Debug;

use getset::Getters;
use itertools::{izip, zip_eq};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::NATIVE_AS;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    adapter::AccessAdapterInventory,
    offline_checker::MemoryBus,
    paged_vec::{AddressMap, PAGE_SIZE},
    Address, PagedVec,
};
use crate::{
    arch::MemoryConfig,
    system::memory::{
        adapter::records::{AccessLayout, MERGE_BEFORE_FLAG, SPLIT_AFTER_FLAG},
        MemoryImage,
    },
};

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

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, derive_new::new)]
pub struct AccessMetadata {
    /// The offset of the record in the corresponding adapter's arena
    pub offset: u32,
    /// The block size of the memory access
    pub block_size: u32,
    /// The timestamp of the last access.
    /// We don't _have_ to store it, but this is probably faster
    /// in terms of cache locality
    pub timestamp: u32,
}

impl AccessMetadata {
    pub const UNSPLITTABLE: u32 = u32::MAX;
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

    /// Updates the metadata with the given block.
    fn set_meta_block(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
        block_size: usize,
        timestamp: u32,
        offset: u32,
    ) {
        let ptr = pointer / align;
        let meta = unsafe { self.meta.get_unchecked_mut(address_space) };
        for i in 0..(block_size / align) {
            meta.set(
                (ptr + i) * size_of::<AccessMetadata>(),
                &AccessMetadata {
                    offset,
                    block_size: block_size as u32,
                    timestamp,
                },
            );
        }
    }

    /// Given all the necessary information about an access,
    /// adds a record for each of the relevant adapters about the access.
    /// Updates the metadata if `UPDATE_META` is true.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_access<T, const UPDATE_META: bool>(
        &mut self,
        block_size: usize,
        address_space: usize,
        pointer: usize,
        lowest_block_size: usize,
        timestamp: u32,
        prev_timestamps: Option<&[u32]>,
        values: &[T],
        prev_values: &[T],
        split_after: bool,
    ) {
        let mut adapter_block_size = lowest_block_size;
        let align = self.min_block_size[address_space] as usize;
        let offset = if block_size > lowest_block_size {
            self.access_adapter_inventory.current_size(block_size) as u32
        } else {
            AccessMetadata::UNSPLITTABLE
        };
        while adapter_block_size <= block_size / 2 {
            adapter_block_size *= 2;
            let record_mut = self.access_adapter_inventory.alloc_record(
                adapter_block_size,
                AccessLayout {
                    block_size,
                    cell_size: align,
                    type_size: size_of::<T>(),
                },
            );
            *record_mut.timestamp_and_mask = timestamp
                | (if prev_timestamps.is_some() {
                    MERGE_BEFORE_FLAG
                } else {
                    0
                })
                | (if split_after { SPLIT_AFTER_FLAG } else { 0 });
            *record_mut.address_space = address_space as u32;
            *record_mut.pointer = pointer as u32;
            *record_mut.block_size = block_size as u32;
            let data_slice = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    block_size * size_of::<T>(),
                )
            };
            record_mut.data.copy_from_slice(data_slice);
            let prev_data_slice = unsafe {
                std::slice::from_raw_parts(
                    prev_values.as_ptr() as *const u8,
                    block_size * size_of::<T>(),
                )
            };
            record_mut.prev_data.copy_from_slice(prev_data_slice);
            if let Some(prev_timestamps) = prev_timestamps {
                record_mut.timestamps.copy_from_slice(prev_timestamps);
            } // else we don't mind garbage values
        }
        if UPDATE_META {
            if split_after {
                for i in (0..block_size).step_by(lowest_block_size) {
                    self.set_meta_block(
                        address_space,
                        pointer + i,
                        align,
                        lowest_block_size,
                        timestamp,
                        AccessMetadata::UNSPLITTABLE,
                    );
                }
            } else {
                self.set_meta_block(address_space, pointer, align, block_size, timestamp, offset);
            }
        }
    }

    /// Returns the timestamp of the previous access to `[pointer:BLOCK_SIZE]_{address_space}`
    /// and the offset of the record in bytes.
    ///
    /// Caller must ensure alignment (e.g. via `assert_alignment`) prior to calling this function.
    fn prev_access_time<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
        values: &[T],
        prev_values: &[T],
    ) -> u32 {
        /***
         * Each element of meta contains the `block_size` and `timestamp` of the last access
         * to the corresponding `[align]` block, as well as the offset of the corresponding
         * record in the corresponding record arena. It also records the memory access it
         * is called for, as well as all the required accesses corresponding to initializing
         * new blocks.
         * If any of the previous memory accesses turn out in need to be split,
         * this function sets their corresponding flags.
         */
        let num_segs = BLOCK_SIZE / align;

        let begin = pointer / align;

        let first_meta =
            self.meta[address_space].get::<AccessMetadata>(begin * size_of::<AccessMetadata>());
        let mut need_to_merge = first_meta.block_size != BLOCK_SIZE as u32;
        let existing_metadatas = (0..num_segs)
            .flat_map(|i| {
                let meta = self.meta[address_space]
                    .get::<AccessMetadata>((begin + i) * size_of::<AccessMetadata>());
                if meta != first_meta {
                    need_to_merge = true;
                }
                if meta.block_size > 0 && meta.offset != AccessMetadata::UNSPLITTABLE {
                    Some(meta)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if need_to_merge {
            // Then we need to split everything we touched there
            for meta in existing_metadatas {
                self.access_adapter_inventory
                    .mark_to_split(meta.block_size as usize, meta.offset as usize);
            }
        }

        let prev_ts = (0..num_segs)
            .map(|i| {
                let meta = self.meta[address_space]
                    .get::<AccessMetadata>((begin + i) * size_of::<AccessMetadata>());
                if meta.block_size > 0 {
                    meta.timestamp
                } else {
                    // Initialize
                    if self.initial_block_size >= align {
                        // We need to split the initial block into chunks
                        let block_start = (begin + i) & !(self.initial_block_size / align - 1);
                        if (address_space as u32) < NATIVE_AS {
                            let initial_values = unsafe {
                                self.data.memory.get_slice::<u8>(
                                    (address_space as u32, (block_start * align) as u32),
                                    self.initial_block_size,
                                )
                            };
                            self.record_access::<u8, true>(
                                self.initial_block_size,
                                address_space,
                                block_start * align,
                                align,
                                INITIAL_TIMESTAMP,
                                None,
                                &initial_values,
                                &initial_values,
                                true,
                            );
                        } else {
                            let initial_values = (0..self.initial_block_size)
                                .map(|i| {
                                    self.data
                                        .memory
                                        .get_f(address_space as u32, (pointer + i) as u32)
                                })
                                .collect::<Vec<_>>();
                            self.record_access::<F, true>(
                                self.initial_block_size,
                                address_space,
                                block_start * align,
                                align,
                                INITIAL_TIMESTAMP,
                                None,
                                &initial_values,
                                &initial_values,
                                true,
                            );
                        }
                    } else {
                        debug_assert_eq!(self.initial_block_size, 1);
                        debug_assert!((address_space as u32) < NATIVE_AS);
                        self.record_access::<u8, true>(
                            align,
                            address_space,
                            pointer + i * align,
                            1,
                            INITIAL_TIMESTAMP,
                            Some(&[INITIAL_TIMESTAMP]),
                            &vec![0; align],
                            &vec![0; align],
                            false,
                        );
                    }
                    INITIAL_TIMESTAMP
                }
            })
            .collect::<Vec<_>>(); // TODO(AG): small buffer or small vec or something

        self.record_access::<T, true>(
            BLOCK_SIZE,
            address_space,
            pointer,
            align,
            self.timestamp,
            if need_to_merge { Some(&prev_ts) } else { None },
            values,
            prev_values,
            false,
        );

        *prev_ts.iter().max().unwrap()
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
        let values = self.data.read(address_space, pointer);
        let t_prev = self.prev_access_time::<T, BLOCK_SIZE>(
            address_space as usize,
            pointer as usize,
            ALIGN,
            &values,
            &values,
        );
        self.timestamp += 1;

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
        let values_prev = self.data.read(address_space, pointer);
        let t_prev = self.prev_access_time::<T, BLOCK_SIZE>(
            address_space as usize,
            pointer as usize,
            ALIGN,
            values,
            &values_prev,
        );
        self.data.write(address_space, pointer, values);
        self.timestamp += 1;

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

    /// Returns the list of all touched blocks. The list is sorted by address.
    /// If a block hasn't been explicitly accessed and is created by a split,
    /// the corresponding metadata has `offset` set to `AccessMetadata::UNSPLITTABLE`.
    pub fn touched_blocks(&mut self) -> Vec<(Address, AccessMetadata)> {
        let mut blocks = Vec::new();
        for (addr_space, (page, &align)) in zip_eq(&self.meta, &self.min_block_size).enumerate() {
            let mut next_idx = 0;
            for (idx, metadata) in page.iter::<AccessMetadata>() {
                if idx < next_idx {
                    continue;
                }
                if metadata.block_size != 0 {
                    if idx >= next_idx
                        && metadata.block_size > align
                        && metadata.offset != AccessMetadata::UNSPLITTABLE
                        && !self.access_adapter_inventory.is_marked_to_split(
                            metadata.block_size as usize,
                            metadata.offset as usize,
                        )
                    {
                        // This block is only intact if it's not split after
                        blocks.push(((addr_space as u32, idx as u32 * align), metadata));
                        next_idx = idx + (metadata.block_size / align) as usize;
                    } else {
                        // The next block is created by a split
                        // and does not exist in form of a record
                        blocks.push((
                            (addr_space as u32, idx as u32 * align),
                            AccessMetadata {
                                offset: AccessMetadata::UNSPLITTABLE,
                                block_size: align,
                                timestamp: metadata.timestamp,
                            },
                        ));
                        next_idx += 1;
                    }
                }
            }
        }
        blocks
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
