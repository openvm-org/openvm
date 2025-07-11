use std::{fmt::Debug, slice::from_raw_parts};

use getset::Getters;
use itertools::{izip, zip_eq};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::{exe::SparseMemoryImage, NATIVE_AS};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
};

use super::{adapter::AccessAdapterInventory, offline_checker::MemoryBus};
use crate::{
    arch::MemoryConfig,
    system::memory::{
        adapter::records::{AccessLayout, AccessRecordHeader, MERGE_AND_NOT_SPLIT_FLAG},
        MemoryImage,
    },
};

mod basic;
#[cfg(any(unix, windows))]
mod memmap;
mod paged_vec;

#[cfg(not(any(unix, windows)))]
pub use basic::*;
#[cfg(any(unix, windows))]
pub use memmap::*;
pub use paged_vec::PagedVec;

#[cfg(all(any(unix, windows), not(feature = "basic-memory")))]
pub type MemoryBackend = memmap::MmapMemory;
#[cfg(any(not(any(unix, windows)), feature = "basic-memory"))]
pub type MemoryBackend = basic::BasicMemory;

pub const INITIAL_TIMESTAMP: u32 = 0;
/// Default mmap page size. Change this if using THB.
pub const PAGE_SIZE: usize = 4096;

/// (address_space, pointer)
pub type Address = (u32, u32);

/// API for any memory implementation that allocates a contiguous region of memory.
pub trait LinearMemory {
    /// Create instance of `Self` with `size` bytes.
    fn new(size: usize) -> Self;
    /// Allocated size of the memory in bytes.
    fn size(&self) -> usize;
    /// Returns the entire memory as a raw byte slice.
    fn as_slice(&self) -> &[u8];
    /// Returns the entire memory as a raw byte slice.
    fn as_mut_slice(&mut self) -> &mut [u8];
    /// Read `BLOCK` from `self` at `from` address without moving it.
    ///
    /// Panics or segfaults if `from..from + size_of::<BLOCK>()` is out of bounds.
    ///
    /// # Safety
    /// - `BLOCK` should be "plain old data" (see [`Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)).
    ///   We do not add a trait bound due to Plonky3 types not implementing the trait.
    /// - See [`core::ptr::read`] for similar considerations.
    /// - Memory at `from` must be properly aligned for `BLOCK`. Use [`Self::read_unaligned`] if
    ///   alignment is not guaranteed.
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK;
    /// Read `BLOCK` from `self` at `from` address without moving it.
    /// Same as [`Self::read`] except that it does not require alignment.
    ///
    /// Panics or segfaults if `from..from + size_of::<BLOCK>()` is out of bounds.
    ///
    /// # Safety
    /// - `BLOCK` should be "plain old data" (see [`Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)).
    ///   We do not add a trait bound due to Plonky3 types not implementing the trait.
    /// - See [`core::ptr::read`] for similar considerations.
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK;
    /// Write `BLOCK` to `self` at `start` address without reading the old value. Does not drop
    /// `values`. Semantically, `values` is moved into the location pointed to by `start`.
    ///
    /// Panics or segfaults if `start..start + size_of::<BLOCK>()` is out of bounds.
    ///
    /// # Safety
    /// - See [`core::ptr::write`] for similar considerations.
    /// - Memory at `start` must be properly aligned for `BLOCK`. Use [`Self::write_unaligned`] if
    ///   alignment is not guaranteed.
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK);
    /// Write `BLOCK` to `self` at `start` address without reading the old value. Does not drop
    /// `values`. Semantically, `values` is moved into the location pointed to by `start`.
    /// Same as [`Self::write`] but without alignment requirement.
    ///
    /// Panics or segfaults if `start..start + size_of::<BLOCK>()` is out of bounds.
    ///
    /// # Safety
    /// - See [`core::ptr::write`] for similar considerations.
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK);
    /// Swaps `values` with memory at `start..start + size_of::<BLOCK>()`.
    ///
    /// Panics or segfaults if `start..start + size_of::<BLOCK>()` is out of bounds.
    ///
    /// # Safety
    /// - `BLOCK` should be "plain old data" (see [`Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)).
    ///   We do not add a trait bound due to Plonky3 types not implementing the trait.
    /// - Memory at `start` must be properly aligned for `BLOCK`.
    /// - The data in `values` should not overlap with memory in `self`.
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK);
    /// Copies `data` into memory at `to` address.
    ///
    /// Panics or segfaults if `to..to + size_of_val(data)` is out of bounds.
    ///
    /// # Safety
    /// - `T` should be "plain old data" (see [`Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)).
    ///   We do not add a trait bound due to Plonky3 types not implementing the trait.
    /// - The underlying memory of `data` should not overlap with `self`.
    /// - The starting pointer of `self` should be aligned to `T`.
    /// - The memory pointer at `to` should be aligned to `T`.
    unsafe fn copy_nonoverlapping<T: Copy>(&mut self, to: usize, data: &[T]);
    /// Returns a slice `&[T]` for the memory region `start..start + len`.
    ///
    /// Panics or segfaults if `start..start + len * size_of::<T>()` is out of bounds.
    ///
    /// # Safety
    /// - `T` should be "plain old data" (see [`Pod`](https://docs.rs/bytemuck/latest/bytemuck/trait.Pod.html)).
    ///   We do not add a trait bound due to Plonky3 types not implementing the trait.
    /// - Memory at `start` must be properly aligned for `T`.
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T];
}

/// Map from address space to linear memory.
/// The underlying memory is typeless, stored as raw bytes, but usage implicitly assumes that each
/// address space has memory cells of a fixed type (e.g., `u8, F`). We do not use a typemap for
/// performance reasons, and it is up to the user to enforce types. Needless to say, this is a very
/// `unsafe` API.
#[derive(Debug, Clone)]
pub struct AddressMap<M: LinearMemory = MemoryBackend> {
    pub mem: Vec<M>,
    /// byte size of cells per address space
    pub cell_size: Vec<usize>, // TODO: move to MmapWrapper
}

impl Default for AddressMap {
    fn default() -> Self {
        Self::from_mem_config(&MemoryConfig::default())
    }
}

impl<M: LinearMemory> AddressMap<M> {
    /// `mem_size` is the number of **cells** in each address space. It is required that
    /// `mem_size[0] = 0`.
    pub fn new(mem_size: Vec<usize>) -> Self {
        // TMP: hardcoding for now
        let mut cell_size = vec![1; 4];
        cell_size.resize(mem_size.len(), 4);
        let mem = zip_eq(&cell_size, &mem_size)
            .map(|(cell_size, mem_size)| M::new(mem_size.checked_mul(*cell_size).unwrap()))
            .collect();
        Self { mem, cell_size }
    }

    pub fn from_mem_config(mem_config: &MemoryConfig) -> Self {
        Self::new(mem_config.addr_space_sizes.clone())
    }

    #[inline(always)]
    pub fn get_memory(&self) -> &Vec<M> {
        &self.mem
    }

    #[inline(always)]
    pub fn get_memory_mut(&mut self) -> &mut Vec<M> {
        &mut self.mem
    }

    pub fn get_f<F: PrimeField32>(&self, addr_space: u32, ptr: u32) -> F {
        debug_assert_ne!(addr_space, 0);
        // TODO: fix this
        unsafe {
            if self.cell_size[addr_space as usize] == 1 {
                F::from_canonical_u8(self.get::<u8>((addr_space, ptr)))
            } else {
                debug_assert_eq!(self.cell_size[addr_space as usize], 4);
                self.get::<F>((addr_space, ptr))
            }
        }
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get<T: Copy>(&self, (addr_space, ptr): Address) -> T {
        debug_assert_eq!(size_of::<T>(), self.cell_size[addr_space as usize]);
        // SAFETY:
        // - alignment is automatic since we multiply by `size_of::<T>()`
        self.mem
            .get_unchecked(addr_space as usize)
            .read((ptr as usize) * size_of::<T>())
    }

    /// Panics or segfaults if `ptr..ptr + len` is out of bounds
    ///
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get_slice<T: Copy + Debug>(
        &self,
        (addr_space, ptr): Address,
        len: usize,
    ) -> &[T] {
        debug_assert_eq!(size_of::<T>(), self.cell_size[addr_space as usize]);
        let start = (ptr as usize) * size_of::<T>();
        let mem = self.mem.get_unchecked(addr_space as usize);
        // SAFETY:
        // - alignment is automatic since we multiply by `size_of::<T>()`
        mem.get_aligned_slice(start, len)
    }

    /// Panics or segfaults if `ptr..ptr + len` is out of bounds
    ///
    /// # Safety
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get_u8_slice(&self, addr_space: usize, start: usize, len: usize) -> &[u8] {
        let mem = self.mem.get_unchecked(addr_space);
        mem.get_aligned_slice(start, len)
    }

    /// Copies `data` into the memory at `(addr_space, ptr)`.
    ///
    /// Panics or segfaults if `ptr + size_of_val(data)` is out of bounds.
    ///
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - The linear memory in `addr_space` is aligned to `T`.
    pub unsafe fn copy_slice_nonoverlapping<T: Copy>(
        &mut self,
        (addr_space, ptr): Address,
        data: &[T],
    ) {
        let start = (ptr as usize) * size_of::<T>();
        // SAFETY:
        // - Linear memory is aligned to `T` and `start` is multiple of `size_of::<T>()` so
        //   alignment is satisfied.
        // - `data` and `self.mem` are non-overlapping
        self.mem
            .get_unchecked_mut(addr_space as usize)
            .copy_nonoverlapping(start, data);
    }

    // TODO[jpw]: stabilize the boundary memory image format and how to construct
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub fn from_sparse(mem_size: Vec<usize>, sparse_map: SparseMemoryImage) -> Self {
        let mut vec = Self::new(mem_size);
        for ((addr_space, index), data_byte) in sparse_map.into_iter() {
            // SAFETY:
            // - safety assumptions in function doc comments
            unsafe {
                vec.mem
                    .get_unchecked_mut(addr_space as usize)
                    .write_unaligned(index as usize, data_byte);
            }
        }
        vec
    }
}

/// API for guest memory conforming to OpenVM ISA
// @dev Note we don't make this a trait because phantom executors currently need a concrete type for
// guest memory
#[derive(Debug, Clone, derive_new::new)]
pub struct GuestMemory {
    pub memory: AddressMap,
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
        debug_assert_eq!(size_of::<T>(), self.memory.cell_size[addr_space as usize]);
        // SAFETY:
        // - `T` should be "plain old data"
        // - alignment for `[T; BLOCK_SIZE]` is automatic since we multiply by `size_of::<T>()`
        self.memory
            .get_memory()
            .get_unchecked(addr_space as usize)
            .read((ptr as usize) * size_of::<T>())
    }

    /// Writes `values` to `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// See [`GuestMemory::read`].
    pub unsafe fn write<T, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        values: [T; BLOCK_SIZE],
    ) where
        T: Copy + Debug,
    {
        debug_assert_eq!(size_of::<T>(), self.memory.cell_size[addr_space as usize]);
        // SAFETY:
        // - alignment for `[T; BLOCK_SIZE]` is automatic since we multiply by `size_of::<T>()`
        self.memory
            .get_memory_mut()
            .get_unchecked_mut(addr_space as usize)
            .write((ptr as usize) * size_of::<T>(), values);
    }

    /// Swaps `values` with `[pointer:BLOCK_SIZE]_{address_space}`.
    ///
    /// # Safety
    /// See [`GuestMemory::read`] and [`LinearMemory::swap`].
    #[inline(always)]
    pub unsafe fn swap<T, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        values: &mut [T; BLOCK_SIZE],
    ) where
        T: Copy + Debug,
    {
        debug_assert_eq!(size_of::<T>(), self.memory.cell_size[addr_space as usize]);
        // SAFETY:
        // - alignment for `[T; BLOCK_SIZE]` is automatic since we multiply by `size_of::<T>()`
        self.memory
            .get_memory_mut()
            .get_unchecked_mut(addr_space as usize)
            .swap((ptr as usize) * size_of::<T>(), values);
    }
}

// perf[jpw]: since we restrict `timestamp < 2^29`, we could pack `timestamp, log2(block_size)`
// into a single u32 to save some memory, since `block_size` is a power of 2 and its log2
// is less than 2^3.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, derive_new::new)]
pub struct AccessMetadata {
    /// The starting pointer of the access
    pub start_ptr: u32,
    /// The block size of the memory access
    pub block_size: u32,
    /// The timestamp of the last access.
    /// We don't _have_ to store it, but this is probably faster
    /// in terms of cache locality
    pub timestamp: u32,
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
    /// u32))` for the timestamp and block size of the latest access. Each
    /// `PagedVec<AccessMetadata>` stores metadata in a paged manner for memory efficiency.
    pub(super) meta: Vec<PagedVec<AccessMetadata>>,
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
        let num_cells = mem_config.addr_space_sizes.clone();
        let num_addr_sp = 1 + (1 << mem_config.addr_space_height);
        let mut min_block_size = vec![1; num_addr_sp];
        // TMP: hardcoding for now
        min_block_size[1] = 4;
        min_block_size[2] = 4;
        min_block_size[3] = 4;
        let meta = zip_eq(&min_block_size, &num_cells)
            .map(|(min_block_size, num_cells)| {
                let total_metadata_len = num_cells.div_ceil(*min_block_size as usize);
                PagedVec::new(total_metadata_len, PAGE_SIZE)
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
    pub fn with_image(mut self, image: MemoryImage) -> Self {
        for (i, (mem, cell_size)) in izip!(image.get_memory(), &image.cell_size).enumerate() {
            let num_cells = mem.size() / cell_size;

            let total_metadata_len = num_cells.div_ceil(self.min_block_size[i] as usize);
            self.meta[i] = PagedVec::new(total_metadata_len, PAGE_SIZE);
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
    #[inline]
    fn set_meta_block(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
        block_size: usize,
        timestamp: u32,
    ) {
        let ptr = pointer / align;
        // SAFETY: address_space is assumed to be valid and within bounds
        let meta = unsafe { self.meta.get_unchecked_mut(address_space) };
        for i in 0..(block_size / align) {
            meta.set(
                ptr + i,
                AccessMetadata {
                    start_ptr: pointer as u32,
                    block_size: block_size as u32,
                    timestamp,
                },
            );
        }
    }

    pub(crate) fn add_split_record(&mut self, header: AccessRecordHeader) {
        if header.block_size == header.lowest_block_size {
            return;
        }
        let data_slice = unsafe {
            self.data.memory.get_u8_slice(
                header.address_space as usize,
                (header.pointer * header.type_size) as usize,
                (header.block_size * header.type_size) as usize,
            )
        };

        let record_mut = self
            .access_adapter_inventory
            .alloc_record(AccessLayout::from_record_header(&header));
        *record_mut.header = header;
        record_mut.data.copy_from_slice(data_slice);
        // we don't mind garbage values in prev_*
    }

    pub(crate) fn add_merge_record<T>(
        &mut self,
        header: AccessRecordHeader,
        data: &[T],
        prev_ts: &[u32],
    ) {
        if header.block_size == header.lowest_block_size {
            return;
        }

        let data_slice =
            unsafe { from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };

        let record_mut = self
            .access_adapter_inventory
            .alloc_record(AccessLayout::from_record_header(&header));
        *record_mut.header = header;
        record_mut.header.timestamp_and_mask |= MERGE_AND_NOT_SPLIT_FLAG;
        record_mut.data.copy_from_slice(data_slice);
        record_mut.timestamps.copy_from_slice(prev_ts);
    }

    fn split_by_meta<T: Copy>(
        &mut self,
        meta: &AccessMetadata,
        address_space: usize,
        lowest_block_size: usize,
    ) {
        if meta.block_size == lowest_block_size as u32 {
            return;
        }
        let begin = meta.start_ptr as usize / lowest_block_size;
        for i in 0..(meta.block_size as usize / lowest_block_size) {
            self.meta[address_space].set(
                begin + i,
                AccessMetadata {
                    start_ptr: (meta.start_ptr + (i * lowest_block_size) as u32),
                    block_size: lowest_block_size as u32,
                    timestamp: meta.timestamp,
                },
            );
        }
        self.add_split_record(AccessRecordHeader {
            timestamp_and_mask: meta.timestamp,
            address_space: address_space as u32,
            pointer: meta.start_ptr,
            block_size: meta.block_size,
            lowest_block_size: lowest_block_size as u32,
            type_size: size_of::<T>() as u32,
        });
    }

    /// Returns the timestamp of the previous access to `[pointer:BLOCK_SIZE]_{address_space}`
    /// and the offset of the record in bytes.
    ///
    /// Caller must ensure alignment (e.g. via `assert_alignment`) prior to calling this function.
    fn prev_access_time<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        align: usize,
        prev_values: &[T; BLOCK_SIZE],
    ) -> u32 {
        let num_segs = BLOCK_SIZE / align;

        let begin = pointer / align;

        let first_meta = self.meta[address_space].get(begin);
        let need_to_merge =
            first_meta.block_size != BLOCK_SIZE as u32 || first_meta.start_ptr != pointer as u32;
        let result = if need_to_merge {
            // Then we need to split everything we touched there
            // And add a merge record in the end
            let mut i = 0;
            while i < num_segs {
                let meta = self.meta[address_space].get(begin + i);
                if meta.block_size == 0 {
                    i += 1;
                    continue;
                }
                let meta = *meta;
                self.split_by_meta::<T>(&meta, address_space, align);
                i = (meta.start_ptr + meta.block_size) as usize / align - begin;
            }

            let prev_ts = (0..num_segs)
                .map(|i| {
                    let meta = self.meta[address_space].get(begin + i);
                    if meta.block_size > 0 {
                        meta.timestamp
                    } else {
                        // Initialize
                        if self.initial_block_size >= align {
                            // We need to split the initial block into chunks
                            let block_start = (begin + i) & !(self.initial_block_size / align - 1);
                            self.split_by_meta::<T>(
                                &AccessMetadata {
                                    start_ptr: (block_start * align) as u32,
                                    block_size: self.initial_block_size as u32,
                                    timestamp: INITIAL_TIMESTAMP,
                                },
                                address_space,
                                align,
                            );
                        } else {
                            debug_assert_eq!(self.initial_block_size, 1);
                            debug_assert!((address_space as u32) < NATIVE_AS); // TODO: normal way
                            self.add_merge_record::<u8>(
                                AccessRecordHeader {
                                    timestamp_and_mask: INITIAL_TIMESTAMP,
                                    address_space: address_space as u32,
                                    pointer: (pointer + i * align) as u32,
                                    block_size: align as u32,
                                    lowest_block_size: self.initial_block_size as u32,
                                    type_size: 1,
                                },
                                &vec![0; align], // TODO: not vec maybe
                                &vec![INITIAL_TIMESTAMP; align], // TODO: not vec maybe
                            );
                        }
                        INITIAL_TIMESTAMP
                    }
                })
                .collect::<Vec<_>>(); // TODO(AG): small buffer or small vec or something

            let timestamp = *prev_ts.iter().max().unwrap();
            self.add_merge_record(
                AccessRecordHeader {
                    timestamp_and_mask: timestamp,
                    address_space: address_space as u32,
                    pointer: pointer as u32,
                    block_size: BLOCK_SIZE as u32,
                    lowest_block_size: align as u32,
                    type_size: size_of::<T>() as u32,
                },
                prev_values,
                &prev_ts,
            );
            timestamp
        } else {
            first_meta.timestamp
        };
        self.set_meta_block(address_space, pointer, align, BLOCK_SIZE, self.timestamp);
        result
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
        values: [T; BLOCK_SIZE],
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
    pub fn touched_blocks(&self) -> Vec<(Address, AccessMetadata)> {
        assert_eq!(self.meta.len(), self.min_block_size.len());
        self.meta
            .par_iter()
            .zip(self.min_block_size.par_iter())
            .enumerate()
            .flat_map(|(addr_space, (page, &align))| {
                page.par_iter()
                    .filter_map(move |(idx, metadata)| {
                        let ptr = idx as u32 * align;
                        if ptr == metadata.start_ptr && metadata.block_size != 0 {
                            Some(((addr_space as u32, ptr), metadata))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::array;

    use openvm_stark_backend::p3_field::FieldAlgebra;
    use openvm_stark_sdk::utils::create_seeded_rng;
    use p3_baby_bear::BabyBear;
    use rand::Rng;

    use crate::arch::{testing::VmChipTestBuilder, MemoryConfig};

    type F = BabyBear;

    fn test_memory_write_by_tester(mut tester: VmChipTestBuilder<F>) {
        let mut rng = create_seeded_rng();

        // The point here is to have a lot of equal
        // and intersecting/overlapping blocks,
        // by limiting the space of valid pointers.
        let max_ptr = 20;
        let aligns = [4, 4, 4, 1];
        let value_bounds = [256, 256, 256, (1 << 30)];
        let max_log_block_size = 4;
        let its = 1000;
        for _ in 0..its {
            let addr_sp = rng.gen_range(1..=aligns.len());
            let align: usize = aligns[addr_sp - 1];
            let value_bound: u32 = value_bounds[addr_sp - 1];
            let ptr = rng.gen_range(0..max_ptr / align) * align;
            let log_len = rng.gen_range(align.trailing_zeros()..=max_log_block_size);
            match log_len {
                0 => tester.write::<1>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                1 => tester.write::<2>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                2 => tester.write::<4>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                3 => tester.write::<8>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                4 => tester.write::<16>(
                    addr_sp,
                    ptr,
                    array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..value_bound))),
                ),
                _ => unreachable!(),
            }
        }

        let tester = tester.build().finalize();
        tester.simple_test().expect("Verification failed");
    }

    #[test]
    fn test_memory_write_volatile() {
        test_memory_write_by_tester(VmChipTestBuilder::<F>::volatile(MemoryConfig::default()));
    }

    #[test]
    fn test_memory_write_persistent() {
        test_memory_write_by_tester(VmChipTestBuilder::<F>::persistent(MemoryConfig::default()));
    }
}
