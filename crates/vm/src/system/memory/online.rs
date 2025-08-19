use std::{
    array::from_fn,
    fmt::Debug,
    mem::MaybeUninit,
    num::NonZero,
    ptr::addr_of_mut,
    slice::from_raw_parts,
    sync::atomic::{AtomicUsize, Ordering},
};

use getset::Getters;
use itertools::zip_eq;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::*,
    p3_util::log2_strict_usize,
};
use tracing::instrument;

use crate::{
    arch::{
        AddressSpaceHostConfig, AddressSpaceHostLayout, DenseRecordArena, MemoryConfig,
        RecordArena, MAX_CELL_BYTE_SIZE,
    },
    system::{
        memory::{
            adapter::records::{AccessLayout, AccessRecordHeader, MERGE_AND_NOT_SPLIT_FLAG},
            MemoryAddress, TimestampedEquipartition, TimestampedValues, CHUNK,
        },
        TouchedMemory,
    },
    utils::slice_as_bytes,
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

use crate::{
    arch::CustomBorrow,
    system::memory::adapter::records::{size_by_layout, AccessRecordMut},
};

#[cfg(all(any(unix, windows), not(feature = "basic-memory")))]
pub type MemoryBackend = memmap::MmapMemory;
#[cfg(any(not(any(unix, windows)), feature = "basic-memory"))]
pub type MemoryBackend = basic::BasicMemory;

pub const INITIAL_TIMESTAMP: u32 = 0;
/// Default mmap page size. Change this if using THB.
pub const PAGE_SIZE: usize = 4096;

// Memory access constraints
const MAX_BLOCK_SIZE: usize = 32;
const MIN_ALIGN: usize = 1;
const MAX_SEGMENTS: usize = MAX_BLOCK_SIZE / MIN_ALIGN;

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
    /// Fill the memory with zeros.
    fn fill_zero(&mut self) {
        self.as_mut_slice().fill(0);
    }
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
    /// Underlying memory data.
    pub mem: Vec<M>,
    /// Host configuration for each address space.
    pub config: Vec<AddressSpaceHostConfig>,
}

impl Default for AddressMap {
    fn default() -> Self {
        Self::from_mem_config(&MemoryConfig::default())
    }
}

impl<M: LinearMemory> AddressMap<M> {
    pub fn new(config: Vec<AddressSpaceHostConfig>) -> Self {
        assert_eq!(config[0].num_cells, 0, "Address space 0 must have 0 cells");
        let mem = config
            .iter()
            .map(|config| M::new(config.num_cells.checked_mul(config.layout.size()).unwrap()))
            .collect();
        Self { mem, config }
    }

    pub fn from_mem_config(mem_config: &MemoryConfig) -> Self {
        Self::new(mem_config.addr_spaces.clone())
    }

    #[inline(always)]
    pub fn get_memory(&self) -> &Vec<M> {
        &self.mem
    }

    #[inline(always)]
    pub fn get_memory_mut(&mut self) -> &mut Vec<M> {
        &mut self.mem
    }

    /// Fill each address space memory with zeros. Does not change the config.
    #[inline]
    pub fn fill_zero(&mut self) {
        for mem in &mut self.mem {
            mem.fill_zero();
        }
    }

    /// # Safety
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    #[inline(always)]
    pub unsafe fn get_f<F: PrimeField32>(&self, addr_space: u32, ptr: u32) -> F {
        let layout = &self.config.get_unchecked(addr_space as usize).layout;
        let start = ptr as usize * layout.size();
        let bytes = self.get_u8_slice(addr_space, start, layout.size());
        layout.to_field(bytes)
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    #[inline(always)]
    pub unsafe fn get<T: Copy>(&self, (addr_space, ptr): Address) -> T {
        debug_assert_eq!(
            size_of::<T>(),
            self.config[addr_space as usize].layout.size()
        );
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
    #[inline(always)]
    pub unsafe fn get_slice<T: Copy>(&self, (addr_space, ptr): Address, len: usize) -> &[T] {
        debug_assert_eq!(
            size_of::<T>(),
            self.config[addr_space as usize].layout.size()
        );
        let start = (ptr as usize) * size_of::<T>();
        let mem = self.mem.get_unchecked(addr_space as usize);
        // SAFETY:
        // - alignment is automatic since we multiply by `size_of::<T>()`
        mem.get_aligned_slice(start, len)
    }

    /// Reads the slice at **byte** addresses `start..start + len` from address space `addr_space`
    /// linear memory. Panics or segfaults if `start..start + len` is out of bounds
    ///
    /// # Safety
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    #[inline(always)]
    pub unsafe fn get_u8_slice(&self, addr_space: u32, start: usize, len: usize) -> &[u8] {
        let mem = self.mem.get_unchecked(addr_space as usize);
        mem.get_aligned_slice(start, len)
    }

    /// Copies `data` into the memory at `(addr_space, ptr)`.
    ///
    /// Panics or segfaults if `ptr + size_of_val(data)` is out of bounds.
    ///
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - The linear memory in `addr_space` is aligned to `T`.
    #[inline(always)]
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
    pub fn set_from_sparse(&mut self, sparse_map: &SparseMemoryImage) {
        for (&(addr_space, index), &data_byte) in sparse_map.iter() {
            // SAFETY:
            // - safety assumptions in function doc comments
            unsafe {
                self.mem
                    .get_unchecked_mut(addr_space as usize)
                    .write_unaligned(index as usize, data_byte);
            }
        }
    }
}

/// API for guest memory conforming to OpenVM ISA
// @dev Note we don't make this a trait because phantom executors currently need a concrete type for
// guest memory
#[derive(Debug, Clone)]
pub struct GuestMemory {
    pub memory: AddressMap,
}

impl GuestMemory {
    pub fn new(addr: AddressMap) -> Self {
        Self { memory: addr }
    }

    /// Returns `[pointer:BLOCK_SIZE]_{address_space}`
    ///
    /// # Safety
    /// The type `T` must be stack-allocated `repr(C)` or `repr(transparent)`,
    /// and it must be the exact type used to represent a single memory cell in
    /// address space `address_space`. For standard usage,
    /// `T` is either `u8` or `F` where `F` is the base field of the ZK backend.
    #[inline(always)]
    pub unsafe fn read<T, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE]
    where
        T: Copy + Debug,
    {
        self.debug_assert_cell_type::<T>(addr_space);
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
    #[inline(always)]
    pub unsafe fn write<T, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        values: [T; BLOCK_SIZE],
    ) where
        T: Copy + Debug,
    {
        self.debug_assert_cell_type::<T>(addr_space);
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
        self.debug_assert_cell_type::<T>(addr_space);
        // SAFETY:
        // - alignment for `[T; BLOCK_SIZE]` is automatic since we multiply by `size_of::<T>()`
        self.memory
            .get_memory_mut()
            .get_unchecked_mut(addr_space as usize)
            .swap((ptr as usize) * size_of::<T>(), values);
    }

    /// Panics or segfaults if `ptr..ptr + len` is out of bounds
    ///
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    #[inline(always)]
    pub unsafe fn get_slice<T: Copy>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        self.memory.get_slice((addr_space, ptr), len)
    }

    #[inline(always)]
    pub fn get_min_block_size(&self, addr_space: u32) -> usize {
        self.memory.config[addr_space as usize].min_block_size
    }

    #[inline(always)]
    fn debug_assert_cell_type<T>(&self, addr_space: u32) {
        debug_assert_eq!(
            size_of::<T>(),
            self.memory.config[addr_space as usize].layout.size()
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AccessMetadata {
    /// Packed timestamp (29 bits) and log2(block_size) (3 bits)
    pub timestamp_and_log_block_size: u32,
    /// Offset to block start (in ALIGN units).
    pub offset_to_start: u8,
}

impl AccessMetadata {
    const TIMESTAMP_MASK: u32 = (1 << 29) - 1;
    const LOG_BLOCK_SIZE_SHIFT: u32 = 29;

    pub fn new(timestamp: u32, block_size: u8, offset_to_start: u8) -> Self {
        debug_assert!(timestamp < (1 << 29), "Timestamp must be less than 2^29");
        debug_assert!(
            block_size == 0 || (block_size.is_power_of_two() && block_size <= MAX_BLOCK_SIZE as u8),
            "Block size must be 0 or power of 2 and <= {}",
            MAX_BLOCK_SIZE
        );

        let encoded_block_size = if block_size == 0 {
            0
        } else {
            // SAFETY: We already asserted that block_size is non-zero in this branch
            unsafe { NonZero::new_unchecked(block_size) }.ilog2() + 1
        };
        let packed = timestamp | (encoded_block_size << Self::LOG_BLOCK_SIZE_SHIFT);

        Self {
            timestamp_and_log_block_size: packed,
            offset_to_start,
        }
    }

    #[inline(always)]
    pub fn timestamp(&self) -> u32 {
        self.timestamp_and_log_block_size & Self::TIMESTAMP_MASK
    }

    #[inline(always)]
    pub fn block_size(&self) -> u8 {
        let encoded = self.timestamp_and_log_block_size >> Self::LOG_BLOCK_SIZE_SHIFT;
        if encoded == 0 {
            0
        } else {
            1 << (encoded - 1)
        }
    }
}

/// Online memory that stores additional information for trace generation purposes.
/// In particular, keeps track of timestamp.
#[derive(Getters)]
pub struct TracingMemory {
    pub timestamp: u32,
    /// The initial block size -- this depends on the type of boundary chip.
    initial_block_size: usize,
    /// The underlying data memory, with memory cells typed by address space: see [AddressMap].
    #[getset(get = "pub")]
    pub data: GuestMemory,
    /// Maps addr_space to (ptr / min_block_size[addr_space] -> AccessMetadata) for latest access
    /// metadata. Uses paged storage for memory efficiency. AccessMetadata stores offset_to_start
    /// (in ALIGN units), block_size, and timestamp (latter two only valid at offset_to_start ==
    /// 0).
    pub(super) meta: Vec<PagedVec<AccessMetadata, PAGE_SIZE>>,
    /// For each `addr_space`, the minimum block size allowed for memory accesses. In other words,
    /// all memory accesses in `addr_space` must be aligned to this block size.
    pub min_block_size: Vec<u32>,
    pub access_adapter_records: DenseRecordArena,
}

// min_block_size * cell_size never exceeds 8
const INITIAL_CELL_BUFFER: &[u8] = &[0u8; 8];
// min_block_size never exceeds 8
const INITIAL_TIMESTAMP_BUFFER: &[u32] = &[INITIAL_TIMESTAMP; 8];

#[derive(Clone, Copy)]
struct SplitData {
    timestamp: u32,
    block_size: u8,
}

#[derive(Clone, Copy)]
struct AccessSplitRecord<const MAX_SEG: usize> {
    start_ptr: u32,
    // Caller could know `split_data` is used up when start_ptr moves out
    // of range.
    split_data: [SplitData; MAX_SEG],
    /// Length of `split_data`. It is only safe to access `split_data[..len]`.
    len: usize,
}

impl TracingMemory {
    pub fn new(
        mem_config: &MemoryConfig,
        initial_block_size: usize,
        access_adapter_arena_size_bound: usize,
    ) -> Self {
        let image = GuestMemory::new(AddressMap::from_mem_config(mem_config));
        Self::from_image(image, initial_block_size, access_adapter_arena_size_bound)
    }

    /// Constructor from pre-existing memory image.
    pub fn from_image(
        image: GuestMemory,
        initial_block_size: usize,
        access_adapter_arena_size_bound: usize,
    ) -> Self {
        let (meta, min_block_size): (Vec<_>, Vec<_>) =
            zip_eq(image.memory.get_memory(), &image.memory.config)
                .map(|(mem, addr_sp)| {
                    let num_cells = mem.size() / addr_sp.layout.size();
                    let min_block_size = addr_sp.min_block_size;
                    let total_metadata_len = num_cells.div_ceil(min_block_size);
                    (PagedVec::new(total_metadata_len), min_block_size as u32)
                })
                .unzip();
        let access_adapter_records =
            DenseRecordArena::with_byte_capacity(access_adapter_arena_size_bound);
        Self {
            data: image,
            meta,
            min_block_size,
            timestamp: INITIAL_TIMESTAMP + 1,
            initial_block_size,
            access_adapter_records,
        }
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

    /// Get block metadata by jumping to the start of the block.
    /// Returns (block_start_pointer, block_metadata).
    #[inline(always)]
    fn get_block_metadata<const ALIGN: usize>(
        &self,
        address_space: usize,
        pointer: usize,
    ) -> (u32, AccessMetadata) {
        let ptr_index = pointer / ALIGN;
        let meta_page = unsafe { self.meta.get_unchecked(address_space) };
        let current_meta = meta_page.get(ptr_index);

        let (block_start_index, block_metadata) = if current_meta.offset_to_start == 0 {
            (ptr_index, current_meta)
        } else {
            let offset = current_meta.offset_to_start;
            let start_idx = ptr_index - offset as usize;
            let start_meta = meta_page.get(start_idx);
            (start_idx, start_meta)
        };

        let block_start_pointer = (block_start_index * ALIGN) as u32;

        (block_start_pointer, block_metadata)
    }

    /// Updates the metadata with the given block.
    /// Stores timestamp and block_size only at block start, offsets elsewhere.
    #[inline(always)]
    fn set_meta_block<const BLOCK_SIZE: usize, const ALIGN: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        timestamp: u32,
    ) {
        let ptr = pointer / ALIGN;
        // SAFETY: address_space is assumed to be valid and within bounds
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };

        // Store full metadata at the block start
        meta_page.set(ptr, AccessMetadata::new(timestamp, BLOCK_SIZE as u8, 0));

        // Store offsets for other positions in the block
        for i in 1..(BLOCK_SIZE / ALIGN) {
            meta_page.set(ptr + i, AccessMetadata::new(0, 0, i as u8));
        }
    }

    #[inline(always)]
    pub(crate) fn add_split_record(&mut self, header: AccessRecordHeader) {
        if header.block_size == header.lowest_block_size {
            return;
        }
        let data_slice = unsafe {
            self.data.memory.get_u8_slice(
                header.address_space,
                (header.pointer * header.type_size) as usize,
                (header.block_size * header.type_size) as usize,
            )
        };

        let record_mut = self
            .access_adapter_records
            .alloc(AccessLayout::from_record_header(&header));
        *record_mut.header = header;
        record_mut.data.copy_from_slice(data_slice);
        // we don't mind garbage values in prev_*
    }

    /// `data_slice` is the underlying data of the record in raw host memory format.
    #[inline(always)]
    pub(crate) fn add_merge_record(
        &mut self,
        header: AccessRecordHeader,
        data_slice: &[u8],
        prev_ts: &[u32],
    ) {
        debug_assert_eq!(header.block_size % header.lowest_block_size, 0);
        debug_assert_eq!(
            prev_ts.len() as u32,
            header.block_size / header.lowest_block_size
        );
        debug_assert_eq!(
            data_slice.len() as u32,
            header.block_size * header.type_size
        );
        if header.block_size == header.lowest_block_size {
            // no merge required
            return;
        }

        let record_mut = self
            .access_adapter_records
            .alloc(AccessLayout::from_record_header(&header));
        *record_mut.header = header;
        record_mut.header.timestamp_and_mask |= MERGE_AND_NOT_SPLIT_FLAG;
        record_mut.data.copy_from_slice(data_slice);
        record_mut.timestamps.copy_from_slice(prev_ts);
    }

    #[inline(always)]
    fn split_by_meta<T: Copy, const MIN_BLOCK_SIZE: usize>(
        &mut self,
        start_ptr: u32,
        timestamp: u32,
        block_size: u8,
        address_space: usize,
    ) {
        if block_size == MIN_BLOCK_SIZE as u8 {
            return;
        }
        let begin = start_ptr as usize / MIN_BLOCK_SIZE;
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };

        for i in 0..(block_size as usize / MIN_BLOCK_SIZE) {
            // Each split piece becomes its own block start
            meta_page.set(
                begin + i,
                AccessMetadata::new(timestamp, MIN_BLOCK_SIZE as u8, 0),
            );
        }
        self.add_split_record(AccessRecordHeader {
            timestamp_and_mask: timestamp,
            address_space: address_space as u32,
            pointer: start_ptr,
            block_size: block_size as u32,
            lowest_block_size: MIN_BLOCK_SIZE as u32,
            type_size: size_of::<T>() as u32,
        });
    }

    /// Returns the timestamp of the previous access to `[pointer:BLOCK_SIZE]_{address_space}`.
    /// This function also updates the access metadata such that the last access to `(address_space,
    /// pointer)` is of block size `BLOCK_SIZE`. This is done by splitting previous accesses
    /// that overlap with `[pointer:BLOCK_SIZE]_{address_space}` into accesses of size `ALIGN`
    /// and then merging the blocks in `pointer..pointer + BLOCK_SIZE` into a single block.
    ///
    /// # Assumptions
    /// The constant generic `ALIGN` must be the minimum block size for the given `address_space`.
    /// This is a constant for performance reasons.
    ///
    /// The constant generic `BLOCK_SIZE` must be a multiple of `ALIGN`.
    ///
    /// The `pointer` must be a multiple of `ALIGN`.
    /// The caller must ensure alignment of `pointer` (e.g. via `assert_alignment`) prior to calling
    /// this function.
    ///
    /// # Safety
    /// The `address_space` must be known to be in bounds.
    #[inline(always)]
    fn prev_access_time<T: Copy, const BLOCK_SIZE: usize, const ALIGN: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        prev_values: &[T; BLOCK_SIZE],
    ) -> u32 {
        debug_assert_eq!(ALIGN, self.data.get_min_block_size(address_space as u32));
        self.data.debug_assert_cell_type::<T>(address_space as u32);
        // Calculate what splits are needed for this memory access. We always merge after splits.
        //
        // IMPORTANT: this function is very perf sensitive, and we want to skip all split and merge
        // operations if the previous block access exactly matches `(BLOCK_SIZE, pointer)`.
        let ptr_index = pointer / ALIGN;
        // SAFETY: assumes that `address_space` is valid
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };
        let current_meta = meta_page.get_mut(ptr_index);
        if current_meta.block_size() == BLOCK_SIZE as u8 && current_meta.offset_to_start == 0 {
            // The fast branch where no split+merges are needed
            let prev_t = current_meta.timestamp();
            debug_assert!(
                self.timestamp < (1 << 29),
                "Timestamp must be less than 2^29"
            );
            // The log_block_size doesn't change, so only update timestamp
            current_meta.timestamp_and_log_block_size &= !AccessMetadata::TIMESTAMP_MASK;
            current_meta.timestamp_and_log_block_size |= self.timestamp;
            return prev_t;
        }

        // Split intersecting blocks to align bytes
        let mut prev_ts_buf: [MaybeUninit<u32>; MAX_SEGMENTS] =
            [MaybeUninit::uninit(); MAX_SEGMENTS];
        // split_idx is the index of the after-split small block of size ALIGN
        let mut split_idx = 0;
        let mut max_timestamp = INITIAL_TIMESTAMP;
        let mut current_ptr = pointer;
        let end_ptr = pointer + BLOCK_SIZE;
        let mut push_ts = |ts: u32| {
            unsafe {
                let _ = *prev_ts_buf.get_unchecked_mut(split_idx).write(ts);
            }
            split_idx += 1;
        };
        while current_ptr < end_ptr {
            let (curr_start_ptr, curr_block_meta) =
                self.get_block_metadata::<ALIGN>(address_space, current_ptr);

            let block_size = curr_block_meta.block_size();
            // SAFETY: seg_idx < MAX_SEGMENTS since we iterate at most merge_size/ALIGN times
            // and merge_size <= BLOCK_SIZE <= MAX_BLOCK_SIZE
            if block_size == 0 {
                let curr_start_ptr =
                    self.handle_uninitialized_memory::<T, ALIGN>(address_space, current_ptr as u32);
                let block_size = self.initial_block_size.max(ALIGN) as u32;
                let curr_end_ptr = curr_start_ptr + block_size;
                let num_ts = (curr_end_ptr.min(end_ptr as u32) - current_ptr as u32) / ALIGN as u32;
                for _ in 0..num_ts {
                    push_ts(INITIAL_TIMESTAMP);
                }
                current_ptr = curr_end_ptr as usize;
            } else {
                let timestamp = curr_block_meta.timestamp();
                if block_size > ALIGN as u8 {
                    self.split_by_meta::<T, ALIGN>(
                        curr_start_ptr,
                        timestamp,
                        block_size,
                        address_space,
                    );
                }
                let curr_end_ptr = curr_start_ptr as usize + block_size as usize;
                while current_ptr < curr_end_ptr && current_ptr < end_ptr {
                    push_ts(timestamp);
                    current_ptr += ALIGN;
                }
                max_timestamp = max_timestamp.max(timestamp);
            }
        }
        let prev_ts_buf: [u32; MAX_SEGMENTS] = unsafe { std::mem::transmute(prev_ts_buf) };
        // Create the merge record
        self.add_merge_record(
            AccessRecordHeader {
                timestamp_and_mask: max_timestamp,
                address_space: address_space as u32,
                pointer: pointer as u32,
                block_size: BLOCK_SIZE as u32,
                lowest_block_size: ALIGN as u32,
                type_size: size_of::<T>() as u32,
            },
            // SAFETY: T is plain old data
            unsafe { slice_as_bytes(prev_values) },
            &prev_ts_buf[..split_idx],
        );

        // Update the metadata for this access
        self.set_meta_block::<BLOCK_SIZE, ALIGN>(address_space, pointer, self.timestamp);
        max_timestamp
    }

    /// Handle uninitialized memory by creating appropriate split or merge records.
    #[inline(always)]
    fn handle_uninitialized_memory<T: Copy, const ALIGN: usize>(
        &mut self,
        address_space: usize,
        pointer: u32,
    ) -> u32 {
        if self.initial_block_size >= ALIGN {
            // Split the initial block into chunks
            let start_ptr = self.uninitialized_memory_chunk_start::<ALIGN>(pointer);
            self.split_by_meta::<T, ALIGN>(
                start_ptr,
                INITIAL_TIMESTAMP,
                self.initial_block_size as u8,
                address_space,
            );
            start_ptr
        } else {
            // Create a merge record for single-byte initialization
            debug_assert_eq!(self.initial_block_size, 1);
            self.add_merge_record(
                AccessRecordHeader {
                    timestamp_and_mask: INITIAL_TIMESTAMP,
                    address_space: address_space as u32,
                    pointer,
                    block_size: ALIGN as u32,
                    lowest_block_size: self.initial_block_size as u32,
                    type_size: size_of::<T>() as u32,
                },
                &INITIAL_CELL_BUFFER[..ALIGN],
                &INITIAL_TIMESTAMP_BUFFER[..ALIGN],
            );
            pointer
        }
    }

    #[inline(always)]
    fn uninitialized_memory_chunk_start<const ALIGN: usize>(&self, pointer: u32) -> u32 {
        if self.initial_block_size >= ALIGN {
            // Split the initial block into chunks
            let segment_index = pointer as usize / ALIGN;
            let block_start = segment_index & !(self.initial_block_size / ALIGN - 1);
            (block_start * ALIGN) as u32
        } else {
            debug_assert_eq!(self.initial_block_size, 1);
            pointer
        }
    }

    #[inline(always)]
    fn compute_access_record<T: Copy, const BLOCK_SIZE: usize, const ALIGN: usize>(
        &self,
        address_space: u32,
        pointer: u32,
    ) -> AccessSplitRecord<BLOCK_SIZE> {
        debug_assert_eq!(ALIGN, self.data.get_min_block_size(address_space));
        self.data.debug_assert_cell_type::<T>(address_space);
        let mut ret = MaybeUninit::<AccessSplitRecord<BLOCK_SIZE>>::uninit();
        let ret_ptr = ret.as_mut_ptr();
        let mut max_timestamp = INITIAL_TIMESTAMP;
        let mut current_ptr = pointer;
        let end_ptr = pointer + BLOCK_SIZE as u32;
        let mut seg_idx = 0;
        while current_ptr < end_ptr {
            let (mut curr_start_ptr, curr_block_meta) =
                self.get_block_metadata::<ALIGN>(address_space as usize, current_ptr as usize);
            let mut block_size = curr_block_meta.block_size();
            if block_size == 0 {
                curr_start_ptr = self.uninitialized_memory_chunk_start::<ALIGN>(pointer);
                block_size = self.initial_block_size.max(ALIGN) as u8;
            }
            if current_ptr == pointer {
                unsafe {
                    addr_of_mut!((*ret_ptr).start_ptr).write(curr_start_ptr);
                }
            }
            current_ptr = curr_start_ptr + block_size as u32;
            unsafe {
                addr_of_mut!((*ret_ptr).split_data[seg_idx].block_size).write(block_size);
            }
            if block_size == 0 {
                unsafe {
                    addr_of_mut!((*ret_ptr).split_data[seg_idx].timestamp).write(INITIAL_TIMESTAMP);
                }
            } else {
                let timestamp = curr_block_meta.timestamp();
                unsafe {
                    addr_of_mut!((*ret_ptr).split_data[seg_idx].timestamp).write(timestamp);
                }
                max_timestamp = max_timestamp.max(timestamp);
            }
            seg_idx += 1;
        }
        // SAFETY: we have written to start_ptr, split_data up to seg_idx, and now len
        // - split_data[len..] is still uninit, and we guarantee we never read from it.
        unsafe {
            addr_of_mut!((*ret_ptr).len).write(seg_idx);
            ret.assume_init()
        }
    }

    /// Performs batch getting of the previous access times of a slice in guest memory in address
    /// space `address_space` starting at `pointer` of length `access_timestamp.len()` elements of
    /// type `T`.
    ///
    /// This function does not mutate the internal timestamp. It
    /// takes in the `access_timestamp`s to update this slice's metadata to as a separate argument.
    /// The `access_timestamp` is a closure that takes `i = 0..len` and outputs the new access
    /// timestamp for the `i`th block.
    ///
    /// Returns a vector of length `len` with the previous access timestamps of
    /// each block. Note that if access adapter split and merges were necessary, then the previous
    /// access timestamp is the access timestamp after all split and merges were performed.
    ///
    /// Note that this function does not mutate `self.data`.
    pub fn batch_prev_access_times<
        T: Copy + Send + Sync,
        const BLOCK_SIZE: usize,
        const ALIGN: usize,
        FN: (Fn(usize) -> u32) + Sync,
    >(
        &mut self,
        address_space: u32,
        pointer: u32,
        len: usize,
        // Access timestamp for each block. Also `access_timestamp.len()` indicates the number of
        // accesses.
        access_timestamp: FN,
    ) -> Vec<u32> {
        // Goal: read a batch of memory with the same `BLOCK_SIZE` at given the timestamps.
        // High-level overview:
        // The batch read splits memory into the following segments:
        // | prev | block 1 | ... | block n | succ |
        // `prev` and `succ` are split from the existing memory segments (which could be
        // uninitialized). `block 1` to `block n` split the existing memory segments and
        // merge `BLOCK_SIZE` cells into one segment.
        //
        // First, we need to know the related segments of each block. We could use the start pointer
        // of each block to find the first overlapped segment, then find all the segments that
        // overlap with the block. This operation is read-only so it's thread-safe.
        //
        // This function acquires a mutable reference to `access_adapter_records`, and then within
        // this function we use an `AtomicU64` to update the access adapter records in parallel in a
        // thread-safe manner.
        //
        // We want to insert the split access adapter record. Here we parallelize
        // by blocks. Because 2 adjacent blocks could have at most 1 overlapped segment. We
        // always let the left block to add the access adapter record for the overlapped
        // segment.
        //
        // The next step is to update the access metadata. Each block is responsible for updating
        // the access metadata in its own block. `prev`/`succ` are handled by `block 1`/`block n`.
        //
        // Finally, we need to add the merge access adapter records. Each block is responsible for
        // adding its own merge access adapter record. Again, `access_adapter_records` is
        // thread-safe so this is OK.
        // Touch all related pages to avoid racing page allocation.
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space as usize) };
        let start = pointer.saturating_sub(MAX_BLOCK_SIZE as u32);
        let batch_end = pointer + len as u32 * BLOCK_SIZE as u32;
        let end = batch_end + MAX_BLOCK_SIZE as u32 - 1;
        meta_page.par_touch(start as usize, end as usize);
        let batch_processor = BatchReadProcessor::<T>::new(self, address_space);
        (0..len)
            .into_par_iter()
            .map(|i| {
                let block_start_ptr = pointer + i as u32 * BLOCK_SIZE as u32;
                let access_t = access_timestamp(i);
                let (start_ptr, start_meta) = self
                    .get_block_metadata::<ALIGN>(address_space as usize, block_start_ptr as usize);
                if start_ptr == block_start_ptr && start_meta.block_size() == BLOCK_SIZE as u8 {
                    let prev_t = start_meta.timestamp(); // no split or merges needed
                    unsafe {
                        batch_processor
                            .set_meta_block::<BLOCK_SIZE, ALIGN>(start_ptr as usize, access_t);
                    }
                    return prev_t;
                }
                let AccessSplitRecord {
                    start_ptr,
                    split_data,
                    len: num_segs,
                } = self
                    .compute_access_record::<T, BLOCK_SIZE, ALIGN>(address_space, block_start_ptr);
                // Edge case: the first block is responsible for the first out-of-bound segment.
                if i == 0 && start_ptr < pointer {
                    unsafe {
                        batch_processor.set_timestamp::<ALIGN>(
                            start_ptr,
                            pointer,
                            split_data[0].timestamp,
                        );
                    }
                }

                let block_end_ptr = pointer + (i as u32 + 1) * BLOCK_SIZE as u32;
                let mut curr_ptr = start_ptr;
                let mut seg_idx = 0;

                let mut prev_ts_buf = [INITIAL_TIMESTAMP; MAX_SEGMENTS];
                let mut split_idx = 0;
                let mut max_timestamp = INITIAL_TIMESTAMP;
                let mut push_ts = |ts: u32| {
                    prev_ts_buf[split_idx] = ts;
                    split_idx += 1;
                };
                while curr_ptr < block_end_ptr {
                    let SplitData {
                        timestamp,
                        mut block_size,
                    } = split_data[seg_idx];
                    if curr_ptr >= block_start_ptr || i == 0 {
                        let mut add_split = true;
                        if block_size == 0 {
                            // In the merge case, all cells in the segment must be in the block.
                            if self.initial_block_size < ALIGN {
                                // Create a merge record for single-byte initialization
                                debug_assert_eq!(self.initial_block_size, 1);
                                batch_processor.add_merge_record(
                                    AccessRecordHeader {
                                        timestamp_and_mask: INITIAL_TIMESTAMP,
                                        address_space,
                                        pointer: curr_ptr,
                                        block_size: ALIGN as u32,
                                        lowest_block_size: self.initial_block_size as u32,
                                        type_size: size_of::<T>() as u32,
                                    },
                                    &INITIAL_TIMESTAMP_BUFFER[..ALIGN],
                                );
                                block_size = ALIGN as u8;
                                add_split = false;
                                push_ts(INITIAL_TIMESTAMP);
                                curr_ptr += block_size as u32;
                            } else {
                                block_size = self.initial_block_size as u8;
                            }
                        }
                        if add_split {
                            batch_processor.add_split_record(AccessRecordHeader {
                                timestamp_and_mask: timestamp,
                                address_space,
                                pointer: curr_ptr,
                                block_size: block_size as u32,
                                lowest_block_size: ALIGN as u32,
                                type_size: size_of::<T>() as u32,
                            });
                            let curr_end_ptr = curr_ptr + block_size as u32;
                            let num_ts =
                                (curr_end_ptr.min(block_end_ptr) - curr_ptr) / ALIGN as u32;
                            for _ in 0..num_ts {
                                push_ts(timestamp);
                            }
                            curr_ptr += block_size as u32;
                            max_timestamp = max_timestamp.max(timestamp);
                        }
                        // No need to update the metadata of cells in the block because we will
                        // merge them later.
                    }
                    seg_idx += 1;
                }
                debug_assert_eq!(seg_idx, num_segs);
                // Create the merge record
                unsafe {
                    batch_processor.add_merge_record(
                        AccessRecordHeader {
                            timestamp_and_mask: max_timestamp,
                            address_space,
                            pointer: block_start_ptr,
                            block_size: BLOCK_SIZE as u32,
                            lowest_block_size: ALIGN as u32,
                            type_size: size_of::<T>() as u32,
                        },
                        &prev_ts_buf[..split_idx],
                    );
                    batch_processor
                        .set_meta_block::<BLOCK_SIZE, ALIGN>(block_start_ptr as usize, access_t);
                }

                // Edge case: the last block is responsible for the last out-of-bounds segment
                if i + 1 == len && curr_ptr > block_end_ptr {
                    unsafe {
                        batch_processor.set_timestamp::<ALIGN>(
                            block_end_ptr,
                            curr_ptr,
                            split_data[seg_idx - 1].timestamp,
                        );
                    }
                }
                max_timestamp
            })
            .collect()
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
    /// plain old data, and it must be the exact type used to represent a single memory cell in
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
        let t_prev = self.prev_access_time::<T, BLOCK_SIZE, ALIGN>(
            address_space as usize,
            pointer as usize,
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
        let t_prev = self.prev_access_time::<T, BLOCK_SIZE, ALIGN>(
            address_space as usize,
            pointer as usize,
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

    /// Finalize the boundary and merkle chips.
    #[instrument(name = "memory_finalize", skip_all)]
    pub fn finalize<F: Field>(&mut self, is_persistent: bool) -> TouchedMemory<F> {
        let touched_blocks = self.touched_blocks();

        match is_persistent {
            false => TouchedMemory::Volatile(
                self.touched_blocks_to_equipartition::<F, 1>(touched_blocks),
            ),
            true => TouchedMemory::Persistent(
                self.touched_blocks_to_equipartition::<F, CHUNK>(touched_blocks),
            ),
        }
    }

    /// Returns the list of all touched blocks. The list is sorted by address.
    fn touched_blocks(&self) -> Vec<(Address, AccessMetadata)> {
        assert_eq!(self.meta.len(), self.min_block_size.len());
        self.meta
            .par_iter()
            .zip(self.min_block_size.par_iter())
            .enumerate()
            .flat_map(|(addr_space, (meta_page, &align))| {
                meta_page
                    .par_iter()
                    .filter_map(move |(idx, metadata)| {
                        let ptr = idx as u32 * align;
                        if metadata.offset_to_start == 0 && metadata.block_size() != 0 {
                            Some(((addr_space as u32, ptr), metadata))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Returns the equipartition of the touched blocks.
    /// Modifies records and adds new to account for the initial/final segments.
    fn touched_blocks_to_equipartition<F: Field, const CHUNK: usize>(
        &mut self,
        touched_blocks: Vec<((u32, u32), AccessMetadata)>,
    ) -> TimestampedEquipartition<F, CHUNK> {
        // [perf] We can `.with_capacity()` if we keep track of the number of segments we initialize
        let mut final_memory = Vec::new();

        debug_assert!(touched_blocks.is_sorted_by_key(|(addr, _)| addr));
        self.handle_touched_blocks::<F, CHUNK>(&mut final_memory, touched_blocks);

        debug_assert!(final_memory.is_sorted_by_key(|(key, _)| *key));
        final_memory
    }

    fn handle_touched_blocks<F: Field, const CHUNK: usize>(
        &mut self,
        final_memory: &mut Vec<((u32, u32), TimestampedValues<F, CHUNK>)>,
        touched_blocks: Vec<((u32, u32), AccessMetadata)>,
    ) {
        let mut current_values = vec![0u8; MAX_CELL_BYTE_SIZE * CHUNK];
        let mut current_cnt = 0;
        let mut current_address = MemoryAddress::new(0, 0);
        let mut current_timestamps = vec![0; CHUNK];
        for ((addr_space, ptr), access_metadata) in touched_blocks {
            // SAFETY: addr_space of touched blocks are all in bounds
            let addr_space_config =
                unsafe { *self.data.memory.config.get_unchecked(addr_space as usize) };
            let min_block_size = addr_space_config.min_block_size;
            let cell_size = addr_space_config.layout.size();
            let timestamp = access_metadata.timestamp();
            let block_size = access_metadata.block_size();
            assert!(
                current_cnt == 0
                    || (current_address.address_space == addr_space
                    && current_address.pointer + current_cnt as u32 == ptr),
                "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`"
            );
            debug_assert!(block_size >= min_block_size as u8);
            debug_assert!(ptr % min_block_size as u32 == 0);

            if current_cnt == 0 {
                assert_eq!(
                    ptr & (CHUNK as u32 - 1),
                    0,
                    "The union of all touched blocks must consist of `CHUNK`-aligned blocks"
                );
                current_address = MemoryAddress::new(addr_space, ptr);
            }

            if block_size > min_block_size as u8 {
                self.add_split_record(AccessRecordHeader {
                    timestamp_and_mask: timestamp,
                    address_space: addr_space,
                    pointer: ptr,
                    block_size: block_size as u32,
                    lowest_block_size: min_block_size as u32,
                    type_size: cell_size as u32,
                });
            }
            if min_block_size > CHUNK {
                assert_eq!(current_cnt, 0);
                for i in (0..block_size as u32).step_by(min_block_size) {
                    self.add_split_record(AccessRecordHeader {
                        timestamp_and_mask: timestamp,
                        address_space: addr_space,
                        pointer: ptr + i,
                        block_size: min_block_size as u32,
                        lowest_block_size: CHUNK as u32,
                        type_size: cell_size as u32,
                    });
                }
                // SAFETY: touched blocks are in bounds
                let values = unsafe {
                    self.data.memory.get_u8_slice(
                        addr_space,
                        ptr as usize * cell_size,
                        block_size as usize * cell_size,
                    )
                };
                for i in (0..block_size as u32).step_by(CHUNK) {
                    final_memory.push((
                        (addr_space, ptr + i),
                        TimestampedValues {
                            timestamp,
                            values: from_fn(|j| {
                                let byte_idx = (i as usize + j) * cell_size;
                                // SAFETY: block_size is multiple of CHUNK and we are reading chunks
                                // of cells within bounds
                                unsafe {
                                    addr_space_config
                                        .layout
                                        .to_field(&values[byte_idx..byte_idx + cell_size])
                                }
                            }),
                        },
                    ));
                }
            } else {
                for i in 0..block_size as u32 {
                    // SAFETY: getting cell data
                    let cell_data = unsafe {
                        self.data.memory.get_u8_slice(
                            addr_space,
                            (ptr + i) as usize * cell_size,
                            cell_size,
                        )
                    };
                    current_values[current_cnt * cell_size..current_cnt * cell_size + cell_size]
                        .copy_from_slice(cell_data);
                    if current_cnt & (min_block_size - 1) == 0 {
                        // SAFETY: current_cnt / min_block_size < CHUNK / min_block_size <= CHUNK
                        unsafe {
                            *current_timestamps.get_unchecked_mut(current_cnt / min_block_size) =
                                timestamp;
                        }
                    }
                    current_cnt += 1;
                    if current_cnt == CHUNK {
                        let timestamp = *current_timestamps[..CHUNK / min_block_size]
                            .iter()
                            .max()
                            .unwrap();

                        self.add_merge_record(
                            AccessRecordHeader {
                                timestamp_and_mask: timestamp,
                                address_space: addr_space,
                                pointer: current_address.pointer,
                                block_size: CHUNK as u32,
                                lowest_block_size: min_block_size as u32,
                                type_size: cell_size as u32,
                            },
                            &current_values[..CHUNK * cell_size],
                            &current_timestamps[..CHUNK / min_block_size],
                        );
                        final_memory.push((
                            (current_address.address_space, current_address.pointer),
                            TimestampedValues {
                                timestamp,
                                values: from_fn(|i| unsafe {
                                    // SAFETY: cell_size is correct, and alignment is guaranteed
                                    addr_space_config.layout.to_field(
                                        &current_values[i * cell_size..i * cell_size + cell_size],
                                    )
                                }),
                            },
                        ));
                        current_address.pointer += current_cnt as u32;
                        current_cnt = 0;
                    }
                }
            }
        }
        assert_eq!(current_cnt, 0, "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`");
    }

    pub fn address_space_alignment(&self) -> Vec<u8> {
        self.min_block_size
            .iter()
            .map(|&x| log2_strict_usize(x as usize) as u8)
            .collect()
    }
}

struct BatchReadProcessor<T> {
    access_adapter_arena: *mut DenseRecordArena,
    position: AtomicUsize,
    data_slice: *const T,
    meta_page: *mut PagedVec<AccessMetadata, PAGE_SIZE>,
}

unsafe impl<T: Copy + Send + Sync> Send for BatchReadProcessor<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for BatchReadProcessor<T> {}

impl<T: Copy + Send + Sync> BatchReadProcessor<T> {
    pub fn new(tracing_memory: &mut TracingMemory, addr_space: u32) -> Self {
        let start = tracing_memory
            .access_adapter_records
            .records_buffer
            .position();
        let access_adapter_arena =
            &mut tracing_memory.access_adapter_records as *mut DenseRecordArena;
        let data_slice = unsafe { tracing_memory.data.get_slice(addr_space, 0, 0).as_ptr() };
        let meta_page = unsafe {
            tracing_memory.meta.get_unchecked_mut(addr_space as usize)
                as *mut PagedVec<AccessMetadata, PAGE_SIZE>
        };
        Self {
            access_adapter_arena,
            position: AtomicUsize::new(start as usize),
            data_slice,
            meta_page,
        }
    }

    #[inline(always)]
    fn allocate(&self, header: &AccessRecordHeader) -> AccessRecordMut {
        let layout = AccessLayout::from_record_header(header);
        let size = size_by_layout(&layout);
        let start = self.position.fetch_add(size, Ordering::AcqRel);
        let access_adapter_arena = unsafe { &mut *self.access_adapter_arena };
        let bytes = &mut access_adapter_arena.records_buffer.get_mut()[start..(start + size)];
        <[u8] as CustomBorrow<AccessRecordMut<'_>, AccessLayout>>::custom_borrow(bytes, layout)
    }

    fn add_split_record(&self, header: AccessRecordHeader) {
        if header.block_size == header.lowest_block_size {
            return;
        }
        let data_slice = unsafe {
            from_raw_parts(
                self.data_slice.add(header.pointer as usize) as *const u8,
                header.block_size as usize * size_of::<T>(),
            )
        };

        let record_mut = self.allocate(&header);
        *record_mut.header = header;
        record_mut.data.copy_from_slice(data_slice);
        // we don't mind garbage values in prev_*
    }

    #[inline(always)]
    fn add_merge_record(&self, header: AccessRecordHeader, prev_ts: &[u32]) {
        if header.block_size == header.lowest_block_size {
            return;
        }
        debug_assert_eq!(header.block_size % header.lowest_block_size, 0);
        debug_assert_eq!(
            prev_ts.len() as u32,
            header.block_size / header.lowest_block_size
        );
        let data_slice = unsafe {
            from_raw_parts(
                self.data_slice.add(header.pointer as usize) as *const u8,
                header.block_size as usize * size_of::<T>(),
            )
        };

        let record_mut = self.allocate(&header);
        *record_mut.header = header;
        record_mut.header.timestamp_and_mask |= MERGE_AND_NOT_SPLIT_FLAG;
        record_mut.data.copy_from_slice(data_slice);
        record_mut.timestamps.copy_from_slice(prev_ts);
    }

    #[inline(always)]
    unsafe fn set_timestamp<const MIN_BLOCK_SIZE: usize>(
        &self,
        start_ptr: u32,
        end_ptr: u32,
        timestamp: u32,
    ) {
        let begin = start_ptr / MIN_BLOCK_SIZE as u32;
        let end = end_ptr / MIN_BLOCK_SIZE as u32;
        let meta_page = unsafe { &mut *self.meta_page };
        for i in begin..end {
            meta_page.set(
                i as usize,
                AccessMetadata::new(timestamp, MIN_BLOCK_SIZE as u8, 0),
            );
        }
    }

    #[inline(always)]
    unsafe fn set_meta_block<const BLOCK_SIZE: usize, const ALIGN: usize>(
        &self,
        pointer: usize,
        timestamp: u32,
    ) {
        let ptr = pointer / ALIGN;
        let meta_page = unsafe { &mut *self.meta_page };
        meta_page.set(ptr, AccessMetadata::new(timestamp, BLOCK_SIZE as u8, 0));
        // Store offsets for other positions in the block
        for i in 1..(BLOCK_SIZE / ALIGN) {
            meta_page.set(ptr + i, AccessMetadata::new(0, 0, i as u8));
        }
    }
}

impl<T> Drop for BatchReadProcessor<T> {
    fn drop(&mut self) {
        let access_adapter_arena = unsafe { &mut *self.access_adapter_arena };
        // No one else is accessing when dropping.
        let new_position = self.position.load(Ordering::Acquire);
        access_adapter_arena
            .records_buffer
            .set_position(new_position as u64);
    }
}
