use std::{array::from_fn, fmt::Debug};

use getset::Getters;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::*,
};
use thiserror::Error;
use tracing::instrument;

use crate::{
    arch::{AddressSpaceHostConfig, AddressSpaceHostLayout, MemoryConfig, BLOCK_FE_WIDTH},
    system::{TouchedBlock, TouchedMemory},
};

mod basic;
#[cfg(all(unix, feature = "rvr", not(feature = "basic-memory")))]
mod guarded;
#[cfg(any(unix, windows))]
mod memmap;
mod paged_vec;
mod touched_pages;

#[cfg(not(any(unix, windows)))]
pub use basic::*;
#[cfg(all(unix, feature = "rvr", not(feature = "basic-memory")))]
pub use guarded::GuardedMemory;
#[cfg(any(unix, windows))]
pub use memmap::*;
pub use paged_vec::PagedVec;
pub use touched_pages::TouchedPages;

#[cfg(all(unix, not(feature = "basic-memory"), feature = "rvr"))]
pub type MemoryBackend = guarded::GuardedMemory;
#[cfg(all(
    any(unix, windows),
    not(feature = "basic-memory"),
    not(all(unix, feature = "rvr"))
))]
pub type MemoryBackend = memmap::MmapMemory;
#[cfg(any(not(any(unix, windows)), feature = "basic-memory"))]
pub type MemoryBackend = basic::BasicMemory;

pub const INITIAL_TIMESTAMP: u32 = 0;
/// Default mmap page size. Change this if using THB.
pub const PAGE_SIZE: usize = 4096;

/// `(address_space, ptr)` for typed memory-cell access.
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
#[repr(C)]
pub struct AddressMap<M: LinearMemory = MemoryBackend> {
    /// Underlying memory data.
    pub mem: Vec<M>,
    /// Host configuration for each address space.
    pub config: Vec<AddressSpaceHostConfig>,
    /// Per-address-space record of which pages may contain non-zero data, used to skip all-zero
    /// pages during the GPU host-to-device transfer. See [`TouchedPages`].
    ///
    /// Invariant: any path that writes non-zero data into memory that may later be transferred via
    /// `set_initial_memory` must mark the written pages (`set_from_sparse`,
    /// `extend_touched_pages_from_touched`). Unmarked pages are transferred as zero. The untracked
    /// write paths (`get_memory_mut`, `GuestMemory::write`/`write_bytes`) do not mark and must not
    /// be the source of transferred initial memory.
    pub touched_pages: Vec<TouchedPages>,
}

impl Default for AddressMap {
    fn default() -> Self {
        Self::from_mem_config(&MemoryConfig::default())
    }
}

impl<M: LinearMemory> AddressMap<M> {
    pub fn new(config: Vec<AddressSpaceHostConfig>) -> Self {
        assert_eq!(config[0].num_cells, 0, "Address space 0 must have 0 cells");
        let mem: Vec<M> = config
            .iter()
            .map(|config| M::new(config.num_cells.checked_mul(config.layout.size()).unwrap()))
            .collect();
        // Pages start unmarked (guaranteed zero); paths that write data (`set_from_sparse`) and the
        // carried-forward extension (`extend_touched_pages_from_touched`) mark the pages they
        // touch. See the invariant on `touched_pages`.
        let touched_pages = mem.iter().map(|m| TouchedPages::new(m.size())).collect();
        Self {
            mem,
            config,
            touched_pages,
        }
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
    pub fn fill_zero(&mut self) {
        for (mem, touched) in self.mem.iter_mut().zip(self.touched_pages.iter_mut()) {
            mem.fill_zero();
            // Memory is now all zero, so no pages are touched.
            *touched = TouchedPages::new(mem.size());
        }
    }

    /// # Safety
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get_f<F: PrimeField32>(&self, addr_space: u32, ptr: u32) -> F {
        let layout = &self.config.get_unchecked(addr_space as usize).layout;
        let start = ptr as usize * layout.size();
        let bytes = self.get_u8_slice(addr_space, start, layout.size());
        layout.to_field(bytes)
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
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

    /// Returns a typed cell slice starting at `ptr`.
    ///
    /// Panics or segfaults if `ptr..ptr + len` is out of bounds.
    ///
    /// # Safety
    /// - `T` must exactly match the AS's cell type.
    /// - `addr_space` must be within the configured memory.
    pub unsafe fn get_slice<T: Copy + Debug>(
        &self,
        (addr_space, ptr): Address,
        len: usize,
    ) -> &[T] {
        assert_eq!(
            size_of::<T>(),
            self.config[addr_space as usize].layout.size(),
            "typed slice access must use the AS cell type; use get_u8_slice for raw bytes"
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

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub fn set_from_sparse(&mut self, sparse_map: &SparseMemoryImage) {
        // Callers always pass freshly-zeroed memory, so reset the touched-page sets: any page not
        // written from the sparse image below is guaranteed zero. This narrows the segment-0
        // initial image to exactly the sparse pages.
        for (mem, touched) in self.mem.iter().zip(self.touched_pages.iter_mut()) {
            *touched = TouchedPages::new(mem.size());
        }
        for (&(addr_space, ptr), &data_byte) in sparse_map.iter() {
            // SAFETY:
            // - safety assumptions in function doc comments
            unsafe {
                self.mem
                    .get_unchecked_mut(addr_space as usize)
                    .write_unaligned(ptr as usize, data_byte);
            }
            self.touched_pages[addr_space as usize].mark_byte_range(ptr as usize, 1);
        }
    }

    /// Marks the pages covering each touched block as possibly non-zero. Grows the touched-page
    /// sets so that a carried-forward memory image (a preflight `to_state`) stays a correct
    /// superset of its non-zero pages across continuation segments.
    ///
    /// `touched` is the [`TouchedMemory`] produced by `TracingMemory::finalize`; its `ptr` is in
    /// AS-native cells and each block spans `BLOCK_FE_WIDTH` cells.
    pub fn extend_touched_pages_from_touched<F>(&mut self, touched: &TouchedMemory<F>) {
        for block in touched.iter() {
            let cell_size = self.config[block.address_space as usize].layout.size();
            let start = block.ptr as usize * cell_size;
            let len = BLOCK_FE_WIDTH * cell_size;
            self.touched_pages[block.address_space as usize].mark_byte_range(start, len);
        }
    }
}

/// API for guest memory conforming to OpenVM ISA
// @dev Note we don't make this a trait because phantom executors currently need a concrete type for
// guest memory
#[derive(Debug, Clone)]
#[repr(C)]
pub struct GuestMemory {
    pub memory: AddressMap,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GuestMemoryAccessError {
    #[error("address space {addr_space} is not configured")]
    InvalidAddressSpace { addr_space: u32 },
    #[error("range overflow")]
    RangeOverflow,
    #[error("memory range out of bounds: start={start} size={len} memory_size={memory_size}")]
    RangeOutOfBounds {
        start: u64,
        len: u64,
        memory_size: usize,
    },
}

impl GuestMemory {
    pub fn new(addr: AddressMap) -> Self {
        Self { memory: addr }
    }

    /// Reads `BLOCK_SIZE` AS-native cells starting at `ptr`.
    ///
    /// # Safety
    /// - `T` must be stack-allocated `repr(C)` or `repr(transparent)`.
    /// - `T` must match the configured cell type for `addr_space`.
    /// - `addr_space` and `ptr..ptr + BLOCK_SIZE` must be in bounds.
    /// - `T` must be plain data compatible with [`LinearMemory::read`].
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

    /// Writes `BLOCK_SIZE` AS-native cells starting at `ptr`.
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

    /// Swaps `BLOCK_SIZE` AS-native cells starting at `ptr`.
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

    #[inline(always)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn get_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        self.memory.get_slice((addr_space, ptr), len)
    }

    /// Reads a raw byte slice at `byte_ptr` within `addr_space`.
    ///
    /// # Safety
    /// The full byte range must lie within the AS's storage.
    #[inline(always)]
    pub unsafe fn get_u8_slice(&self, addr_space: u32, byte_ptr: u32, len: usize) -> &[u8] {
        self.memory.get_u8_slice(addr_space, byte_ptr as usize, len)
    }

    /// Reads a raw byte range addressed by full RV64 register values.
    pub fn checked_u8_slice(
        &self,
        addr_space: u32,
        byte_ptr: u64,
        len: u64,
    ) -> Result<&[u8], GuestMemoryAccessError> {
        let end = byte_ptr
            .checked_add(len)
            .ok_or(GuestMemoryAccessError::RangeOverflow)?;
        let memory = self
            .memory
            .get_memory()
            .get(addr_space as usize)
            .ok_or(GuestMemoryAccessError::InvalidAddressSpace { addr_space })?;
        let memory_size = memory.size();
        if end > memory_size as u64 {
            return Err(GuestMemoryAccessError::RangeOutOfBounds {
                start: byte_ptr,
                len,
                memory_size,
            });
        }
        Ok(&memory.as_slice()[byte_ptr as usize..end as usize])
    }

    #[inline(always)]
    fn debug_assert_cell_type<T>(&self, addr_space: u32) {
        debug_assert_eq!(
            size_of::<T>(),
            self.memory.config[addr_space as usize].layout.size(),
            "typed cell access must use the AS cell type"
        );
    }

    /// Reads `N` raw storage bytes starting at `byte_ptr`.
    ///
    /// # Safety
    /// The full byte range must lie within the AS's storage.
    #[inline(always)]
    pub unsafe fn read_bytes<const N: usize>(&self, addr_space: u32, byte_ptr: u32) -> [u8; N] {
        self.memory
            .get_memory()
            .get_unchecked(addr_space as usize)
            .read(byte_ptr as usize)
    }

    /// Writes `N` raw storage bytes starting at `byte_ptr`.
    ///
    /// # Safety
    /// See [`GuestMemory::read_bytes`].
    #[inline(always)]
    pub unsafe fn write_bytes<const N: usize>(
        &mut self,
        addr_space: u32,
        byte_ptr: u32,
        values: [u8; N],
    ) {
        self.memory
            .get_memory_mut()
            .get_unchecked_mut(addr_space as usize)
            .write(byte_ptr as usize, values);
    }
}

/// Online memory that stores additional information for trace generation purposes.
/// In particular, keeps track of timestamp.
#[derive(Getters)]
pub struct TracingMemory {
    pub timestamp: u32,
    /// The underlying data memory, with memory cells typed by address space: see [AddressMap].
    #[getset(get = "pub")]
    pub data: GuestMemory,
    /// Maps `(addr_space, ptr / BLOCK_FE_WIDTH)` to the latest access timestamp.
    /// A value of 0 means the touched-memory slot has never been accessed.
    pub(super) meta: Vec<PagedVec<u32, PAGE_SIZE>>,
}

impl TracingMemory {
    pub fn new(mem_config: &MemoryConfig) -> Self {
        let image = GuestMemory::new(AddressMap::from_mem_config(mem_config));
        Self::from_image(image)
    }

    /// Constructor from pre-existing memory image.
    pub fn from_image(image: GuestMemory) -> Self {
        let meta = image
            .memory
            .config
            .iter()
            .map(|config| PagedVec::new(config.num_cells.div_ceil(BLOCK_FE_WIDTH)))
            .collect();
        Self {
            data: image,
            meta,
            timestamp: INITIAL_TIMESTAMP + 1,
        }
    }

    #[inline(always)]
    fn assert_valid_access<const BLOCK_SIZE: usize>(&self, addr_space: u32, ptr: u32) {
        const {
            assert!(
                BLOCK_SIZE == BLOCK_FE_WIDTH,
                "TracingMemory only supports BLOCK_FE_WIDTH-cell accesses"
            )
        };
        debug_assert_ne!(addr_space, 0);
        assert_eq!(
            ptr % BLOCK_SIZE as u32,
            0,
            "ptr={ptr} not aligned to BLOCK_SIZE {BLOCK_SIZE}"
        );
    }

    #[inline(always)]
    fn assert_valid_byte_access<const N: usize>(&self, addr_space: u32, byte_ptr: u32) {
        debug_assert_ne!(addr_space, 0);
        let block_bytes = self.memory_block_bytes(addr_space);
        assert_eq!(
            N, block_bytes,
            "raw byte access must cover one {block_bytes}-byte memory block; got {N}"
        );
        assert_eq!(
            byte_ptr as usize % block_bytes,
            0,
            "byte_ptr={byte_ptr} not aligned to block_bytes {block_bytes}"
        );
    }

    #[inline(always)]
    fn prev_access_time(&mut self, address_space: usize, ptr: usize) -> u32 {
        let idx = ptr / BLOCK_FE_WIDTH;
        // SAFETY: address_space is validated during instruction decoding
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };
        let prev = meta_page.get(idx);
        meta_page.set(idx, self.timestamp);
        prev
    }

    #[inline(always)]
    fn byte_prev_access_time(&mut self, address_space: usize, byte_ptr: usize) -> u32 {
        let idx = byte_ptr / self.memory_block_bytes(address_space as u32);
        // SAFETY: address_space is validated during instruction decoding
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };
        let prev = meta_page.get(idx);
        meta_page.set(idx, self.timestamp);
        prev
    }

    #[inline(always)]
    fn memory_block_bytes(&self, address_space: u32) -> usize {
        BLOCK_FE_WIDTH
            * self.data.memory.config[address_space as usize]
                .layout
                .size()
    }

    /// Atomic cell read operation which increments the timestamp by 1.
    /// Returns `(t_prev, values)`.
    ///
    /// # Safety
    /// - `T` must be `repr(C)` or `repr(transparent)`.
    /// - `T` must match the configured cell type for `address_space`.
    /// - `ptr` must be aligned to `BLOCK_SIZE`.
    /// - `address_space` must be valid.
    #[inline(always)]
    pub unsafe fn read<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        ptr: u32,
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        self.assert_valid_access::<BLOCK_SIZE>(address_space, ptr);
        let t_prev = self.prev_access_time(address_space as usize, ptr as usize);
        let values = self.data.read(address_space, ptr);
        self.timestamp += 1;

        (t_prev, values)
    }

    /// Atomic cell write operation. Returns `(t_prev, values_prev)`.
    ///
    /// # Safety
    /// - `T` must be `repr(C)` or `repr(transparent)`.
    /// - `T` must match the configured cell type for `address_space`.
    /// - `ptr` must be aligned to `BLOCK_SIZE`.
    /// - `address_space` must be valid.
    #[inline(always)]
    pub unsafe fn write<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        ptr: u32,
        values: [T; BLOCK_SIZE],
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        self.assert_valid_access::<BLOCK_SIZE>(address_space, ptr);
        let t_prev = self.prev_access_time(address_space as usize, ptr as usize);
        let values_prev = self.data.read(address_space, ptr);
        self.data.write(address_space, ptr, values);
        self.timestamp += 1;

        (t_prev, values_prev)
    }

    /// Atomic raw byte read.
    ///
    /// # Safety
    /// - `byte_ptr` must be aligned to the AS memory block size.
    /// - `N` must equal that block size.
    /// - `byte_ptr + N` must be within the AS's storage backing.
    /// - `address_space` must be a valid configured address space.
    #[inline(always)]
    pub unsafe fn read_bytes<const N: usize>(
        &mut self,
        address_space: u32,
        byte_ptr: u32,
    ) -> (u32, [u8; N]) {
        self.assert_valid_byte_access::<N>(address_space, byte_ptr);
        let t_prev = self.byte_prev_access_time(address_space as usize, byte_ptr as usize);
        let values = self.data.read_bytes::<N>(address_space, byte_ptr);
        self.timestamp += 1;

        (t_prev, values)
    }

    /// Atomic raw byte write. See [`TracingMemory::read_bytes`].
    ///
    /// # Safety
    /// Same as [`TracingMemory::read_bytes`].
    #[inline(always)]
    pub unsafe fn write_bytes<const N: usize>(
        &mut self,
        address_space: u32,
        byte_ptr: u32,
        values: [u8; N],
    ) -> (u32, [u8; N]) {
        self.assert_valid_byte_access::<N>(address_space, byte_ptr);
        let t_prev = self.byte_prev_access_time(address_space as usize, byte_ptr as usize);
        let values_prev = self.data.read_bytes::<N>(address_space, byte_ptr);
        self.data.write_bytes::<N>(address_space, byte_ptr, values);
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
    pub fn finalize<F: Field>(&mut self) -> TouchedMemory<F> {
        self.touched_blocks_to_equipartition::<F>(self.touched_blocks())
    }

    /// Returns the list of all touched blocks (address, timestamp), sorted by address.
    fn touched_blocks(&self) -> Vec<(Address, u32)> {
        assert_eq!(self.meta.len(), self.data.memory.config.len());
        let mut touched_blocks: Vec<_> = self
            .meta
            .par_iter()
            .enumerate()
            .flat_map_iter(|(addr_space, meta_page)| {
                meta_page
                    .par_iter()
                    .filter_map(move |(idx, timestamp)| {
                        if timestamp > INITIAL_TIMESTAMP {
                            let ptr = idx as u32 * BLOCK_FE_WIDTH as u32;
                            Some(((addr_space as u32, ptr), timestamp))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // This sort may not be strictly necessary, but it makes the finalize path independent of
        // Rayon ordering.
        touched_blocks.sort_unstable_by_key(|(addr, _)| *addr);
        touched_blocks
    }

    /// Returns touched memory in `BLOCK_FE_WIDTH`-cell blocks.
    fn touched_blocks_to_equipartition<F: Field>(
        &self,
        touched_blocks: Vec<((u32, u32), u32)>,
    ) -> TouchedMemory<F> {
        debug_assert!(touched_blocks.is_sorted_by_key(|(addr, _)| addr));
        touched_blocks
            .into_par_iter()
            .map(|((addr_space, ptr), timestamp)| {
                let addr_space_config = &self.data.memory.config[addr_space as usize];
                let cell_size = addr_space_config.layout.size();
                let values = from_fn(|i| unsafe {
                    addr_space_config
                        .layout
                        .to_field(self.data.memory.get_u8_slice(
                            addr_space,
                            (ptr as usize + i) * cell_size,
                            cell_size,
                        ))
                });
                TouchedBlock {
                    address_space: addr_space,
                    ptr,
                    timestamp,
                    values,
                }
            })
            .collect()
    }
}
