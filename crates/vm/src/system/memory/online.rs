use std::{array::from_fn, fmt::Debug};

use getset::Getters;
use itertools::zip_eq;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::*,
};
use tracing::instrument;

use crate::{
    arch::{
        AddressSpaceHostConfig, AddressSpaceHostLayout, MemoryConfig, CONST_BLOCK_SIZE,
        MAX_CELL_BYTE_SIZE,
    },
    system::{
        memory::{MemoryAddress, TimestampedEquipartition, TimestampedValues},
        TouchedMemory,
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
    pub fn fill_zero(&mut self) {
        for mem in &mut self.mem {
            mem.fill_zero();
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
#[repr(C)]
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

    #[inline(always)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn get_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        self.memory.get_slice((addr_space, ptr), len)
    }

    #[inline(always)]
    fn debug_assert_cell_type<T>(&self, addr_space: u32) {
        debug_assert_eq!(
            size_of::<T>(),
            self.memory.config[addr_space as usize].layout.size()
        );
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
    /// Maps (addr_space, ptr / CONST_BLOCK_SIZE) → timestamp of last access.
    /// A value of 0 means the slot has never been accessed.
    pub(super) meta: Vec<PagedVec<u32, PAGE_SIZE>>,
}

impl TracingMemory {
    pub fn new(mem_config: &MemoryConfig) -> Self {
        let image = GuestMemory::new(AddressMap::from_mem_config(mem_config));
        Self::from_image(image)
    }

    /// Constructor from pre-existing memory image.
    pub fn from_image(image: GuestMemory) -> Self {
        let meta: Vec<_> = zip_eq(image.memory.get_memory(), &image.memory.config)
            .map(|(mem, addr_sp)| {
                let num_cells = mem.size() / addr_sp.layout.size();
                let total_metadata_len = num_cells.div_ceil(CONST_BLOCK_SIZE);
                PagedVec::new(total_metadata_len)
            })
            .collect();
        Self {
            data: image,
            meta,
            timestamp: INITIAL_TIMESTAMP + 1,
        }
    }

    #[inline(always)]
    fn assert_alignment(block_size: usize, addr_space: u32, ptr: u32) {
        debug_assert!(block_size.is_power_of_two());
        debug_assert_eq!(block_size % CONST_BLOCK_SIZE, 0);
        debug_assert_ne!(addr_space, 0);
        assert_eq!(
            ptr % (CONST_BLOCK_SIZE as u32),
            0,
            "pointer={ptr} not aligned to {CONST_BLOCK_SIZE}"
        );
    }

    /// Returns the previous access timestamp and updates metadata for this access.
    #[inline(always)]
    fn prev_access_time<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
    ) -> u32 {
        debug_assert_eq!(
            unsafe {
                self.data
                    .memory
                    .config
                    .get_unchecked(address_space)
                    .layout
                    .size()
            },
            size_of::<T>()
        );

        let base_idx = pointer / CONST_BLOCK_SIZE;
        let num_slots = BLOCK_SIZE / CONST_BLOCK_SIZE;
        // SAFETY: address_space is validated during instruction decoding
        let meta_page = unsafe { self.meta.get_unchecked_mut(address_space) };

        let mut max_timestamp = INITIAL_TIMESTAMP;
        for i in 0..num_slots {
            max_timestamp = max_timestamp.max(meta_page.get(base_idx + i));
            meta_page.set(base_idx + i, self.timestamp);
        }
        max_timestamp
    }

    /// Atomic read operation which increments the timestamp by 1.
    /// Returns `(t_prev, [pointer:BLOCK_SIZE]_{address_space})`.
    ///
    /// # Safety
    /// - `T` must be `repr(C)` or `repr(transparent)` and match the cell type for `address_space`.
    /// - `address_space` must be valid.
    /// - `BLOCK_SIZE` must be a multiple of `CONST_BLOCK_SIZE`.
    #[inline(always)]
    pub unsafe fn read<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        Self::assert_alignment(BLOCK_SIZE, address_space, pointer);
        let values = self.data.read(address_space, pointer);
        let t_prev =
            self.prev_access_time::<T, BLOCK_SIZE>(address_space as usize, pointer as usize);
        self.timestamp += 1;

        (t_prev, values)
    }

    /// Atomic write operation. Returns `(t_prev, values_prev)`.
    ///
    /// # Safety
    /// - `T` must be `repr(C)` or `repr(transparent)` and match the cell type for `address_space`.
    /// - `address_space` must be valid.
    /// - `BLOCK_SIZE` must be a multiple of `CONST_BLOCK_SIZE`.
    #[inline(always)]
    pub unsafe fn write<T, const BLOCK_SIZE: usize>(
        &mut self,
        address_space: u32,
        pointer: u32,
        values: [T; BLOCK_SIZE],
    ) -> (u32, [T; BLOCK_SIZE])
    where
        T: Copy + Debug,
    {
        Self::assert_alignment(BLOCK_SIZE, address_space, pointer);
        let values_prev = self.data.read(address_space, pointer);
        let t_prev =
            self.prev_access_time::<T, BLOCK_SIZE>(address_space as usize, pointer as usize);
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
    pub fn finalize<F: Field>(&mut self) -> TouchedMemory<F> {
        let touched_blocks = self.touched_blocks();
        self.touched_blocks_to_equipartition::<F, CONST_BLOCK_SIZE>(touched_blocks)
    }

    /// Returns the list of all touched blocks (address, timestamp). Sorted by address.
    fn touched_blocks(&self) -> Vec<(Address, u32)> {
        self.meta
            .par_iter()
            .enumerate()
            .flat_map(|(addr_space, meta_page)| {
                meta_page
                    .par_iter()
                    .filter_map(move |(idx, timestamp)| {
                        if timestamp > INITIAL_TIMESTAMP {
                            let ptr = idx as u32 * CONST_BLOCK_SIZE as u32;
                            Some(((addr_space as u32, ptr), timestamp))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Returns the equipartition of the touched blocks.
    fn touched_blocks_to_equipartition<F: Field, const CHUNK: usize>(
        &mut self,
        touched_blocks: Vec<((u32, u32), u32)>,
    ) -> TimestampedEquipartition<F, CHUNK> {
        let mut final_memory = Vec::new();

        debug_assert!(touched_blocks.is_sorted_by_key(|(addr, _)| addr));
        self.handle_touched_blocks::<F, CHUNK>(&mut final_memory, touched_blocks);

        debug_assert!(final_memory.is_sorted_by_key(|(key, _)| *key));
        final_memory
    }

    fn handle_touched_blocks<F: Field, const CHUNK: usize>(
        &mut self,
        final_memory: &mut Vec<((u32, u32), TimestampedValues<F, CHUNK>)>,
        touched_blocks: Vec<((u32, u32), u32)>,
    ) {
        let mut current_values = vec![0u8; MAX_CELL_BYTE_SIZE * CHUNK];
        let mut current_cnt = 0;
        let mut current_address = MemoryAddress::new(0, 0);
        let mut current_timestamps = vec![0u32; CHUNK / CONST_BLOCK_SIZE];
        for ((addr_space, ptr), timestamp) in touched_blocks {
            // SAFETY: addr_space of touched blocks are all in bounds
            let addr_space_config =
                unsafe { *self.data.memory.config.get_unchecked(addr_space as usize) };
            let cell_size = addr_space_config.layout.size();
            debug_assert!(ptr % CONST_BLOCK_SIZE as u32 == 0);

            assert!(
                current_cnt == 0
                    || (current_address.address_space == addr_space
                        && current_address.pointer + current_cnt as u32 == ptr),
                "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`"
            );

            if current_cnt == 0 {
                assert_eq!(
                    ptr & (CHUNK as u32 - 1),
                    0,
                    "The union of all touched blocks must consist of `CHUNK`-aligned blocks"
                );
                current_address = MemoryAddress::new(addr_space, ptr);
            }

            // SAFETY: current_cnt / CONST_BLOCK_SIZE < CHUNK / CONST_BLOCK_SIZE
            unsafe {
                *current_timestamps.get_unchecked_mut(current_cnt / CONST_BLOCK_SIZE) = timestamp;
            }

            for i in 0..CONST_BLOCK_SIZE as u32 {
                let cell_data = unsafe {
                    self.data.memory.get_u8_slice(
                        addr_space,
                        (ptr + i) as usize * cell_size,
                        cell_size,
                    )
                };
                current_values[current_cnt * cell_size..current_cnt * cell_size + cell_size]
                    .copy_from_slice(cell_data);
                current_cnt += 1;
                if current_cnt == CHUNK {
                    let timestamp = *current_timestamps.iter().max().unwrap();
                    final_memory.push((
                        (current_address.address_space, current_address.pointer),
                        TimestampedValues {
                            timestamp,
                            values: from_fn(|i| unsafe {
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
        assert_eq!(current_cnt, 0, "The union of all touched blocks must consist of blocks with sizes divisible by `CHUNK`");
    }
}
