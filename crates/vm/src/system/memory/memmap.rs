use std::{fmt::Debug, marker::PhantomData, mem::MaybeUninit, ptr::copy_nonoverlapping};

use itertools::{zip_eq, Itertools};
use memmap2::MmapMut;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::arch::MemoryConfig;

/// (address_space, pointer)
pub type Address = (u32, u32);

/// A wrapper around MmapMut that implements Clone and provides memory operations
#[derive(Debug)]
pub struct MmapWrapper {
    mem: MmapMut,
}

impl Clone for MmapWrapper {
    fn clone(&self) -> Self {
        let mut new_mem = MmapMut::map_anon(self.mem.len()).unwrap();
        new_mem.copy_from_slice(&self.mem);
        Self { mem: new_mem }
    }
}

/// Iterator over MmapWrapper that yields elements of type T
pub struct MmapWrapperIter<'a, T: Copy> {
    wrapper: &'a MmapWrapper,
    current_index: usize,
    phantom: PhantomData<T>,
}

impl<'a, T: Copy> MmapWrapperIter<'a, T> {
    fn new(wrapper: &'a MmapWrapper) -> Self {
        Self {
            wrapper,
            current_index: 0,
            phantom: PhantomData,
        }
    }
}

impl<T: Copy> Iterator for MmapWrapperIter<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        let size = std::mem::size_of::<T>();
        if self.current_index + size > self.wrapper.len() {
            return None;
        }

        let value = self.wrapper.get::<T>(self.current_index);
        let index = self.current_index / size;
        self.current_index += size;
        Some((index, value))
    }
}

impl MmapWrapper {
    pub fn new(len: usize) -> Self {
        Self {
            mem: MmapMut::map_anon(len).unwrap(),
        }
    }

    pub fn len(&self) -> usize {
        self.mem.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mem.is_empty()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.mem.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mem.as_mut_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.mem
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mem
    }

    /// Iterate over MmapWrapper as iterator of elements of type `T`.
    /// Iterator is over `(index, element)` where `index` is the byte index divided by
    /// `size_of::<T>()`.
    ///
    /// `T` must be stack allocated
    pub fn iter<T: Copy>(&self) -> MmapWrapperIter<'_, T> {
        MmapWrapperIter::new(self)
    }

    // Copies a range of length `len` starting at index `start`
    // into the memory pointed to by `dst`. If the relevant range is not
    // initialized, fills that range with `0u8`.
    /// # Safety
    /// - `dst` must be a valid pointer to a memory location
    /// - `start` and `start + len` must be within the bounds of the memory
    #[inline]
    pub unsafe fn read_range_generic(&self, start: usize, len: usize, dst: *mut u8) {
        // Calculate how much we can actually copy
        let copy_len = std::cmp::min(len, self.len() - start);

        // Copy the data
        copy_nonoverlapping(self.as_ptr().add(start), dst, copy_len);

        // If we couldn't copy everything, ensure the rest is zeroed
        if copy_len < len {
            std::slice::from_raw_parts_mut(dst.add(copy_len), len - copy_len).fill(0u8);
        }
    }

    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn get<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        // Create an uninitialized array of MaybeUninit<BLOCK>
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        unsafe {
            self.read_range_generic(from, size_of::<BLOCK>(), result.as_mut_ptr() as *mut u8);
        }
        // SAFETY:
        // - All elements have been initialized (zero-initialized if page didn't exist).
        // - `result` is aligned to `BLOCK`
        unsafe { result.assume_init() }
    }

    /// # Panics
    /// If `start..start + size_of<BLOCK>()` is out of bounds.
    // @dev: `values` is passed by reference since the data is copied into memory. Even though the
    // compiler probably optimizes it, we use reference to avoid any unnecessary copy of `values`
    // onto the stack in the function call.
    #[inline(always)]
    pub fn set<BLOCK: Copy>(&mut self, start: usize, values: &BLOCK) {
        unsafe {
            copy_nonoverlapping(
                values as *const _ as *const u8,
                self.as_mut_ptr().add(start),
                std::mem::size_of::<BLOCK>(),
            );
        }
    }

    /// memcpy of new `values` into from..from + size_of<BLOCK>(), memcpy of old existing values into new returned value.
    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn replace<BLOCK: Copy>(&mut self, from: usize, values: &BLOCK) -> BLOCK {
        // Create an uninitialized array of MaybeUninit<BLOCK>
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        unsafe {
            copy_nonoverlapping(
                self.as_ptr().add(from),
                result.as_mut_ptr() as *mut u8,
                std::mem::size_of::<BLOCK>(),
            );
            copy_nonoverlapping(
                values as *const _ as *const u8,
                self.as_mut_ptr().add(from),
                std::mem::size_of::<BLOCK>(),
            );
        }
        // SAFETY:
        // - All elements have been initialized (zero-initialized if didn't exist).
        // - `result` is aligned to `BLOCK`
        unsafe { result.assume_init() }
    }
}

#[derive(Debug, Clone)]
pub struct AddressMap {
    pub mem: Vec<MmapWrapper>,
    pub cell_size: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddressMapSerializeHelper {
    compressed: Vec<Vec<(usize, Vec<u8>)>>,
    lengths: Vec<usize>,
    cell_size: Vec<usize>,
}

impl AddressMapSerializeHelper {
    fn from(map: &AddressMap) -> Self {
        assert!(map.mem.len() % PAGE_SIZE == 0);
        let compressed = map
            .mem
            .iter()
            .map(|space_mem| {
                space_mem
                    .iter::<u8>()
                    .map(|(_, x)| x)
                    .chunks(PAGE_SIZE)
                    .into_iter()
                    .enumerate()
                    .flat_map(|(i, chunk)| {
                        let cv = chunk.collect::<Vec<_>>();
                        if cv.iter().all(|x| x == &0) {
                            None
                        } else {
                            Some((i, cv))
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let lengths = map.mem.iter().map(|m| m.len()).collect::<Vec<_>>();
        Self {
            compressed,
            lengths,
            cell_size: map.cell_size.clone(),
        }
    }

    fn to(&self) -> AddressMap {
        let mut mem = self
            .lengths
            .iter()
            .map(|l| MmapWrapper::new(*l))
            .collect::<Vec<_>>();
        for (space_compr, space_mem) in self.compressed.iter().zip(mem.iter_mut()) {
            for (i, chunk) in space_compr.iter() {
                space_mem.as_mut_slice()[i * PAGE_SIZE..(i + 1) * PAGE_SIZE].copy_from_slice(chunk);
            }
        }
        AddressMap {
            mem,
            cell_size: self.cell_size.clone(),
        }
    }
}

const PAGE_SIZE: usize = 4096;

impl Serialize for AddressMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        AddressMapSerializeHelper::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for AddressMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = AddressMapSerializeHelper::deserialize(deserializer)?;
        Ok(helper.to())
    }
}

impl Default for AddressMap {
    fn default() -> Self {
        unimplemented!()
    }
}

impl AddressMap {
    pub fn new(mem_size: Vec<usize>) -> Self {
        // TMP: hardcoding for now
        let mut cell_size = vec![1; 4];
        cell_size.resize(mem_size.len(), 4);
        let mem = zip_eq(&cell_size, &mem_size)
            .map(|(cell_size, mem_size)| {
                MmapWrapper::new(mem_size.checked_mul(*cell_size).unwrap())
            })
            .collect();
        Self {
            mem,
            cell_size,
        }
    }

    pub fn from_mem_config(mem_config: &MemoryConfig) -> Self {
        Self::new(
            mem_config.as_sizes.clone(),
        )
    }

    pub fn items<F: PrimeField32>(&self) -> impl Iterator<Item = (Address, F)> + '_ {
        zip_eq(&self.mem, &self.cell_size).enumerate().flat_map(
            move |(as_idx, (space_mem, &cell_size))| {
                // TODO: better way to handle address space conversions to F
                if cell_size == 1 {
                    space_mem
                        .iter::<u8>()
                        .map(move |(ptr_idx, x)| {
                            (
                                (as_idx as u32, ptr_idx as u32),
                                F::from_canonical_u8(x),
                            )
                        })
                        .collect_vec()
                } else {
                    // TEMP
                    assert_eq!(cell_size, 4);
                    space_mem
                        .iter::<F>()
                        .map(move |(ptr_idx, x)| {
                            ((as_idx as u32, ptr_idx as u32), x)
                        })
                        .collect_vec()
                }
            },
        )
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
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            self.cell_size[addr_space as usize]
        );
        self.mem[addr_space as usize].get(ptr as usize)
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn insert<T: Copy>(&mut self, (addr_space, ptr): Address, data: T) -> T {
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            self.cell_size[addr_space as usize]
        );
        self.mem[addr_space as usize].replace(ptr as usize, &data)
    }

    pub fn is_empty(&self) -> bool {
        self.mem.is_empty()
    }

    pub fn read_range_generic<T: Copy + Debug>(
        &self,
        (addr_space, ptr): Address,
        len: usize,
    ) -> Vec<T> {
        let mut block: Vec<T> = Vec::with_capacity(len);
        unsafe {
            self.mem[addr_space as usize].read_range_generic(
                ptr as usize,
                len,
                block.as_mut_ptr() as *mut u8,
            );
            block.set_len(len);
        }
        block
    }

    // TODO[jpw]: stabilize the boundary memory image format and how to construct
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub fn from_sparse(
        mem_size: Vec<usize>,
        sparse_map: SparseMemoryImage,
    ) -> Self {
        let mut vec = Self::new(mem_size);
        for ((addr_space, index), data_byte) in sparse_map.into_iter() {
            vec.mem[addr_space as usize].set(index as usize, &data_byte);
        }
        vec
    }
}
