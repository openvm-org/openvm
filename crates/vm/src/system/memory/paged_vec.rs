use std::{marker::PhantomData, mem::MaybeUninit, ptr};

use itertools::{zip_eq, Itertools};
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::online::GuestMemory;
use crate::arch::MemoryConfig;

/// (address_space, pointer)
pub type Address = (u32, u32);
/// 4096 is the default page size on host architectures if huge pages is not enabled
pub const PAGE_SIZE: usize = 1 << 12;

// TODO[jpw]: replace this with mmap implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedVec<const PAGE_SIZE: usize> {
    /// Assume each page in `pages` is either unalloc or PAGE_SIZE bytes long and aligned to
    /// PAGE_SIZE
    pub pages: Vec<Option<Vec<u8>>>,
}

// ------------------------------------------------------------------
// Common Helper Functions
// These functions encapsulate the common logic for copying ranges
// across pages, both for read-only and read-write (set) cases.
impl<const PAGE_SIZE: usize> PagedVec<PAGE_SIZE> {
    // Copies a range of length `len` starting at index `start`
    // into the memory pointed to by `dst`. If the relevant page is not
    // initialized, fills that portion with `0u8`.
    #[inline]
    fn read_range_generic(&self, start: usize, len: usize, dst: *mut u8) {
        let start_page = start / PAGE_SIZE;
        let end_page = (start + len - 1) / PAGE_SIZE;
        unsafe {
            if start_page == end_page {
                let offset = start % PAGE_SIZE;
                if let Some(page) = self.pages[start_page].as_ref() {
                    ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, len);
                } else {
                    std::slice::from_raw_parts_mut(dst, len).fill(0u8);
                }
            } else {
                debug_assert_eq!(start_page + 1, end_page);
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                if let Some(page) = self.pages[start_page].as_ref() {
                    ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, first_part);
                } else {
                    std::slice::from_raw_parts_mut(dst, first_part).fill(0u8);
                }
                let second_part = len - first_part;
                if let Some(page) = self.pages[end_page].as_ref() {
                    ptr::copy_nonoverlapping(page.as_ptr(), dst.add(first_part), second_part);
                } else {
                    std::slice::from_raw_parts_mut(dst.add(first_part), second_part).fill(0u8);
                }
            }
        }
    }

    // Updates a range of length `len` starting at index `start` with new values.
    // It copies the current values into the memory pointed to by `dst`
    // and then writes the new values into the underlying pages,
    // allocating pages (with defaults) if necessary.
    #[inline]
    fn set_range_generic(&mut self, start: usize, len: usize, new: *const u8, dst: *mut u8) {
        let start_page = start / PAGE_SIZE;
        let end_page = (start + len - 1) / PAGE_SIZE;
        unsafe {
            if start_page == end_page {
                let offset = start % PAGE_SIZE;
                let page = self.pages[start_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, len);
                ptr::copy_nonoverlapping(new, page.as_mut_ptr().add(offset), len);
            } else {
                assert_eq!(start_page + 1, end_page);
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                {
                    let page = self.pages[start_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, first_part);
                    ptr::copy_nonoverlapping(new, page.as_mut_ptr().add(offset), first_part);
                }
                let second_part = len - first_part;
                {
                    let page = self.pages[end_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(page.as_ptr(), dst.add(first_part), second_part);
                    ptr::copy_nonoverlapping(new.add(first_part), page.as_mut_ptr(), second_part);
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Implementation for types requiring Default + Clone
impl<const PAGE_SIZE: usize> PagedVec<PAGE_SIZE> {
    pub fn new(num_pages: usize) -> Self {
        Self {
            pages: vec![None; num_pages],
        }
    }

    /// Total capacity across available pages, in bytes.
    pub fn bytes_capacity(&self) -> usize {
        self.pages.len().checked_mul(PAGE_SIZE).unwrap()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.iter().all(|page| page.is_none())
    }

    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn get<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        // Create an uninitialized array of MaybeUninit<BLOCK>
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        self.read_range_generic(from, size_of::<BLOCK>(), result.as_mut_ptr() as *mut u8);
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
        let len = size_of::<BLOCK>();
        let start_page = start / PAGE_SIZE;
        let end_page = (start + len - 1) / PAGE_SIZE;
        let src = values as *const _ as *const u8;
        unsafe {
            if start_page == end_page {
                let offset = start % PAGE_SIZE;
                let page = self.pages[start_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                ptr::copy_nonoverlapping(src, page.as_mut_ptr().add(offset), len);
            } else {
                assert_eq!(start_page + 1, end_page);
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                {
                    let page = self.pages[start_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(src, page.as_mut_ptr().add(offset), first_part);
                }
                let second_part = len - first_part;
                {
                    let page = self.pages[end_page].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(src.add(first_part), page.as_mut_ptr(), second_part);
                }
            }
        }
    }

    /// memcpy of new `values` into pages, memcpy of old existing values into new returned value.
    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn replace<BLOCK: Copy>(&mut self, from: usize, values: &BLOCK) -> BLOCK {
        // Create an uninitialized array for old values.
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        self.set_range_generic(
            from,
            size_of::<BLOCK>(),
            values as *const _ as *const u8,
            result.as_mut_ptr() as *mut u8,
        );
        // SAFETY:
        // - All elements have been initialized (zero-initialized if page didn't exist).
        // - `result` is aligned to `BLOCK`
        unsafe { result.assume_init() }
    }
}

impl<const PAGE_SIZE: usize> PagedVec<PAGE_SIZE> {
    /// Iterate over [PagedVec] as iterator of elements of type `T`.
    /// Iterator is over `(index, element)` where `index` is the byte index divided by
    /// `size_of::<T>()`.
    ///
    /// `T` must be stack allocated
    pub fn iter<T: Copy>(&self) -> PagedVecIter<'_, T, PAGE_SIZE> {
        assert!(size_of::<T>() <= PAGE_SIZE);
        PagedVecIter {
            vec: self,
            current_page: 0,
            current_index_in_page: 0,
            phantom: PhantomData,
        }
    }
}

pub struct PagedVecIter<'a, T, const PAGE_SIZE: usize> {
    vec: &'a PagedVec<PAGE_SIZE>,
    current_page: usize,
    current_index_in_page: usize,
    phantom: PhantomData<T>,
}

impl<T: Copy, const PAGE_SIZE: usize> Iterator for PagedVecIter<'_, T, PAGE_SIZE> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_page < self.vec.pages.len()
            && self.vec.pages[self.current_page].is_none()
        {
            self.current_page += 1;
            debug_assert_eq!(self.current_index_in_page, 0);
            self.current_index_in_page = 0;
        }
        let global_index = self.current_page * PAGE_SIZE + self.current_index_in_page;
        if global_index + size_of::<T>() > self.vec.bytes_capacity() {
            return None;
        }

        // PERF: this can be optimized
        let value = self.vec.get(global_index);

        self.current_index_in_page += size_of::<T>();
        if self.current_index_in_page >= PAGE_SIZE {
            self.current_page += 1;
            self.current_index_in_page -= PAGE_SIZE;
        }
        Some((global_index / size_of::<T>(), value))
    }
}

/// Map from address space to guest memory.
/// The underlying memory is typeless, stored as raw bytes, but usage
/// implicitly assumes that each address space has memory cells of a fixed type (e.g., `u8, F`).
/// We do not use a typemap for performance reasons, and it is up to the user to enforce types.
/// Needless to say, this is a very `unsafe` API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressMap<const PAGE_SIZE: usize> {
    pub paged_vecs: Vec<PagedVec<PAGE_SIZE>>,
    /// byte size of cells per address space
    pub cell_size: Vec<usize>,
    pub as_offset: u32,
}

impl<const PAGE_SIZE: usize> Default for AddressMap<PAGE_SIZE> {
    fn default() -> Self {
        Self::from_mem_config(&MemoryConfig::default())
    }
}

impl<const PAGE_SIZE: usize> AddressMap<PAGE_SIZE> {
    pub fn new(as_offset: u32, as_cnt: usize, mem_size: usize) -> Self {
        // TMP: hardcoding for now
        let mut cell_size = vec![1, 1];
        cell_size.resize(as_cnt, 4);
        let paged_vecs = cell_size
            .iter()
            .map(|&cell_size| {
                PagedVec::new(mem_size.checked_mul(cell_size).unwrap().div_ceil(PAGE_SIZE))
            })
            .collect();
        Self {
            paged_vecs,
            cell_size,
            as_offset,
        }
    }
    pub fn from_mem_config(mem_config: &MemoryConfig) -> Self {
        Self::new(
            mem_config.as_offset,
            1 << mem_config.as_height,
            1 << mem_config.pointer_max_bits,
        )
    }
    pub fn items<F: PrimeField32>(&self) -> impl Iterator<Item = (Address, F)> + '_ {
        zip_eq(&self.paged_vecs, &self.cell_size)
            .enumerate()
            .flat_map(move |(as_idx, (page, &cell_size))| {
                // TODO: better way to handle address space conversions to F
                if cell_size == 1 {
                    page.iter::<u8>()
                        .map(move |(ptr_idx, x)| {
                            (
                                (as_idx as u32 + self.as_offset, ptr_idx as u32),
                                F::from_canonical_u8(x),
                            )
                        })
                        .collect_vec()
                } else {
                    // TEMP
                    assert_eq!(cell_size, 4);
                    page.iter::<F>()
                        .map(move |(ptr_idx, x)| {
                            ((as_idx as u32 + self.as_offset, ptr_idx as u32), x)
                        })
                        .collect_vec()
                }
            })
    }

    pub fn get_f<F: PrimeField32>(&self, addr_space: u32, ptr: u32) -> F {
        debug_assert_ne!(addr_space, 0);
        // TODO: fix this
        unsafe {
            if addr_space <= 2 {
                F::from_canonical_u8(self.get::<u8>((addr_space, ptr)))
            } else {
                self.get::<F>((addr_space, ptr))
            }
        }
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn get<T: Copy>(&self, (addr_space, ptr): Address) -> T {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize]
        );
        self.paged_vecs
            .get_unchecked((addr_space - self.as_offset) as usize)
            .get((ptr as usize) * size_of::<T>())
    }

    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub unsafe fn insert<T: Copy>(&mut self, (addr_space, ptr): Address, data: T) -> T {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize]
        );
        self.paged_vecs
            .get_unchecked_mut((addr_space - self.as_offset) as usize)
            .replace((ptr as usize) * size_of::<T>(), &data)
    }
    pub fn is_empty(&self) -> bool {
        self.paged_vecs.iter().all(|page| page.is_empty())
    }

    // TODO[jpw]: stabilize the boundary memory image format and how to construct
    /// # Safety
    /// - `T` **must** be the correct type for a single memory cell for `addr_space`
    /// - Assumes `addr_space` is within the configured memory and not out of bounds
    pub fn from_sparse(
        as_offset: u32,
        as_cnt: usize,
        mem_size: usize,
        sparse_map: SparseMemoryImage,
    ) -> Self {
        let mut vec = Self::new(as_offset, as_cnt, mem_size);
        for ((addr_space, index), data_byte) in sparse_map.into_iter() {
            vec.paged_vecs[(addr_space - as_offset) as usize].set(index as usize, &data_byte);
        }
        vec
    }
}

impl<const PAGE_SIZE: usize> GuestMemory for AddressMap<PAGE_SIZE> {
    unsafe fn read<T: Copy, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize]
        );
        self.paged_vecs
            .get_unchecked((addr_space - self.as_offset) as usize)
            .get((ptr as usize) * size_of::<T>())
    }

    unsafe fn write<T: Copy, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        values: &[T; BLOCK_SIZE],
    ) {
        debug_assert_eq!(
            size_of::<T>(),
            self.cell_size[(addr_space - self.as_offset) as usize],
            "addr_space={addr_space}"
        );
        self.paged_vecs
            .get_unchecked_mut((addr_space - self.as_offset) as usize)
            .set((ptr as usize) * size_of::<T>(), values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let mut v = PagedVec::<16>::new(3);
        assert_eq!(v.get::<u32>(0), 0u32);
        v.set(0, &42u32);
        assert_eq!(v.get::<u32>(0), 42u32);
    }

    // TEMP: disable tests (need to update indexing * size_of<u32>)
    // #[test]
    // fn test_cross_page_operations() {
    //     let mut v = PagedVec::<16>::new(3);
    //     v.set(3, 10u32); // Last element of first page
    //     v.set(4, 20u32); // First element of second page
    //     assert_eq!(v.get(3), 10u32);
    //     assert_eq!(v.get(4), 20u32);
    // }

    // #[test]
    // fn test_page_boundaries() {
    //     let mut v = PagedVec::<16>::new(2);
    //     // Fill first page
    //     v.set(0, 1u32);
    //     v.set(1, 2u32);
    //     v.set(2, 3u32);
    //     v.set(3, 4u32);
    //     // Fill second page
    //     v.set(4, 5u32);
    //     v.set(5, 6u32);
    //     v.set(6, 7u32);
    //     v.set(7, 8u32);

    //     // Verify all values
    //     assert_eq!(v.get::<[u32; 8]>(0), [1, 2, 3, 4, 5, 6, 7, 8]);
    // }

    // #[test]
    // fn test_range_cross_page_boundary() {
    //     let mut v = PagedVec::<16>::new(2);
    //     v.set::<[u32; 6]>(2, [10, 11, 12, 13, 14, 15]);
    //     assert_eq!(v.get::<[u32; 6]>(2), [10, 11, 12, 13, 14, 15]);
    // }

    // #[test]
    // fn test_large_indices() {
    //     let mut v = PagedVec::<16>::new(100);
    //     let large_index = 399;
    //     v.set(large_index, 42u32);
    //     assert_eq!(v.get(large_index), 42u32);
    // }

    // #[test]
    // fn test_range_operations_with_defaults() {
    //     let mut v = PagedVec::<16>::new(3);
    //     v.set(2, 5u32);
    //     v.set(5, 10u32);

    //     // Should include both set values and defaults
    //     assert_eq!(v.range_vec(1..7), [0, 5, 0, 0, 10, 0]);
    // }

    // #[test]
    // fn test_non_zero_default_type() {
    //     let mut v: PagedVec<4> = PagedVec::new(2);
    //     assert_eq!(v.get(0), false); // bool's default
    //     v.set(0, true);
    //     assert_eq!(v.get(0), true);
    //     assert_eq!(v.get(1), false); // because we created the page
    // }

    // #[test]
    // fn test_set_range_overlapping_pages() {
    //     let mut v = PagedVec::<_, 16>::new(3);
    //     let test_data = [1u32, 2, 3, 4, 5, 6];
    //     v.set(2, test_data);

    //     // Verify first page
    //     assert_eq!(v.get(2), 1u32);
    //     assert_eq!(v.get(3), 2u32);

    //     // Verify second page
    //     assert_eq!(v.get(4), 3u32);
    //     assert_eq!(v.get(5), 4u32);
    //     assert_eq!(v.get(6), 5u32);
    //     assert_eq!(v.get(7), 6u32);
    // }

    // #[test]
    // fn test_overlapping_set_ranges() {
    //     let mut v = PagedVec::<_, 16>::new(3);

    //     // Initial set_range
    //     v.set(0, [1u32, 2, 3, 4, 5]);
    //     assert_eq!(v.get::<[u32; 5]>(0), [1, 2, 3, 4, 5]);

    //     // Overlap from beginning
    //     v.set(0, [10u32, 20, 30]);
    //     assert_eq!(v.get::<[u32; 5]>(0), [10, 20, 30, 4, 5]);

    //     // Overlap in middle
    //     v.set(2, [42u32, 43]);
    //     assert_eq!(v.get::<[u32; 5]>(0), [10, 20, 42, 43, 5]);

    //     // Overlap at end
    //     v.set(4, [91u32, 92]);
    //     assert_eq!(v.get::<[u32; 6]>(0), [10, 20, 42, 43, 91, 92]);
    // }

    // #[test]
    // fn test_overlapping_set_ranges_cross_pages() {
    //     let mut v = PagedVec::<16>::new(3);

    //     // Fill across first two pages
    //     v.set::<[u32; 8]>(0, [1, 2, 3, 4, 5, 6, 7, 8]);

    //     // Overlap end of first page and start of second
    //     v.set::<[u32; 4]>(2, [21, 22, 23, 24]);
    //     assert_eq!(v.get::<[u32; 8]>(0), [1, 2, 21, 22, 23, 24, 7, 8]);

    //     // Overlap multiple pages
    //     v.set::<[u32; 6]>(1, [31, 32, 33, 34, 35, 36]);
    //     assert_eq!(v.get::<[u32; 8]>(0), [1, 31, 32, 33, 34, 35, 36, 8]);
    // }

    // #[test]
    // fn test_iterator() {
    //     let mut v = PagedVec::<16>::new(3);

    //     v.set(4..10, &[1, 2, 3, 4, 5, 6]);
    //     let contents: Vec<_> = v.iter().collect();
    //     assert_eq!(contents.len(), 8); // two pages

    //     contents
    //         .iter()
    //         .take(6)
    //         .enumerate()
    //         .for_each(|(i, &(idx, val))| {
    //             assert_eq!((idx, val), (4 + i, 1 + i));
    //         });
    //     assert_eq!(contents[6], (10, 0));
    //     assert_eq!(contents[7], (11, 0));
    // }
}
