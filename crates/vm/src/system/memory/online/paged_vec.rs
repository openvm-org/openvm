use std::{fmt::Debug, marker::PhantomData, mem::MaybeUninit, ptr};

use serde::{Deserialize, Serialize};

/// 4096 is the default page size on host architectures if huge pages is not enabled
const PAGE_SIZE: usize = 1 << 12;
pub const CELL_STRIDE: usize = 1 << 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedVec {
    /// Assume each page in `pages` is either unalloc or PAGE_SIZE bytes long and aligned to
    /// PAGE_SIZE
    pub pages: Vec<Option<Vec<u8>>>,
}

// ------------------------------------------------------------------
// Common Helper Functions
// These functions encapsulate the common logic for copying ranges
// across pages, both for read-only and read-write (set) cases.
impl PagedVec {
    // Copies a range of length `len` starting at index `start`
    // into the memory pointed to by `dst`. If the relevant page is not
    // initialized, fills that portion with `0u8`.
    #[inline]
    pub fn read_range_generic(&self, start: usize, len: usize, mut dst: *mut u8) {
        let start_page_idx = start / PAGE_SIZE;
        let end_page_idx = (start + len - 1) / PAGE_SIZE;
        unsafe {
            if start_page_idx == end_page_idx {
                let offset = start % PAGE_SIZE;
                if let Some(page) = self.pages[start_page_idx].as_ref() {
                    ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, len);
                } else {
                    std::slice::from_raw_parts_mut(dst, len).fill(0u8);
                }
            } else {
                // First page
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                {
                    if let Some(page) = self.pages[start_page_idx].as_ref() {
                        ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, first_part);
                    } else {
                        std::slice::from_raw_parts_mut(dst, first_part).fill(0u8);
                    }
                    dst = dst.add(first_part);
                }

                // Middle pages
                for page_idx in (start_page_idx + 1)..end_page_idx {
                    if let Some(page) = self.pages[page_idx].as_ref() {
                        ptr::copy_nonoverlapping(page.as_ptr(), dst, PAGE_SIZE);
                    } else {
                        std::slice::from_raw_parts_mut(dst, PAGE_SIZE).fill(0u8);
                    }
                    dst = dst.add(PAGE_SIZE);
                }

                // Last page
                let last_part = (len - first_part) % PAGE_SIZE;
                {
                    if let Some(page) = self.pages[end_page_idx].as_ref() {
                        ptr::copy_nonoverlapping(page.as_ptr(), dst, last_part);
                    } else {
                        std::slice::from_raw_parts_mut(dst, last_part).fill(0u8);
                    }
                }
            }
        }
    }

    // Updates a range of length `len` starting at index `start` with new values.
    // and then writes the new values into the underlying pages,
    // allocating pages (with defaults) if necessary.
    #[inline]
    pub fn set_range_generic(&mut self, start: usize, len: usize, mut new: *const u8) {
        let start_page_idx = start / PAGE_SIZE;
        let end_page_idx = (start + len - 1) / PAGE_SIZE;
        unsafe {
            if start_page_idx == end_page_idx {
                let offset = start % PAGE_SIZE;
                let page = self.pages[start_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                ptr::copy_nonoverlapping(new, page.as_mut_ptr().add(offset), len);
            } else {
                // First page
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                {
                    let start_page =
                        self.pages[start_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(new, start_page.as_mut_ptr().add(offset), first_part);
                    new = new.add(first_part);
                }

                // Middle pages
                for page_idx in (start_page_idx + 1)..end_page_idx {
                    let page = self.pages[page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(new, page.as_mut_ptr(), PAGE_SIZE);
                    new = new.add(PAGE_SIZE);
                }

                // Last page
                let last_part = (len - first_part) % PAGE_SIZE;
                {
                    let end_page =
                        self.pages[end_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(new, end_page.as_mut_ptr(), last_part);
                }
            }
        }
    }

    // Updates a range of length `len` starting at index `start` with new values.
    // It copies the current values into the memory pointed to by `dst`
    // and then writes the new values into the underlying pages,
    // allocating pages (with defaults) if necessary.
    #[inline]
    pub fn replace_range_generic(
        &mut self,
        start: usize,
        len: usize,
        mut new: *const u8,
        mut dst: *mut u8,
    ) {
        let start_page_idx = start / PAGE_SIZE;
        let end_page_idx = (start + len - 1) / PAGE_SIZE;
        unsafe {
            if start_page_idx == end_page_idx {
                let offset = start % PAGE_SIZE;
                let page = self.pages[start_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                ptr::copy_nonoverlapping(page.as_ptr().add(offset), dst, len);
                ptr::copy_nonoverlapping(new, page.as_mut_ptr().add(offset), len);
            } else {
                // First page
                let offset = start % PAGE_SIZE;
                let first_part = PAGE_SIZE - offset;
                {
                    let start_page =
                        self.pages[start_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(start_page.as_ptr().add(offset), dst, first_part);
                    ptr::copy_nonoverlapping(new, start_page.as_mut_ptr().add(offset), first_part);
                    dst = dst.add(first_part);
                    new = new.add(first_part);
                }

                // Middle pages
                for page_idx in (start_page_idx + 1)..end_page_idx {
                    let page = self.pages[page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(page.as_ptr(), dst, PAGE_SIZE);
                    ptr::copy_nonoverlapping(new, page.as_mut_ptr(), PAGE_SIZE);
                    new = new.add(PAGE_SIZE);
                    dst = dst.add(PAGE_SIZE);
                }

                // Last page
                let last_part = (len - first_part) % PAGE_SIZE;
                {
                    let end_page =
                        self.pages[end_page_idx].get_or_insert_with(|| vec![0u8; PAGE_SIZE]);
                    ptr::copy_nonoverlapping(end_page.as_ptr(), dst, last_part);
                    ptr::copy_nonoverlapping(new, end_page.as_mut_ptr(), last_part);
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Implementation for types requiring Default + Clone
impl PagedVec {
    pub fn new(num_pages: usize) -> Self {
        Self {
            pages: vec![None; num_pages],
        }
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Total capacity across available pages, in bytes.
    pub fn bytes_capacity(&self) -> usize {
        self.len().checked_mul(PAGE_SIZE).unwrap()
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
        self.set_range_generic(start, size_of::<BLOCK>(), values as *const _ as *const u8);
    }

    /// memcpy of new `values` into pages, memcpy of old existing values into new returned value.
    /// # Panics
    /// If `from..from + size_of<BLOCK>()` is out of bounds.
    #[inline(always)]
    pub fn replace<BLOCK: Copy>(&mut self, from: usize, values: &BLOCK) -> BLOCK {
        // Create an uninitialized array for old values.
        let mut result: MaybeUninit<BLOCK> = MaybeUninit::uninit();
        self.replace_range_generic(
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

impl PagedVec {
    /// Iterate over [PagedVec] as iterator of elements of type `T`.
    /// Iterator is over `(index, element)` where `index` is the byte index divided by
    /// `size_of::<T>()`.
    ///
    /// `T` must be stack allocated
    pub fn iter<T: Copy>(&self) -> PagedVecIter<'_, T> {
        assert!(size_of::<T>() <= PAGE_SIZE);
        PagedVecIter {
            vec: self,
            current_page: 0,
            current_index_in_page: 0,
            phantom: PhantomData,
        }
    }
}

pub struct PagedVecIter<'a, T> {
    vec: &'a PagedVec,
    current_page: usize,
    current_index_in_page: usize,
    phantom: PhantomData<T>,
}

impl<T: Copy> Iterator for PagedVecIter<'_, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_page < self.vec.len() && self.vec.pages[self.current_page].is_none() {
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
