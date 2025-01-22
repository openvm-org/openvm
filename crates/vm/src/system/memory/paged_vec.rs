use std::{mem::MaybeUninit, ops::Range, ptr};

use serde::{Deserialize, Serialize};

use crate::arch::MemoryConfig;

/// (address_space, pointer)
pub(crate) type Address = (u32, u32);
pub(crate) const PAGE_SIZE: usize = 1 << 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PagedVec<T, const PAGE_SIZE: usize> {
    pages: Vec<Option<Vec<T>>>,
}

impl<T: Default + Clone, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    pub fn new(num_pages: usize) -> Self {
        Self {
            pages: vec![None; num_pages],
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let page_idx = index / PAGE_SIZE;
        self.pages[page_idx]
            .as_ref()
            .map(|page| &page[index % PAGE_SIZE])
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let page_idx = index / PAGE_SIZE;
        self.pages[page_idx]
            .as_mut()
            .map(|page| &mut page[index % PAGE_SIZE])
    }

    pub fn set(&mut self, index: usize, value: T) -> Option<T> {
        let page_idx = index / PAGE_SIZE;
        if let Some(page) = self.pages[page_idx].as_mut() {
            Some(std::mem::replace(&mut page[index % PAGE_SIZE], value))
        } else {
            let page = self.pages[page_idx].get_or_insert_with(|| vec![T::default(); PAGE_SIZE]);
            page[index % PAGE_SIZE] = value;
            None
        }
    }

    #[inline(always)]
    pub fn range_vec(&self, range: Range<usize>) -> Vec<T> {
        let mut result = Vec::with_capacity(range.len());
        for page_idx in (range.start / PAGE_SIZE)..range.end.div_ceil(PAGE_SIZE) {
            let in_page_start = range.start.saturating_sub(page_idx * PAGE_SIZE);
            let in_page_end = (range.end - page_idx * PAGE_SIZE).min(PAGE_SIZE);
            if let Some(page) = self.pages[page_idx].as_ref() {
                result.extend(page[in_page_start..in_page_end].iter().cloned());
            } else {
                result.extend(vec![T::default(); in_page_end - in_page_start]);
            }
        }
        result
    }

    pub fn set_range(&mut self, range: Range<usize>, values: &[T]) -> Vec<T> {
        let mut result = Vec::with_capacity(range.len());
        let mut values = values.iter();
        for page_idx in (range.start / PAGE_SIZE)..range.end.div_ceil(PAGE_SIZE) {
            let in_page_start = range.start.saturating_sub(page_idx * PAGE_SIZE);
            let in_page_end = (range.end - page_idx * PAGE_SIZE).min(PAGE_SIZE);
            let page = self.pages[page_idx].get_or_insert_with(|| vec![T::default(); PAGE_SIZE]);
            result.extend(
                page[in_page_start..in_page_end]
                    .iter_mut()
                    .map(|x| std::mem::replace(x, values.next().unwrap().clone())),
            );
        }
        result
    }

    pub fn memory_size(&self) -> usize {
        self.pages.len() * PAGE_SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.pages.iter().all(|page| page.is_none())
    }
}

impl<T: Default + Copy, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    #[inline(always)]
    pub fn range_array<const N: usize>(&self, from: usize) -> [T; N] {
        // Step 1: Create an uninitialized array of MaybeUninit<T>
        let mut result: [MaybeUninit<T>; N] = unsafe {
            // SAFETY: An uninitialized `[MaybeUninit<T>; N]` is valid.
            MaybeUninit::uninit().assume_init()
        };

        // Step 2: Get a mutable slice of T references from the MaybeUninit array.
        let result_slice = unsafe {
            // SAFETY: We are converting a pointer from MaybeUninit<T> to T.
            // This is safe because we will fully initialize every element before reading.
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut T, N)
        };

        let start_page = from / PAGE_SIZE;
        let end_page = (from + N - 1) / PAGE_SIZE;

        if start_page == end_page {
            if let Some(page) = self.pages[start_page].as_ref() {
                // Copy data from the page into result_slice.
                let page_offset = from - start_page * PAGE_SIZE;
                let src = &page[page_offset..page_offset + N];
                result_slice.copy_from_slice(src);
            } else {
                // If no page available, fill with default values.
                result_slice.fill(T::default());
            }
        } else {
            debug_assert!(start_page + 1 == end_page);
            let first_part = PAGE_SIZE - (from - start_page * PAGE_SIZE);
            if let Some(page) = self.pages[start_page].as_ref() {
                let page_offset = from - start_page * PAGE_SIZE;
                let src = &page[page_offset..];
                result_slice[..first_part].copy_from_slice(src);
            } else {
                result_slice[..first_part].fill(T::default());
            }
            if let Some(page) = self.pages[end_page].as_ref() {
                let second_part = N - first_part;
                let src = &page[0..second_part];
                result_slice[first_part..].copy_from_slice(src);
            } else {
                result_slice[first_part..].fill(T::default());
            }
        }

        // Step 4: Convert the fully initialized array of MaybeUninit<T> to [T; N].
        // SAFETY: We have initialized every element of `result`.
        unsafe {
            // Transmute the array; at this point each element is initialized.
            ptr::read(&result as *const _ as *const [T; N])
        }
    }

    #[inline(always)]
    pub fn set_range_array<const N: usize>(&mut self, from: usize, values: &[T; N]) -> [T; N] {
        // Step 1: Create an uninitialized array for old values.
        let mut result: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        let result_slice = unsafe {
            // SAFETY: We will fully initialize `result_slice` before reading.
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut T, N)
        };

        let start_page = from / PAGE_SIZE;
        let end_page = (from + N - 1) / PAGE_SIZE;

        if start_page == end_page {
            // Ensure the page exists; if not, allocate and fill with defaults.
            if self.pages[start_page].is_none() {
                self.pages[start_page] = Some(vec![T::default(); PAGE_SIZE]);
            }
            let page = self.pages[start_page].as_mut().unwrap();
            let page_offset = from - start_page * PAGE_SIZE;

            // Copy old values from the page into result_slice.
            result_slice.copy_from_slice(&page[page_offset..page_offset + N]);
            // Write the new values from input into the page.
            page[page_offset..page_offset + N].copy_from_slice(&values[..]);
        } else {
            debug_assert!(start_page + 1 == end_page);
            let first_part = PAGE_SIZE - (from - start_page * PAGE_SIZE);

            // Handle the first page.
            if self.pages[start_page].is_none() {
                self.pages[start_page] = Some(vec![T::default(); PAGE_SIZE]);
            }
            let page0 = self.pages[start_page].as_mut().unwrap();
            let page_offset = from - start_page * PAGE_SIZE;

            result_slice[..first_part].copy_from_slice(&page0[page_offset..]);
            page0[page_offset..].copy_from_slice(&values[..first_part]);

            // Handle the second page.
            let second_part = N - first_part;
            if self.pages[end_page].is_none() {
                self.pages[end_page] = Some(vec![T::default(); PAGE_SIZE]);
            }
            let page1 = self.pages[end_page].as_mut().unwrap();

            result_slice[first_part..].copy_from_slice(&page1[0..second_part]);
            page1[0..second_part].copy_from_slice(&values[first_part..]);
        }

        // Step 4: Convert the fully initialized result array to [T; N].
        unsafe { ptr::read(&result as *const _ as *const [T; N]) }
    }
}

impl<T, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    pub fn iter(&self) -> PagedVecIter<'_, T, PAGE_SIZE> {
        PagedVecIter {
            vec: self,
            current_page: 0,
            current_index_in_page: 0,
        }
    }
}

pub struct PagedVecIter<'a, T, const PAGE_SIZE: usize> {
    vec: &'a PagedVec<T, PAGE_SIZE>,
    current_page: usize,
    current_index_in_page: usize,
}

impl<T: Clone, const PAGE_SIZE: usize> Iterator for PagedVecIter<'_, T, PAGE_SIZE> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_page < self.vec.pages.len()
            && self.vec.pages[self.current_page].is_none()
        {
            self.current_page += 1;
            debug_assert_eq!(self.current_index_in_page, 0);
            self.current_index_in_page = 0;
        }
        if self.current_page >= self.vec.pages.len() {
            return None;
        }
        let global_index = self.current_page * PAGE_SIZE + self.current_index_in_page;

        let page = self.vec.pages[self.current_page].as_ref()?;
        let value = page[self.current_index_in_page].clone();

        self.current_index_in_page += 1;
        if self.current_index_in_page == PAGE_SIZE {
            self.current_page += 1;
            self.current_index_in_page = 0;
        }
        Some((global_index, value))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddressMap<T, const PAGE_SIZE: usize> {
    paged_vecs: Vec<PagedVec<T, PAGE_SIZE>>,
    as_offset: u32,
}

impl<T: Clone + Default, const PAGE_SIZE: usize> Default for AddressMap<T, PAGE_SIZE> {
    fn default() -> Self {
        Self::from_mem_config(&MemoryConfig::default())
    }
}

impl<T: Clone + Default, const PAGE_SIZE: usize> AddressMap<T, PAGE_SIZE> {
    pub fn new(as_offset: u32, as_cnt: usize, mem_size: usize) -> Self {
        Self {
            paged_vecs: vec![PagedVec::new(mem_size.div_ceil(PAGE_SIZE)); as_cnt],
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
    pub fn items(&self) -> impl Iterator<Item = (Address, T)> + '_ {
        self.paged_vecs
            .iter()
            .enumerate()
            .flat_map(move |(as_idx, page)| {
                page.iter()
                    .map(move |(ptr_idx, x)| ((as_idx as u32 + self.as_offset, ptr_idx as u32), x))
            })
    }
    pub fn get(&self, address: &Address) -> Option<&T> {
        self.paged_vecs[(address.0 - self.as_offset) as usize].get(address.1 as usize)
    }
    pub fn get_mut(&mut self, address: &Address) -> Option<&mut T> {
        self.paged_vecs[(address.0 - self.as_offset) as usize].get_mut(address.1 as usize)
    }
    pub fn insert(&mut self, address: &Address, data: T) -> Option<T> {
        self.paged_vecs[(address.0 - self.as_offset) as usize].set(address.1 as usize, data)
    }
    pub fn is_empty(&self) -> bool {
        self.paged_vecs.iter().all(|page| page.is_empty())
    }

    pub fn from_iter(
        as_offset: u32,
        as_cnt: usize,
        mem_size: usize,
        iter: impl IntoIterator<Item = (Address, T)>,
    ) -> Self {
        let mut vec = Self::new(as_offset, as_cnt, mem_size);
        for (address, data) in iter {
            vec.insert(&address, data);
        }
        vec
    }
}

impl<T: Copy + Default, const PAGE_SIZE: usize> AddressMap<T, PAGE_SIZE> {
    pub fn get_range<const N: usize>(&self, address: &Address) -> [T; N] {
        self.paged_vecs[(address.0 - self.as_offset) as usize].range_array(address.1 as usize)
    }
    pub fn set_range<const N: usize>(&mut self, address: &Address, values: &[T; N]) -> [T; N] {
        self.paged_vecs[(address.0 - self.as_offset) as usize]
            .set_range_array(address.1 as usize, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let mut v = PagedVec::<_, 4>::new(3);
        assert_eq!(v.get(0), None);
        v.set(0, 42);
        assert_eq!(v.get(0), Some(&42));
    }

    #[test]
    fn test_cross_page_operations() {
        let mut v = PagedVec::<_, 4>::new(3);
        v.set(3, 10); // Last element of first page
        v.set(4, 20); // First element of second page
        assert_eq!(v.get(3), Some(&10));
        assert_eq!(v.get(4), Some(&20));
    }

    #[test]
    fn test_page_boundaries() {
        let mut v = PagedVec::<_, 4>::new(2);
        // Fill first page
        v.set(0, 1);
        v.set(1, 2);
        v.set(2, 3);
        v.set(3, 4);
        // Fill second page
        v.set(4, 5);
        v.set(5, 6);
        v.set(6, 7);
        v.set(7, 8);

        // Verify all values
        assert_eq!(v.range_vec(0..8), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_range_cross_page_boundary() {
        let mut v = PagedVec::<_, 4>::new(2);
        v.set_range(2..8, &[10, 11, 12, 13, 14, 15]);
        assert_eq!(v.range_vec(2..8), [10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_large_indices() {
        let mut v = PagedVec::<_, 4>::new(100);
        let large_index = 399;
        v.set(large_index, 42);
        assert_eq!(v.get(large_index), Some(&42));
    }

    #[test]
    fn test_range_operations_with_defaults() {
        let mut v = PagedVec::<_, 4>::new(3);
        v.set(2, 5);
        v.set(5, 10);

        // Should include both set values and defaults
        assert_eq!(v.range_vec(1..7), [0, 5, 0, 0, 10, 0]);
    }

    #[test]
    fn test_non_zero_default_type() {
        let mut v: PagedVec<bool, 4> = PagedVec::new(2);
        assert_eq!(v.get(0), None); // bool's default
        v.set(0, true);
        assert_eq!(v.get(0), Some(&true));
        assert_eq!(v.get(1), Some(&false)); // because we created the page
    }

    #[test]
    fn test_set_range_overlapping_pages() {
        let mut v = PagedVec::<_, 4>::new(3);
        let test_data = [1, 2, 3, 4, 5, 6];
        v.set_range(2..8, &test_data);

        // Verify first page
        assert_eq!(v.get(2), Some(&1));
        assert_eq!(v.get(3), Some(&2));

        // Verify second page
        assert_eq!(v.get(4), Some(&3));
        assert_eq!(v.get(5), Some(&4));
        assert_eq!(v.get(6), Some(&5));
        assert_eq!(v.get(7), Some(&6));
    }

    #[test]
    fn test_overlapping_set_ranges() {
        let mut v = PagedVec::<_, 4>::new(3);

        // Initial set_range
        v.set_range(0..5, &[1, 2, 3, 4, 5]);
        assert_eq!(v.range_vec(0..5), [1, 2, 3, 4, 5]);

        // Overlap from beginning
        v.set_range(0..3, &[10, 20, 30]);
        assert_eq!(v.range_vec(0..5), [10, 20, 30, 4, 5]);

        // Overlap in middle
        v.set_range(2..4, &[42, 43]);
        assert_eq!(v.range_vec(0..5), [10, 20, 42, 43, 5]);

        // Overlap at end
        v.set_range(4..6, &[91, 92]);
        assert_eq!(v.range_vec(0..6), [10, 20, 42, 43, 91, 92]);
    }

    #[test]
    fn test_overlapping_set_ranges_cross_pages() {
        let mut v = PagedVec::<_, 4>::new(3);

        // Fill across first two pages
        v.set_range(0..8, &[1, 2, 3, 4, 5, 6, 7, 8]);

        // Overlap end of first page and start of second
        v.set_range(2..6, &[21, 22, 23, 24]);
        assert_eq!(v.range_vec(0..8), [1, 2, 21, 22, 23, 24, 7, 8]);

        // Overlap multiple pages
        v.set_range(1..7, &[31, 32, 33, 34, 35, 36]);
        assert_eq!(v.range_vec(0..8), [1, 31, 32, 33, 34, 35, 36, 8]);
    }

    #[test]
    fn test_iterator() {
        let mut v = PagedVec::<_, 4>::new(3);

        v.set_range(4..10, &[1, 2, 3, 4, 5, 6]);
        let contents: Vec<_> = v.iter().collect();
        assert_eq!(contents.len(), 8); // two pages

        contents
            .iter()
            .take(6)
            .enumerate()
            .for_each(|(i, &(idx, val))| {
                assert_eq!((idx, val), (4 + i, 1 + i));
            });
        assert_eq!(contents[6], (10, 0));
        assert_eq!(contents[7], (11, 0));
    }
}
