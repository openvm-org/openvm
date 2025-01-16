use std::ops::Range;

pub(crate) const PAGE_SIZE: usize = 1 << 12;

#[derive(Debug, Clone)]
pub(crate) struct PagedVec<T, const SIZE: usize> {
    pages: Vec<Option<Vec<Option<T>>>>,
}

impl<T: Copy, const SIZE: usize> PagedVec<T, SIZE> {
    pub fn new(num_pages: usize) -> Self {
        Self {
            pages: vec![None; num_pages],
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let page = self.pages[index / SIZE].as_ref()?;
        let value = unsafe { page.get_unchecked(index % SIZE) };
        value.as_ref()
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let page_idx = index / SIZE;
        if let Some(page) = self.pages[page_idx].as_mut() {
            page[index % SIZE].as_mut()
        } else {
            None
        }
    }

    pub fn insert(&mut self, index: usize, value: T) -> Option<T> {
        let page_idx = index / SIZE;
        let page = self.pages[page_idx].get_or_insert_with(|| vec![None; SIZE]);
        unsafe { page.get_unchecked_mut(index % SIZE).replace(value) }
    }
}

impl<T: Default + Copy, const SIZE: usize> PagedVec<T, SIZE> {
    pub fn get_range(&self, range: Range<usize>) -> Vec<T> {
        let mut result = Vec::with_capacity(range.len());
        for page_idx in (range.start / SIZE)..range.end.div_ceil(SIZE) {
            let in_page_start = range.start.saturating_sub(page_idx * SIZE);
            let in_page_end = (range.end - page_idx * SIZE).min(SIZE);
            if let Some(page) = self.pages[page_idx].as_ref() {
                result.extend(
                    page[in_page_start..in_page_end]
                        .iter()
                        .map(|&x| x.unwrap_or_default()),
                );
            } else {
                result.extend(vec![T::default(); in_page_end - in_page_start]);
            }
        }
        result
    }

    pub fn set_range<'a>(
        &mut self,
        range: Range<usize>,
        values: impl IntoIterator<Item = &'a T>,
    ) -> Vec<T>
    where
        T: 'a,
    {
        let mut result = Vec::with_capacity(range.len());
        let mut values = values.into_iter();
        for page_idx in (range.start / SIZE)..range.end.div_ceil(SIZE) {
            let in_page_start = range.start.saturating_sub(page_idx * SIZE);
            let in_page_end = (range.end - page_idx * SIZE).min(SIZE);
            let page = self.pages[page_idx].get_or_insert_with(|| vec![None; SIZE]);
            result.extend(
                page[in_page_start..in_page_end]
                    .iter_mut()
                    .map(|x| x.replace(*values.next().unwrap()).unwrap_or_default())
                    .collect::<Vec<_>>(),
            );
        }
        result
    }
}

impl<T, const SIZE: usize> PagedVec<T, SIZE> {
    pub fn iter(&self) -> PagedVecIter<'_, T, SIZE> {
        PagedVecIter {
            vec: self,
            current_page: 0,
            current_index_in_page: 0,
        }
    }
}

pub struct PagedVecIter<'a, T, const SIZE: usize> {
    vec: &'a PagedVec<T, SIZE>,
    current_page: usize,
    current_index_in_page: usize,
}

impl<T: Copy, const SIZE: usize> Iterator for PagedVecIter<'_, T, SIZE> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_page < self.vec.pages.len() {
            if let Some(page) = self.vec.pages[self.current_page].as_ref() {
                while self.current_index_in_page < page.len()
                    && page[self.current_index_in_page].is_none()
                {
                    self.current_index_in_page += 1;
                }
                if self.current_index_in_page < page.len() {
                    break;
                }
            }
            self.current_page += 1;
            self.current_index_in_page = 0;
        }

        if self.current_page >= self.vec.pages.len() {
            None
        } else {
            let global_index = self.current_page * SIZE + self.current_index_in_page;
            let page = self.vec.pages[self.current_page].as_ref().unwrap();
            let value = page[self.current_index_in_page].unwrap();
            self.current_index_in_page += 1;
            Some((global_index, value))
        }
    }
}

/// (address_space, pointer)
pub(crate) type Address = (u32, u32);

#[derive(Debug, Clone)]
pub struct AddressMap<T> {
    paged_vecs: Vec<PagedVec<T, PAGE_SIZE>>,
    as_offset: u32,
}

impl<T: Copy> AddressMap<T> {
    pub fn new(as_offset: u32, as_cnt: usize, mem_size: usize) -> Self {
        let mem_size = mem_size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        Self {
            paged_vecs: vec![PagedVec::<T, PAGE_SIZE>::new(mem_size / PAGE_SIZE); as_cnt],
            as_offset,
        }
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

    pub fn insert(&mut self, address: Address, data: T) -> Option<T> {
        self.paged_vecs[(address.0 - self.as_offset) as usize].insert(address.1 as usize, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let mut v = PagedVec::<_, 4>::new(3);
        assert_eq!(v.get(0), None);
        v.insert(0, 42);
        assert_eq!(v.get(0), Some(&42));
    }

    #[test]
    fn test_cross_page_operations() {
        let mut v = PagedVec::<_, 4>::new(3);
        v.insert(3, 10); // Last element of first page
        v.insert(4, 20); // First element of second page
        assert_eq!(v.get(3), Some(&10));
        assert_eq!(v.get(4), Some(&20));
    }

    #[test]
    fn test_page_boundaries() {
        let mut v = PagedVec::<_, 4>::new(2);
        // Fill first page
        v.insert(0, 1);
        v.insert(1, 2);
        v.insert(2, 3);
        v.insert(3, 4);
        // Fill second page
        v.insert(4, 5);
        v.insert(5, 6);
        v.insert(6, 7);
        v.insert(7, 8);

        // Verify all values
        assert_eq!(v.get_range(0..8), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_range_cross_page_boundary() {
        let mut v = PagedVec::<_, 4>::new(2);
        v.set_range(2..8, &[10, 11, 12, 13, 14, 15]);
        assert_eq!(v.get_range(2..8), [10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_large_indices() {
        let mut v = PagedVec::<_, 4>::new(100);
        let large_index = 399;
        v.insert(large_index, 42);
        assert_eq!(v.get(large_index), Some(&42));
    }

    #[test]
    fn test_range_operations_with_defaults() {
        let mut v = PagedVec::<_, 4>::new(3);
        v.insert(2, 5);
        v.insert(5, 10);

        // Should include both insert values and defaults
        assert_eq!(v.get_range(1..7), [0, 5, 0, 0, 10, 0]);
    }

    #[test]
    fn test_non_zero_default_type() {
        let mut v: PagedVec<bool, 4> = PagedVec::new(2);
        assert_eq!(v.get(0), None); // bool's default
        v.insert(0, true);
        assert_eq!(v.get(0), Some(&true));
        assert_eq!(v.get(1), None);
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
        assert_eq!(v.get_range(0..5), [1, 2, 3, 4, 5]);

        // Overlap from beginning
        v.set_range(0..3, &[10, 20, 30]);
        assert_eq!(v.get_range(0..5), [10, 20, 30, 4, 5]);

        // Overlap in middle
        v.set_range(2..4, &[42, 43]);
        assert_eq!(v.get_range(0..5), [10, 20, 42, 43, 5]);

        // Overlap at end
        v.set_range(4..6, &[91, 92]);
        assert_eq!(v.get_range(0..6), [10, 20, 42, 43, 91, 92]);
    }

    #[test]
    fn test_overlapping_set_ranges_cross_pages() {
        let mut v = PagedVec::<_, 4>::new(3);

        // Fill across first two pages
        v.set_range(0..8, &[1, 2, 3, 4, 5, 6, 7, 8]);

        // Overlap end of first page and start of second
        v.set_range(2..6, &[21, 22, 23, 24]);
        assert_eq!(v.get_range(0..8), [1, 2, 21, 22, 23, 24, 7, 8]);

        // Overlap multiple pages
        v.set_range(1..7, &[31, 32, 33, 34, 35, 36]);
        assert_eq!(v.get_range(0..8), [1, 31, 32, 33, 34, 35, 36, 8]);
    }

    #[test]
    fn test_iterator() {
        let mut v = PagedVec::<_, 4>::new(3);

        v.set_range(4..10, &[1, 2, 3, 4, 5, 6]);
        let contents: Vec<_> = v.iter().collect();
        assert_eq!(contents.len(), 6);

        contents
            .iter()
            .take(6)
            .enumerate()
            .for_each(|(i, &(idx, val))| {
                assert_eq!((idx, val), (4 + i, 1 + i));
            });
    }
}
