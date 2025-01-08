use std::array;

#[derive(Debug, Clone)]
pub(crate) struct PagedVec<T> {
    page_size: u64,
    pages: Vec<Option<Vec<T>>>,
}

impl<T: Default + Copy> PagedVec<T> {
    pub fn new(page_size: usize, num_pages: usize) -> Self {
        Self {
            page_size: page_size as u64,
            pages: vec![None; num_pages],
        }
    }

    pub fn get(&self, index: u64) -> T {
        let page_idx = (index / self.page_size) as usize;
        if let Some(page) = &self.pages[page_idx] {
            page[(index % self.page_size) as usize]
        } else {
            T::default()
        }
    }

    pub fn set(&mut self, index: u64, value: T) {
        let page_idx = (index / self.page_size) as usize;
        if let Some(page) = self.pages[page_idx].as_mut() {
            page[(index % self.page_size) as usize] = value;
        } else {
            let page = self.pages[page_idx]
                .get_or_insert_with(|| vec![T::default(); self.page_size as usize]);
            page[(index % self.page_size) as usize] = value;
        }
    }

    pub fn get_range<const N: usize>(&self, index: u64) -> [T; N] {
        let start_page_idx = (index / self.page_size) as usize;
        let end_page_idx = ((index + N as u64) / self.page_size) as usize;

        if start_page_idx == end_page_idx {
            if let Some(start_page) = &self.pages[start_page_idx] {
                let i = (index % self.page_size) as usize;
                start_page[i..i + N]
                    .try_into()
                    .expect("Slice length matches const generic N")
            } else {
                [T::default(); N]
            }
        } else {
            // TODO: This can be more efficient by copying from two slices (but most queries should
            // not be cross-page).
            array::from_fn(|i| self.get(index + i as u64))
        }
    }

    pub fn set_range<const N: usize>(&mut self, index: u64, values: [T; N]) {
        let start_page_idx = (index / self.page_size) as usize;
        let end_page_idx = ((index + N as u64) / self.page_size) as usize;

        if start_page_idx == end_page_idx {
            let page = self.pages[start_page_idx]
                .get_or_insert_with(|| vec![T::default(); self.page_size as usize]);
            let i = (index % self.page_size) as usize;
            page[i..i + N].copy_from_slice(&values);
        } else {
            // TODO: This can be more efficient by copying into two slices (but most queries should
            // not be cross-page).
            for i in 0..N {
                self.set(index + i as u64, values[i]);
            }
        }
    }

    pub fn iter(&self) -> PagedVecIter<'_, T> {
        PagedVecIter {
            vec: self,
            current_page: 0,
            current_index_in_page: 0,
        }
    }
}

pub struct PagedVecIter<'a, T> {
    vec: &'a PagedVec<T>,
    current_page: usize,
    current_index_in_page: usize,
}

impl<'a, T: Default + Copy> Iterator for PagedVecIter<'a, T> {
    type Item = (u64, T);

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
        let global_index =
            (self.current_page as u64 * self.vec.page_size) + self.current_index_in_page as u64;

        let page = self.vec.pages[self.current_page].as_ref()?;
        let value = page[self.current_index_in_page];

        self.current_index_in_page += 1;
        if self.current_index_in_page == self.vec.page_size as usize {
            self.current_page += 1;
            self.current_index_in_page = 0;
        }
        Some((global_index, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let mut v = PagedVec::new(4, 3);
        assert_eq!(v.get(0), 0);
        v.set(0, 42);
        assert_eq!(v.get(0), 42);
    }

    #[test]
    fn test_cross_page_operations() {
        let mut v = PagedVec::new(4, 3);
        v.set(3, 10); // Last element of first page
        v.set(4, 20); // First element of second page
        assert_eq!(v.get(3), 10);
        assert_eq!(v.get(4), 20);
    }

    #[test]
    fn test_page_boundaries() {
        let mut v = PagedVec::new(4, 2);
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
        assert_eq!(v.get_range::<8>(0), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_range_cross_page_boundary() {
        let mut v = PagedVec::new(4, 2);
        v.set_range::<6>(2, [10, 11, 12, 13, 14, 15]);
        assert_eq!(v.get_range::<6>(2), [10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_large_indices() {
        let mut v = PagedVec::new(4, 100);
        let large_index = 399;
        v.set(large_index, 42);
        assert_eq!(v.get(large_index), 42);
    }

    #[test]
    fn test_range_operations_with_defaults() {
        let mut v = PagedVec::new(4, 3);
        v.set(2, 5);
        v.set(5, 10);

        // Should include both set values and defaults
        assert_eq!(v.get_range::<6>(1), [0, 5, 0, 0, 10, 0]);
    }

    #[test]
    fn test_non_zero_default_type() {
        let mut v: PagedVec<bool> = PagedVec::new(4, 2);
        assert_eq!(v.get(0), false); // bool's default
        v.set(0, true);
        assert_eq!(v.get(0), true);
        assert_eq!(v.get(1), false);
    }

    #[test]
    fn test_set_range_overlapping_pages() {
        let mut v = PagedVec::new(4, 3);
        let test_data = [1, 2, 3, 4, 5, 6];
        v.set_range::<6>(2, test_data);

        // Verify first page
        assert_eq!(v.get(2), 1);
        assert_eq!(v.get(3), 2);

        // Verify second page
        assert_eq!(v.get(4), 3);
        assert_eq!(v.get(5), 4);
        assert_eq!(v.get(6), 5);
        assert_eq!(v.get(7), 6);
    }

    #[test]
    fn test_overlapping_set_ranges() {
        let mut v = PagedVec::new(4, 3);

        // Initial set_range
        v.set_range::<5>(0, [1, 2, 3, 4, 5]);
        assert_eq!(v.get_range::<5>(0), [1, 2, 3, 4, 5]);

        // Overlap from beginning
        v.set_range::<3>(0, [10, 20, 30]);
        assert_eq!(v.get_range::<5>(0), [10, 20, 30, 4, 5]);

        // Overlap in middle
        v.set_range::<2>(2, [42, 43]);
        assert_eq!(v.get_range::<5>(0), [10, 20, 42, 43, 5]);

        // Overlap at end
        v.set_range::<2>(4, [91, 92]);
        assert_eq!(v.get_range::<6>(0), [10, 20, 42, 43, 91, 92]);
    }

    #[test]
    fn test_overlapping_set_ranges_cross_pages() {
        let mut v = PagedVec::new(4, 3);

        // Fill across first two pages
        v.set_range::<8>(0, [1, 2, 3, 4, 5, 6, 7, 8]);

        // Overlap end of first page and start of second
        v.set_range::<4>(2, [21, 22, 23, 24]);
        assert_eq!(v.get_range::<8>(0), [1, 2, 21, 22, 23, 24, 7, 8]);

        // Overlap multiple pages
        v.set_range::<6>(1, [31, 32, 33, 34, 35, 36]);
        assert_eq!(v.get_range::<8>(0), [1, 31, 32, 33, 34, 35, 36, 8]);
    }

    #[test]
    fn test_iterator() {
        let mut v = PagedVec::new(4, 3);

        v.set_range::<6>(4, [1, 2, 3, 4, 5, 6]);
        let contents: Vec<_> = v.iter().collect();
        assert_eq!(contents.len(), 8); // two pages

        for i in 0..6 {
            assert_eq!(contents[i], (4 + i as u64, 1 + i));
        }
        assert_eq!(contents[6], (10, 0));
        assert_eq!(contents[7], (11, 0));
    }
}
