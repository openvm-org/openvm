use std::fmt::Debug;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PagedVec<T, const PAGE_SIZE: usize> {
    pages: Vec<Option<Box<[T; PAGE_SIZE]>>>,
}

unsafe impl<T: Send, const PAGE_SIZE: usize> Send for PagedVec<T, PAGE_SIZE> {}
unsafe impl<T: Sync, const PAGE_SIZE: usize> Sync for PagedVec<T, PAGE_SIZE> {}

impl<T: Copy + Default, const PAGE_SIZE: usize> PagedVec<T, PAGE_SIZE> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        let num_pages = total_size.div_ceil(PAGE_SIZE);
        Self {
            pages: vec![None; num_pages],
        }
    }

    #[cold]
    #[inline(never)]
    fn create_zeroed_page() -> Box<[T; PAGE_SIZE]> {
        // SAFETY:
        // - layout is valid since PAGE_SIZE is non-zero
        // - alloc_zeroed returns properly aligned memory for T
        // - Box::from_raw takes ownership of the allocated memory
        unsafe {
            let layout = std::alloc::Layout::array::<T>(PAGE_SIZE).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut [T; PAGE_SIZE];
            Box::from_raw(ptr)
        }
    }

    /// Get value at index without allocating new pages.
    /// Panics if index is out of bounds. Returns default value if page doesn't exist.
    #[inline]
    pub fn get(&self, index: usize) -> T {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        // SAFETY:
        // - offset < PAGE_SIZE by construction (from modulo operation)
        // - page exists when as_ref() returns Some
        self.pages[page_idx]
            .as_ref()
            .map(|page| unsafe { *page.get_unchecked(offset) })
            .unwrap_or_default()
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        let page_idx = index / PAGE_SIZE;
        let offset = index % PAGE_SIZE;

        let page = self.pages[page_idx].get_or_insert_with(Self::create_zeroed_page);

        // SAFETY: offset < PAGE_SIZE by construction
        unsafe {
            *page.get_unchecked_mut(offset) = value;
        }
    }

    pub fn par_iter(&self) -> impl ParallelIterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .par_iter()
            .enumerate()
            .filter_map(move |(page_idx, page)| {
                page.as_ref().map(move |p| {
                    p.par_iter()
                        .enumerate()
                        .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value))
                })
            })
            .flatten()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, T)> + '_
    where
        T: Send + Sync,
    {
        self.pages
            .iter()
            .enumerate()
            .filter_map(move |(page_idx, page)| {
                page.as_ref().map(move |p| {
                    p.iter()
                        .enumerate()
                        .map(move |(offset, &value)| (page_idx * PAGE_SIZE + offset, value))
                })
            })
            .flatten()
    }
}
