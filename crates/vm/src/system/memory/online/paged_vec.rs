use std::fmt::Debug;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PagedVec<T> {
    pages: Vec<Option<Box<[T]>>>,
    page_size: usize,
}

unsafe impl<T: Send> Send for PagedVec<T> {}
unsafe impl<T: Sync> Sync for PagedVec<T> {}

impl<T: Copy + Default> PagedVec<T> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize, page_size: usize) -> Self {
        let num_pages = total_size.div_ceil(page_size);
        Self {
            pages: vec![None; num_pages],
            page_size,
        }
    }

    /// Panics if the index is out of bounds. Creates a new page with default values if no page
    /// exists.
    #[inline]
    pub fn get(&mut self, index: usize) -> &T {
        let page_idx = index / self.page_size;
        let offset = index % self.page_size;

        assert!(
            page_idx < self.pages.len(),
            "PagedVec::get index out of bounds: {} >= {}",
            index,
            self.pages.len() * self.page_size
        );

        if self.pages[page_idx].is_none() {
            let page = vec![T::default(); self.page_size];
            self.pages[page_idx] = Some(page.into_boxed_slice());
        }

        unsafe {
            // SAFETY:
            // - We just ensured the page exists and has size `page_size`
            // - offset < page_size by construction
            self.pages
                .get_unchecked(page_idx)
                .as_ref()
                .unwrap()
                .get_unchecked(offset)
        }
    }

    /// Panics if the index is out of bounds. Creates new page before write when necessary.
    #[inline]
    pub fn set(&mut self, index: usize, value: T) {
        let page_idx = index / self.page_size;
        let offset = index % self.page_size;

        assert!(
            page_idx < self.pages.len(),
            "PagedVec::set index out of bounds: {} >= {}",
            index,
            self.pages.len() * self.page_size
        );

        if let Some(page) = &mut self.pages[page_idx] {
            // SAFETY:
            // - If page exists, then it has size `page_size`
            unsafe {
                *page.get_unchecked_mut(offset) = value;
            }
        } else {
            let mut page = vec![T::default(); self.page_size];
            page[offset] = value;
            self.pages[page_idx] = Some(page.into_boxed_slice());
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
                        .map(move |(offset, &value)| (page_idx * self.page_size + offset, value))
                })
            })
            .flatten()
    }
}
