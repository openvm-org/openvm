use std::fmt::Debug;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;

use crate::system::memory::online::{PAGE_SIZE, PAGE_SIZE_LOG2};

#[derive(Debug, Clone)]
pub struct PagedVec<T> {
    pages: Vec<Option<Box<[T]>>>,
}

unsafe impl<T: Send> Send for PagedVec<T> {}
unsafe impl<T: Sync> Sync for PagedVec<T> {}

impl<T: Copy + Default> PagedVec<T> {
    #[inline]
    /// `total_size` is the capacity of elements of type `T`.
    pub fn new(total_size: usize) -> Self {
        let num_pages = total_size.div_ceil(PAGE_SIZE);
        Self {
            pages: vec![Some(vec![T::default(); PAGE_SIZE].into_boxed_slice()); num_pages],
        }
    }

    /// Panics if the index is out of bounds. Creates a new page with default values if no page
    /// exists.
    #[inline]
    pub fn get(&mut self, index: usize) -> &T {
        let page_idx = index >> PAGE_SIZE_LOG2;
        let offset = index & (PAGE_SIZE - 1);

        assert!(
            page_idx < self.pages.len(),
            "PagedVec::get index out of bounds: {} >= {}",
            index,
            self.pages.len() << PAGE_SIZE_LOG2
        );

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
        let page_idx = index >> PAGE_SIZE_LOG2;
        let offset = index & (PAGE_SIZE - 1);

        assert!(
            page_idx < self.pages.len(),
            "PagedVec::set index out of bounds: {} >= {}",
            index,
            self.pages.len() << PAGE_SIZE_LOG2
        );

        let page = self.pages[page_idx].as_mut().unwrap();
        // SAFETY:
        // - If page exists, then it has size `page_size`
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
